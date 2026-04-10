"""
SVGDataset — data loader for ArchCAD-400k.

Loads JSON + PNG samples from the ArchCAD HuggingFace dataset and produces
batches matching the SVGNet.forward() signature.

Batch format expected by SVGNet.forward():
    coords       — [N, 3] float     (x_norm, y_norm, 0)
    feats        — [N, 10] float    (angle, length, R, G, B, is_line, is_arc, is_circle, is_ellipse, stroke_width)
    sem_labels   — [N, 2] long      (semantic_id, instance_int_id)
    offsets      — [B] int          cumulative point counts per sample
    lengths      — [B] float        number of primitives per sample (pre-pad)
    layerIds     — [N] long         layer group IDs (not used by DPSS, set to 0)
    imgs         — list[Tensor]     [B x (3, H, W)] rasterized tile images
    centers      — list[Tensor]     [B x (N_i, 2)] normalized primitive centers in [-1, 1]
    json_files   — list[str]        file paths
"""

import json
import math
import os
import os.path as osp
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Semantic label mapping from ArchCAD HuggingFace README
SEMANTIC_CLASSES = {
    0: "axis_grid",
    1: "single_door",
    2: "double_door",
    3: "parent_child_door",
    4: "other_door",
    5: "elevator",
    6: "staircase",
    7: "sink",
    8: "urinal",
    9: "toilet",
    10: "bathtub",
    11: "squat_toilet",
    12: "other_fixtures",
    13: "drain",
    14: "table",
    15: "chair",
    16: "bed",
    17: "sofa",
    18: "hole",
    19: "glass",
    20: "wall",
    21: "concrete_column",
    22: "steel_column",
    23: "concrete_beam",
    24: "steel_beam",
    25: "parking_space",
    26: "foundation",
    27: "pile",
    28: "rebar",
    29: "fire_hydrant",
}
NUM_CLASSES = 30  # 0-29
BG_LABEL = 100  # "Others" in the dataset, mapped to 30 internally

# Thing (countable) vs Stuff (uncountable)
THING_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 25, 27, 29]
STUFF_IDS = [0, 12, 13, 19, 20, 23, 24, 26, 28]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _compute_primitive_features(entity: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute center (x,y) and 10D feature vector for a primitive.

    Returns:
        center: [2] array (x, y) in drawing coordinates
        feats:  [10] array (angle, length, R, G, B, is_line, is_arc, is_circle, is_ellipse, stroke_width)
    """
    etype = entity["type"]
    rgb = entity.get("rgb", [0, 0, 0])
    stroke_width = entity.get("line_width", 0.25)

    # Normalize RGB to [-1, 1]
    r, g, b = rgb[0] / 127.5 - 1, rgb[1] / 127.5 - 1, rgb[2] / 127.5 - 1

    # One-hot entity type
    is_line = 1.0 if etype == "LINE" else 0.0
    is_arc = 1.0 if etype == "ARC" else 0.0
    is_circle = 1.0 if etype == "CIRCLE" else 0.0
    is_ellipse = 0.0  # dataset doesn't contain ellipses

    if etype == "LINE":
        sx, sy = entity["start"]
        ex, ey = entity["end"]
        cx, cy = (sx + ex) / 2, (sy + ey) / 2
        length = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)
        angle = math.atan2(ey - sy, ex - sx)
    elif etype == "ARC":
        cx, cy = entity["center"]
        radius = entity["radius"]
        sa = math.radians(entity.get("start_angle", 0))
        ea = math.radians(entity.get("end_angle", 360))
        angle_span = ea - sa
        if angle_span < 0:
            angle_span += 2 * math.pi
        length = radius * abs(angle_span)
        angle = (sa + ea) / 2
    elif etype == "CIRCLE":
        cx, cy = entity["center"]
        radius = entity["radius"]
        length = 2 * math.pi * radius
        angle = 0.0
    else:
        # Fallback for unknown types
        cx, cy = 0.0, 0.0
        length = 0.0
        angle = 0.0

    feats = np.array(
        [angle, length, r, g, b, is_line, is_arc, is_circle, is_ellipse, stroke_width],
        dtype=np.float32,
    )
    center = np.array([cx, cy], dtype=np.float64)
    return center, feats


def _parse_instance_id(instance_str: Optional[str]) -> int:
    """Convert instance string like 'Instance_i_18_63' to integer ID.

    Returns -1 for stuff classes (no instance) or missing values.
    """
    if not instance_str or instance_str == "":
        return -1
    # Format: Instance_i_{sem}_{id} or {class_name}_{id}
    parts = instance_str.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


class SVGDataset(Dataset):
    def __init__(self, cfg, logger=None):
        self.data_root = cfg.data_root
        self.split = cfg.get("split", "train")
        self.img_size = cfg.get("img_size", 980)
        self.min_points = 64
        self.data_norm = cfg.get("data_norm", "mean")
        self.aug = cfg.get("aug", False)
        self.repeat = cfg.get("repeat", 1)

        # Find split file or list all json files
        split_path = cfg.get("split_path", None)
        if split_path and osp.exists(osp.join(self.data_root, split_path)):
            with open(osp.join(self.data_root, split_path)) as f:
                split_data = json.load(f)
            self.file_ids = split_data.get(self.split, [])
        else:
            # No split file — use all json files
            json_dir = osp.join(self.data_root, "json")
            self.file_ids = [
                f.replace(".json", "")
                for f in os.listdir(json_dir)
                if f.endswith(".json")
            ]
            # Deterministic split: 70% train, 10% val, 20% test
            self.file_ids.sort()
            n = len(self.file_ids)
            if self.split == "train":
                self.file_ids = self.file_ids[: int(0.7 * n)]
            elif self.split == "val":
                self.file_ids = self.file_ids[int(0.7 * n) : int(0.8 * n)]
            elif self.split == "test":
                self.file_ids = self.file_ids[int(0.8 * n) :]

        self.file_ids = self.file_ids * self.repeat

        # Image transforms
        transforms = [T.Resize((self.img_size, self.img_size))]
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.img_transform = T.Compose(transforms)

        # Augmentation transforms
        self.aug_prob = 0.3 if self.aug else 0.0

        if logger:
            logger.info(
                f"SVGDataset [{self.split}]: {len(self.file_ids)} samples "
                f"from {self.data_root}"
            )

    def __len__(self):
        return len(self.file_ids)

    @staticmethod
    def load(
        data_root: str,
        file_name: str,
        idx: int = 1,
        min_points: int = 64,
        img_size: int = 980,
    ):
        """Load a single sample. Used by inference.py."""
        json_path = osp.join(data_root, "json", f"{file_name}.json")
        png_path = osp.join(data_root, "png", f"{file_name}.png")

        with open(json_path) as f:
            data = json.load(f)

        entities = data["entities"]
        if not entities:
            raise ValueError(f"Empty entities in {json_path}")

        centers_list = []
        feats_list = []
        sem_labels = []
        inst_labels = []
        layer_ids = []

        for entity in entities:
            center, feat = _compute_primitive_features(entity)
            centers_list.append(center)
            feats_list.append(feat)

            sem = entity.get("semantic", BG_LABEL)
            if sem == BG_LABEL:
                sem = NUM_CLASSES  # map 100 -> 30 (background class)
            sem_labels.append(sem)

            inst_id = _parse_instance_id(entity.get("instance", None))
            inst_labels.append(inst_id)
            layer_ids.append(0)  # DPSS doesn't use layer IDs

        centers = np.array(centers_list, dtype=np.float64)
        feats = np.array(feats_list, dtype=np.float32)
        labels = np.column_stack([sem_labels, inst_labels]).astype(np.int64)
        layer_ids = np.array(layer_ids, dtype=np.int64)
        n_points = len(entities)

        # Normalize coordinates to [0, 1]
        if centers.shape[0] > 0:
            mins = centers.min(axis=0)
            maxs = centers.max(axis=0)
            span = maxs - mins
            span[span < 1e-6] = 1.0
            centers = (centers - mins) / span

        # Pad to 3D (z=0) for Point Transformer
        coords = np.column_stack([centers, np.zeros(len(centers))]).astype(np.float64)

        # Pad if too few points
        if n_points < min_points:
            pad_n = min_points - n_points
            coords = np.pad(coords, ((0, pad_n), (0, 0)), mode="wrap")
            feats = np.pad(feats, ((0, pad_n), (0, 0)), mode="wrap")
            labels = np.pad(
                labels, ((0, pad_n), (0, 0)), mode="constant", constant_values=-1
            )
            layer_ids = np.pad(layer_ids, (0, pad_n), mode="wrap")

        # Load image
        img = Image.open(png_path).convert("RGB")

        lengths = np.array([n_points], dtype=np.float32)

        json_file = json_path
        bound = [mins[0], mins[1], maxs[0], maxs[1]] if n_points > 0 else [0, 0, 1, 1]

        return coords, feats, labels, lengths, layer_ids, img, bound, json_file

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        coords, feats, labels, lengths, layer_ids, img, bound, json_file = self.load(
            self.data_root, file_id, idx, self.min_points, self.img_size
        )

        # Augmentation
        if self.aug and random.random() < self.aug_prob:
            # Horizontal flip
            if random.random() < 0.5:
                coords[:, 0] = 1.0 - coords[:, 0]
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # Vertical flip
            if random.random() < 0.5:
                coords[:, 1] = 1.0 - coords[:, 1]
                img = img.transpose(Image.FLIP_TOP_BOTTOM)

        # Image transform
        img = self.img_transform(img)

        # Normalize coords (subtract mean)
        centers = coords[:, :2].copy()
        centers_for_img = centers * 2 - 1  # map [0,1] -> [-1,1] for grid_sample
        coords -= np.mean(coords, 0)

        return (
            torch.FloatTensor(coords),
            torch.FloatTensor(feats),
            torch.LongTensor(labels),
            torch.FloatTensor(lengths),
            torch.LongTensor(layer_ids),
            img,
            torch.FloatTensor(centers_for_img),
            json_file,
        )


def collate_fn(batch):
    """Collate variable-length samples into a batch."""
    coords_list, feats_list, labels_list = [], [], []
    lengths_list, layer_ids_list = [], []
    imgs_list, centers_list, json_files = [], [], []

    offsets = []
    total = 0

    for (coords, feats, labels, lengths, layer_ids, img, centers, json_file) in batch:
        n = coords.shape[0]
        coords_list.append(coords)
        feats_list.append(feats)
        labels_list.append(labels)
        lengths_list.append(lengths)
        layer_ids_list.append(layer_ids)
        imgs_list.append(img)
        centers_list.append(centers)
        json_files.append(json_file)
        total += n
        offsets.append(total)

    coords = torch.cat(coords_list, dim=0)
    feats = torch.cat(feats_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    layer_ids = torch.cat(layer_ids_list, dim=0)
    offsets = torch.IntTensor(offsets)
    lengths = torch.cat(lengths_list, dim=0)

    return (
        coords,
        feats,
        labels,
        offsets,
        lengths,
        layer_ids,
        imgs_list,
        centers_list,
        json_files,
    )


def build_dataset(cfg, logger=None):
    return SVGDataset(cfg, logger)


def build_dataloader(args, dataset, training=True, dist=False, batch_size=8, num_workers=4):
    sampler = DistributedSampler(dataset, shuffle=training) if dist else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(training and sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        sampler=sampler,
        pin_memory=True,
        drop_last=training,
    )
