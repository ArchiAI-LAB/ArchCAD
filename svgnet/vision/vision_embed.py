
from torch.nn import functional as F
import torch
import torch.nn as nn
from .seg_hrnet import get_seg_model


_FEAT_DIMS = {
    "resnet18": (64, 128, 256, 512),
    "resnet34": (64, 128, 256, 512),
    "resnet50": (256, 512, 1024, 2048),
    "resnet101": (256, 512, 1024, 2048),
    "resnet152": (256, 512, 1024, 2048),
    "hrnet18": (18, 36, 72, 144),
    "hrnet48": (48, 96, 192, 384), #
    "resnet101_fpn": (256, 256, 256, 256)
}


def vert_align_custom(feats, verts, interp_mode='bilinear',
    padding_mode='zeros', align_corners=True):
    if torch.is_tensor(verts):
        if verts.dim() != 3:
            raise ValueError("verts tensor should be 3 dimensional")
        grid = verts
    else:
        raise ValueError(
            "verts must be a tensor or have a "
            + "`points_padded' or`verts_padded` attribute."
        )
    grid = grid[:, None, :, :2]  # (N, 1, V, 2)
    if torch.is_tensor(feats):
        feats = [feats]
    for feat in feats:
        if feat.dim() != 4:
            raise ValueError("feats must have shape (N, C, H, W)")
        if grid.shape[0] != feat.shape[0]:
            raise ValueError("inconsistent batch dimension")
    feats_sampled = []
    for feat in feats:
        # print(feat.shape)
        feat_sampled = F.grid_sample(
            feat,
            grid,
            mode=interp_mode,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )  # (N, C, 1, V)
        feat_sampled = feat_sampled.squeeze(dim=2).transpose(1, 2)  # (N, V, C)
        feats_sampled.append(feat_sampled)
    feats_sampled = torch.cat(feats_sampled, dim=2)  # (N, V, sum(C))
    return feats_sampled


class ImgEmbed(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        #print(cfg)
        #print(f"> InputEmbed: {cfg.vision.backbone}")
        # print(type(cfg))
        if cfg.vision.backbone == "hrnet48":
            self.EmbedBackbone = get_seg_model(cfg)
        elif cfg.vision.backbone == "hrnet18":
            self.EmbedBackbone = get_seg_model(cfg)
        else:
            raise NotImplementedError

        self.EmbedDim = _FEAT_DIMS[cfg.vision.backbone]
        self.bottleneck = nn.Linear(sum(self.EmbedDim), cfg.vision.dim)

        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)
    

    def forward(self, image, x_batch):

        image = torch.stack(image)
        device, dtype = x_batch[0].device, x_batch[0].dtype

        img_feats_batch = self.EmbedBackbone(image)
        vert_align_feats_batch = []
        for i, x in enumerate(x_batch):
            img_feats = [f[i:i+1] for f in img_feats_batch]
            x = x.unsqueeze(0)

            factor = torch.tensor([1, 1], device=device, dtype=dtype).view(1, 1, 2)
            xy_norm = x * factor

            vert_align_feats = vert_align_custom(img_feats, xy_norm)
            vert_align_feats = self.bottleneck(vert_align_feats)

            image = image.squeeze(0)
            x = x.squeeze(0)
            vert_align_feats = vert_align_feats.squeeze(0)
            vert_align_feats_batch.append(vert_align_feats)
        return torch.cat(vert_align_feats_batch, dim=0)