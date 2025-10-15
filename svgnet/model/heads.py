"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/heads.py

NOTICE: Original repository does not specify a license.
This code is used for research purposes only.
Will be removed or replaced if licensing issues arise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from modules.pointops.functions import pointops
from .basic_operators import *
from .basic_operators import _eps, _inf
from .utils import *
from .blocks import *


class ContrastHead(nn.Module):
    """ currently used as criterion - need to be wrapped with DataParallel if params used (eg. project)
    """
    def __init__(self, config):
        super().__init__()
        self.nsample = torch.tensor([36, 24, 24, 24, 24])
        self.nstride = torch.tensor([4, 4, 4, 4])
        self.num_class = torch.tensor(config.num_classes)
        self.config = config
        self.config = config
        self.stages = parse_stage(config.stage, config.num_layers)
        self.ftype = get_ftype(config.ftype)[0]
        self.dist_func = getattr(self, f'dist_{config.dist}')
        self.posmask_func = getattr(self, f'posmask_{config.pos}')
        self.contrast_func = getattr(self, f'contrast_{config.contrast_func}')
        assert config.sample in ['cnt', 'glb', 'sub', 'subspatial', 'pts', 'label', 'vote'], f'not support sample = {config.sample}'
        # self.sample_func = getattr(self, f'sample_{config.sample}') if 'sample' in config and config.sample else self.sample_label
        self.main_contrast = getattr(self, f'{config.main}_contrast') if 'main' in config and config.main else self.point_contrast

        self.temperature = config.temperature


    def sample_label(self, n, i, stage_list, target):
        p, features, offset = fetch_pxo(n, i, stage_list, self.ftype)
        nsample = self.nsample[i]
        labels = get_subscene_label(n, i, stage_list, target, self.nstride, self.num_class)  # (m, ncls) - distribution / onehot

        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o) # (m, nsample)
        # exclude self-loop
        nsample -= 1
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        m = neighbor_idx.shape[0]

        neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1]) # (m, nsample, ncls)
        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])
        # neighbor_label = pointops.queryandgroup(nsample, p, p, labels, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, ncls)
        # neighbor_feature = pointops.queryandgroup(nsample, p, p, features, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, c)

        # print('shape', p.shape, features.shape, labels.shape, 'neighbor_idx', neighbor_idx.shape, 'nsample =', nsample)
        # print('o = ', o)
        # print('shape - neighbor', neighbor_feature.shape, neighbor_label.shape)
        return nsample, labels, neighbor_label, features, neighbor_feature


    def dist_l2(self, features, neighbor_feature):
        dist = torch.unsqueeze(features, -2) - neighbor_feature
        dist = torch.sqrt(torch.sum(dist ** 2, axis=-1) + _eps) # [m, nsample]
        return dist

    def dist_kl(self, features, neighbor_feature, normalized, normalized_neighbor):
        # kl dist from featuers (gt) to neighbors (pred)
        if normalized in [False, 'softmax']:  # if still not a prob distribution - prefered
            features = F.log_softmax(features, dim=-1)
            log_target = True
        elif normalized == True:
            log_target = False
        else:
            raise ValueError(f'kl dist not support normalized = {normalized}')
        features = features.unsqueeze(-2)

        if normalized_neighbor in [False, 'softmax']:
            neighbor_feature = F.log_softmax(neighbor_feature, dim=-1)
        elif normalized_neighbor == True:
            neighbor_feature = torch.maximum(neighbor_feature, neighbor_feature.new_full([], _eps)).log()
        else:
            raise ValueError(f'kl dist not support normalized_neighbor = {normalized}')
        
        # (input, target) - i.e. (pred, gt), where input/pred should be in log space
        dist = F.kl_div(neighbor_feature, features, reduction='none', log_target=log_target)  # [m, nsample, d] - kl(pred, gt) to calculate kl = gt * [ log(gt) - log(pred) ]
        dist = dist.sum(-1)  # [m, nsample]
        return dist


    def posmask_cnt(self, labels, neighbor_label):
        labels = torch.argmax(torch.unsqueeze(labels, -2), -1)  # [m, 1]
        neighbor_label = torch.argmax(neighbor_label, -1)  # [m, nsample]
        mask = labels == neighbor_label  # [m, nsample]
        return mask

    def contrast_softnn(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        pos = torch.sum(exp * posmask, axis=-1)  # (m)
        neg = torch.sum(exp, axis=-1)  # (m)
        loss = -torch.log(pos / neg + _eps)
        
        return loss

    def contrast_nce(self, dist, posmask, invalid_mask=None):
        dist = -dist
        dist = dist - torch.max(dist, -1, keepdim=True)[0]  # NOTE: max return both (max value, index)
        if self.temperature is not None:
            dist = dist / self.temperature
        exp = torch.exp(dist)

        if invalid_mask is not None:
            valid_mask = 1 - invalid_mask
            exp = exp * valid_mask

        # each Log term an example; per-pos vs. all negs
        neg = torch.sum(exp * (1 - posmask), axis=-1)  # (m)
        under = exp + neg
        loss = (exp / (exp + neg))[posmask]  # each Log term an example
        loss = -torch.log(loss)
        return loss

    def point_contrast(self, n, i, stage_list, target):
        p, features, o = fetch_pxo(n, i, stage_list, self.ftype)
       

        nsample = self.nsample[i]
        labels = get_subscene_label(n, i, stage_list, target, self.nstride, self.config.num_classes)  # (m, ncls) - distribution / onehot
        neighbor_idx, _ = pointops.knnquery(nsample, p, p, o, o) # (m, nsample)

        # exclude self-loop
        nsample = self.nsample[i] - 1  # nsample -= 1 can only be used if nsample is py-number - results in decreasing number if is tensor, e.g. 4,3,2,1,...
        neighbor_idx = neighbor_idx[..., 1:].contiguous()
        m = neighbor_idx.shape[0]

        neighbor_label = labels[neighbor_idx.view(-1).long(), :].view(m, nsample, labels.shape[1]) # (m, nsample, ncls)

        if 'norm' in self.config.dist or self.config.dist == 'cos':
            features = F.normalize(features, dim=-1)  # p2-norm

        neighbor_feature = features[neighbor_idx.view(-1).long(), :].view(m, nsample, features.shape[1])
        # neighbor_label = pointops.queryandgroup(nsample, p, p, labels, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, ncls)
        # neighbor_feature = pointops.queryandgroup(nsample, p, p, features, neighbor_idx, o, o, use_xyz=False)  # (m, nsample, c)

        # print('shape', p.shape, features.shape, labels.shape, 'neighbor_idx', neighbor_idx.shape, 'nsample =', nsample)
        # print('o = ', o)
        # print('shape - neighbor', neighbor_feature.shape, neighbor_label.shape)
        posmask = self.posmask_cnt(labels, neighbor_label)  # (m, nsample) - bool
        # select only pos-neg co-exists
        point_mask = torch.sum(posmask.int(), -1)  # (m)
        point_mask = torch.logical_and(0 < point_mask, point_mask < nsample)
        # print('mask - pos/point', posmask.shape, point_mask.shape)

        if self.config.pos != 'cnt':
            posmask = self.posmask_func(labels, neighbor_label)

        posmask = posmask[point_mask]
        features = features[point_mask]
        neighbor_feature = neighbor_feature[point_mask]
        # print('after masking - ', posmask.shape, features.shape, neighbor_feature.shape)

        dist = self.dist_func(features, neighbor_feature)
        loss = self.contrast_func(dist, posmask)  # (m)

        loss = torch.mean(loss)
        loss *= self.config.weight
        return loss

    def forward(self, stage_list):
        
        target = stage_list["semantic_labels"]
        loss_list = {}
        for n, i in self.stages:
            loss = self.main_contrast(n, i, stage_list, target)
            loss = torch.tensor(0.).to(target.device) if not torch.isfinite(loss) else loss
            loss_list.update({"loss_cbl"+ f"_{i}": loss})
        return loss_list

