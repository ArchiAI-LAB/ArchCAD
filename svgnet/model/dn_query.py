"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/dn_query.py

NOTICE: Original repository does not specify a license.
This code is used for research purposes only.
Will be removed or replaced if licensing issues arise.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .basic_operators import get_subscene_features

class CDNQueries(object):
    
    def __init__(self,
                 num_queries: int = 300,
                 num_classes: int = 50,
                 denoising_groups: int = 1,
                 label_noise_prob: float = 0.2,
                 mask_noise_scale: float = 0.2,
                 head_dn: bool = False
                 ) -> None:
        super(CDNQueries,self).__init__()
        
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        self.denoising_groups = denoising_groups
        self.label_noise_prob = label_noise_prob
        self.mask_noise_scale = mask_noise_scale
        self.head_dn = head_dn
        
    
    def label_noise(self,targets,label_enc):
        
        labels=torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(self.denoising_groups, 1).view(-1)
        dn_label_noise_ratio = self.label_noise_prob
        knwon_labels_expand = known_labels.clone()
        
        if dn_label_noise_ratio > 0:
            prob = torch.rand_like(knwon_labels_expand.float())
            chosen_indice = prob < dn_label_noise_ratio
            new_label = torch.randint_like(knwon_labels_expand[chosen_indice], 0,
                                           self.num_classes)  # randomly put a new one here
            # gt_labels_expand.scatter_(0, chosen_indice, new_label)
            knwon_labels_expand[chosen_indice] = new_label

        noised_known_features = label_enc(knwon_labels_expand)
        return noised_known_features
        
    
    def mask_noise(self, targets):
        masks = torch.cat([t['masks'] for t in targets]).bool()
        group_num = masks.shape[0]
        masks = masks.repeat(1,self.denoising_groups).transpose(0,1)   
        areas= (~masks).sum(1)
        noise_ratio=areas*self.mask_noise_scale/(group_num)
        delta_mask=torch.rand_like(masks,dtype=torch.float)<noise_ratio[:,None]
        masks=torch.logical_xor(masks,delta_mask)  
    
        return masks
    
    def calc_indices(self, targets):
        
        num_boxes = [len(t["labels"]) for t in targets]
        single_pad = max(num_boxes)
        map_known_indices = torch.cat([torch.tensor(range(num)) for num in num_boxes])  # [1,2, 1,2,3]
       
        map_known_indices = [map_known_indices + single_pad * i for i in range(self.denoising_groups)]
        map_known_indices = torch.cat(map_known_indices).long().cuda()
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
        known_bid = batch_idx.repeat(self.denoising_groups, 1).view(-1)
        
        return single_pad,known_bid,map_known_indices
    
    def query_for_dn(self,queries,targets,label_enc):
        
        bs = len(targets)
        single_pad,known_bid,map_known_indices = self.calc_indices(targets)
        pad_size = self.denoising_groups * single_pad
        padding = torch.zeros([bs, pad_size, queries.shape[-1]]).cuda()
        
        padding[(known_bid, map_known_indices)] = self.label_noise(targets,label_enc)
        queries = torch.cat([padding.transpose(0, 1), queries], dim=0)
        
        tgt_size = pad_size + self.num_queries
        tgt_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # attn_mask = attn_mask.to('cuda')
        # match query cannot see the reconstruct
        tgt_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(self.denoising_groups):
            tgt_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            tgt_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        
        dn_args_ = dict()
        dn_args_['max_num'] = single_pad
        dn_args_['pad_size'] = pad_size

        return queries, tgt_mask, dn_args_
        
        
    def mask_for_dn(self,stage_list,step):
        
        targets = stage_list["tgt"]
        bs = len(targets)
        single_pad,known_bid,map_known_indices = self.calc_indices(targets)
        pad_size = self.denoising_groups * single_pad
        masks = self.mask_noise(targets)
        
        group_num = masks.shape[1]
        padding_mask = torch.ones([bs, pad_size, group_num]).cuda().bool()
        
        if self.head_dn:
            padding_mask=padding_mask.unsqueeze(2).repeat([1,1,8,1]) #[bs,q,h,h*w]
            padding_mask[(known_bid, map_known_indices)] = masks
            padding_mask=padding_mask.transpose(1,2)
        else:
            padding_mask[(known_bid, map_known_indices)]=masks
            padding_mask=padding_mask.unsqueeze(1).repeat([1,8,1,1])
        
        padding_mask = padding_mask.flatten(0,1)

        padding_mask = padding_mask.flatten(0,1).transpose(0,1)
        padding_mask = get_subscene_features("up", step, stage_list, padding_mask, 
                                             torch.tensor([4, 4, 4, 4]))
        padding_mask = padding_mask.view(8,pad_size,-1)
        
        return padding_mask.transpose(0,1)
        
        
        
    

        
