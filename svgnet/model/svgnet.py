"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/svgnet.py

Contains substantial modifications to the original code.

NOTICE: Original repository does not specify a license.
This derivative work is for research purposes only.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..util import cuda_cast
from .pointtransformer import Model as PointT
#from .pointnet2 import Model as PointT
from .decoder import Decoder
from ..vision.vision_embed import ImgEmbed #


import numpy as np

class SVGNet(nn.Module):
    def __init__(
        self,cfg,criterion=None):
        super().__init__()
        self.criterion = criterion

        self.image_embed = ImgEmbed(cfg) #


        # NOTE backbone
        self.backbone = PointT(cfg)
        self.decoder = Decoder(cfg,self.backbone.planes)
        self.num_classes = cfg.semantic_classes
        self.test_object_score = 0.1
        
    def train(self, mode=True):
        super().train(mode)
        
    def forward(self, batch,return_loss=True):
        coords,feats,semantic_labels,offsets,lengths,layerIds, imgs, center, json_file = batch
        return self._forward(coords,feats,offsets,semantic_labels,lengths,layerIds,imgs,center,json_file,return_loss=return_loss)

     
    def prepare_targets(self,semantic_labels,bg_ind=-1,bg_sem=30):
        
        instance_ids = semantic_labels[:,1].cpu().numpy()
        semantic_ids = semantic_labels[:,0].cpu().numpy()
        
        keys = []
        for sem_id,ins_id in zip(semantic_ids,
                             instance_ids):
            if (sem_id,ins_id) not in keys:
                keys.append((sem_id,ins_id))
    
        cls_targets,mask_targets = [], []
        svg_len = semantic_ids.shape[0]

        for (sem_id,ins_id) in keys:
            if sem_id==30 and ins_id== -1: continue #


            tensor_mask = torch.zeros(svg_len)
            ind1 = np.where(semantic_ids==sem_id)[0]
            ind2 = np.where(instance_ids==ins_id)[0]
            ind = list(set(ind1).intersection(ind2))
            tensor_mask[ind] = 1
            cls_targets.append(sem_id)
            mask_targets.append(tensor_mask.unsqueeze(1))

        cls_targets = torch.tensor(cls_targets) if cls_targets else torch.tensor([30])   # 
        mask_targets = torch.cat(mask_targets,dim=1) if mask_targets else torch.zeros(svg_len,1)
        
        
        return [{
            "labels": cls_targets.to(semantic_labels.device),
            "masks": mask_targets.to(semantic_labels.device),

        }]

    @cuda_cast
    def _forward(
        self,
        coords,   # [x, 3]
        feats,    # [x, 7]
        offsets,    # [2]
        semantic_labels,
        lengths,
        layerIds,
        imgs,
        centers,
        json_file,
        return_loss=True
    ):

        img_embed = self.image_embed(imgs, centers) #

        stage_list={'inputs': {'p_out':coords,"f_out":feats,"offset":offsets},"semantic_labels":semantic_labels[:,0]}
        targets = self.prepare_targets(semantic_labels)
        stage_list.update({"tgt":targets})
        
        stage_list = self.backbone(stage_list)

        outputs = self.decoder(stage_list,layerIds,img_embed,json_file)

        model_outputs = {}
        if not self.training:
            semantic_scores=self.semantic_inference(outputs["pred_logits"],outputs["pred_masks"])
            instances = self.instance_inference(outputs["pred_logits"],outputs["pred_masks"])
            model_outputs.update(
                dict(
                semantic_scores=semantic_scores,
                ), 
            )
       
            model_outputs.update(
                dict(
                semantic_labels=semantic_labels[:,0],
                    ), 
             )
            model_outputs.update(
                dict(
                instances=instances,
                ),
            )

            model_outputs.update(
                dict(
                targets=targets[0],
                ),
            )
            model_outputs.update(
                dict(
                lengths=lengths,
                ),
            )
         
        
        if not return_loss:
            return model_outputs
        # NOTE cal loss
        
        losses = self.criterion(outputs,targets)
        loss_value,loss_dicts = self.parse_losses(losses)
        
        
        return model_outputs,loss_value,loss_dicts

    
    def semantic_inference(self, mask_cls, mask_pred):
        
        mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1] # Q,C
        mask_pred = mask_pred.sigmoid() # Q,G
        semseg = torch.einsum("bqc,bqg->bgc", mask_cls, mask_pred)
        return semseg[0]

    def instance_inference(self,mask_cls,mask_pred,overlap_threshold=0.8):
        
        mask_cls,mask_pred = mask_cls[0],mask_pred[0]
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores >= self.test_object_score)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep][:, :-1]

        cur_prob_masks = cur_scores[..., None] * cur_masks
        current_segment_id = 0
        nline = cur_masks.shape[-1]

        results = []
        # take argmax
        try:
            cur_mask_ids = cur_prob_masks.argmax(0)
        except: 
            return results
        
        for k in range(cur_classes.shape[0]):

            pred_class = cur_classes[k].item()
            pred_score = cur_scores[k].item()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue
                current_segment_id += 1
                #print(pred_class, pred_score)
                results.append({
                    "masks": mask.cpu().numpy(),
                    "labels": pred_class,
                    "scores": pred_score
                })

        return results
    
    
    def instance_inference2(self, mask_cls, mask_pred):
    
        mask_cls,mask_pred = mask_cls[0],mask_pred[0]
        scores, labels  = F.softmax(mask_cls, dim=-1)[...,:-1].max(-1)
        keep = labels.ne(self.num_classes) & (scores > self.test_object_score)
        mask_pred = mask_pred[keep]
        labels_per_query = labels[keep]
        scores_per_query = scores[keep]

        result_pred_mask = (mask_pred > 0).float()
        heatmap = mask_pred.float().sigmoid()
        mask_scores_per_image = (heatmap * result_pred_mask).sum(1) / (result_pred_mask.sum(1) + 1e-6)
        scores_per_mask = scores_per_query * mask_scores_per_image
        labels_per_mask = labels_per_query

        results = []
        for score, label, mask in zip(scores_per_mask, labels_per_mask, result_pred_mask):
            results.append({
                    "masks": mask.cpu().numpy(),
                    "labels": label.item(),
                    "scores": score.item(),

            })
        return results


    
    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses["loss"] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses



