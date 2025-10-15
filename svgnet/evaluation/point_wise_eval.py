"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/evaluation/point_wise_eval.py

NOTICE: Original repository does not specify a license.
This code is used for research purposes only.
Will be removed or replaced if licensing issues arise.
"""

import numpy as np
from svgnet.data.svg import SVG_CATEGORIES
import torch
import torch.distributed as dist

class PointWiseEval(object):
    def __init__(self, num_classes=30, ignore_label=30,gpu_num=1) -> None:
        self.ignore_label = ignore_label
        self._num_classes = num_classes
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.float32)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._class_names = [x["name"] for x in SVG_CATEGORIES[:-1]]
        self.gpu_num = gpu_num
        
    def update(self, pred_sem, gt_sem):
        
        pos_inds = gt_sem != self.ignore_label
        pred = pred_sem[pos_inds]
        gt = gt_sem[pos_inds]

        self._conf_matrix += np.bincount(
                (self._num_classes + 1) * pred.reshape(-1) + gt.reshape(-1),
                minlength=self._conf_matrix.size,
            ).reshape(self._conf_matrix.shape)

    def get_eval(self, logger):
        
        if self.gpu_num>1:
            t =  torch.from_numpy(self._conf_matrix).to("cuda")
            conf_matrix_list = [torch.full_like(t,0) for _ in range(self.gpu_num)]
            dist.barrier()
            dist.all_gather(conf_matrix_list,t)
            self._conf_matrix = torch.full_like(t,0)
            for conf_matrix in conf_matrix_list:
                self._conf_matrix += conf_matrix
            self._conf_matrix = self._conf_matrix.cpu().numpy()
        
        # mIoU
        acc = np.full(self._num_classes, np.nan, dtype=np.float64)
        iou = np.full(self._num_classes, np.nan, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        class_weights = pos_gt / (np.sum(pos_gt)+1e-8)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / (pos_gt[acc_valid]+1e-8)
        union = pos_gt + pos_pred - tp
        iou_valid = np.logical_and(acc_valid, union > 0)
        iou[iou_valid] = tp[iou_valid] / (union[iou_valid]+1e-8)
        macc = np.sum(acc[acc_valid]) / (np.sum(acc_valid)+1e-8)
        miou = np.sum(iou[iou_valid]) / (np.sum(iou_valid)+1e-8)
        fiou = np.sum(iou[iou_valid] * class_weights[iou_valid])
        pacc = np.sum(tp) / (np.sum(pos_gt)+1e-8)

        miou, fiou, pACC = 100 * miou, 100 * fiou, 100 * pacc
        for i, name in enumerate(self._class_names):
            logger.info('Class_{}  IoU: {:.3f}'.format(name,iou[i]*100))
        
        logger.info('mIoU / fwIoU / pACC : {:.3f} / {:.3f} / {:.3f}'.format(miou, fiou, pACC))
        
        return miou, pACC

class InstanceEval(object):
    def __init__(self, num_classes=30,
                 ignore_label=30,
                 gpu_num=8) -> None:

        self.ignore_label = ignore_label
        self._num_classes = num_classes
        self._class_names = [x["name"] for x in SVG_CATEGORIES[:-1]]
        self.gpu_num = gpu_num
        self.min_obj_score = 0.1
        self.IoU_thres = 0.5

        self.tp_classes = np.zeros(num_classes)
        self.tp_classes_values = np.zeros(num_classes)
        self.fp_classes = np.zeros(num_classes)
        self.fn_classes = np.zeros(num_classes)

        self.thing_class =  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 25, 27, 29]

        self.stuff_class =  [0, 12, 13, 19, 20, 23, 24, 26, 28]

    def update(self, instances, target, lengths):
        
        lengths = np.round( np.log(1 + lengths.cpu().numpy()) , 3)
        tgt_labels = target["labels"].cpu().numpy().tolist()
        tgt_masks = target["masks"].transpose(0,1).cpu().numpy()
        for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):
            if tgt_label==self.ignore_label: continue

            flag = False
            for instance in instances:
                src_label = instance["labels"]
                src_score = instance["scores"]
                if src_label==self.ignore_label: continue
                if src_score< self.min_obj_score: continue
                src_mask = instance["masks"]
                
                interArea = sum(lengths[np.logical_and(src_mask,tgt_mask)])
                unionArea = sum(lengths[np.logical_or(src_mask,tgt_mask)])
                iou = interArea / (unionArea + 1e-6)
                if iou>=self.IoU_thres:
                    flag = True
                    if tgt_label==src_label:
                        self.tp_classes[tgt_label] += 1
                        self.tp_classes_values[tgt_label] += iou
                    else:
                        self.fp_classes[src_label] += 1
            if not flag: self.fn_classes[tgt_label] += 1
    
    def get_eval(self, logger):
        if self.gpu_num>1:
            _tensor = np.stack([self.tp_classes,
                               self.tp_classes_values,
                               self.fp_classes,
                               self.fn_classes])
            _tensor = torch.from_numpy(_tensor).to("cuda")
            _tensor_list = [torch.full_like(_tensor,0) for _ in range(self.gpu_num)]
            dist.barrier()
            dist.all_gather(_tensor_list,_tensor)
            all_tensor = torch.full_like(_tensor,0)
            for tensor_ in _tensor_list:
                all_tensor += tensor_

            all_tensor = all_tensor.cpu().numpy()
            self.tp_classes, self.tp_classes_values, \
                self.fp_classes, self.fn_classes= all_tensor

       # each class
        RQ = self.tp_classes / (self.tp_classes + 0.5* self.fp_classes + 0.5* self.fn_classes + 1e-6)
        SQ = self.tp_classes_values / (self.tp_classes + 1e-6)
        PQ = RQ * SQ
        
        # thing
        thing_RQ = sum(self.tp_classes[self.thing_class]) / (sum(self.tp_classes[self.thing_class]) + 0.5* sum(self.fp_classes[self.thing_class]) + 0.5* sum(self.fn_classes[self.thing_class]) + 1e-6)
        thing_SQ = sum(self.tp_classes_values[self.thing_class]) / (sum(self.tp_classes[self.thing_class]) + 1e-6)
        thing_PQ = thing_RQ * thing_SQ
        
        # stuff
        stuff_RQ = sum(self.tp_classes[self.stuff_class]) / (sum(self.tp_classes[self.stuff_class]) + 0.5* sum(self.fp_classes[self.stuff_class]) + 0.5* sum(self.fn_classes[self.stuff_class]) + 1e-6)
        stuff_SQ = sum(self.tp_classes_values[self.stuff_class]) / (sum(self.tp_classes[self.stuff_class]) + 1e-6)
        stuff_PQ = stuff_RQ * stuff_SQ
        
        #total
        sRQ = sum(self.tp_classes) / (sum(self.tp_classes) + 0.5* sum(self.fp_classes) + 0.5* sum(self.fn_classes) + 1e-6)
        sSQ = sum(self.tp_classes_values) / (sum(self.tp_classes) + 1e-6)
        sPQ = sRQ * sSQ
        
        for i, name in enumerate(self._class_names):
            logger.info('Class_{}  PQ: {:.3f}'.format(name,PQ[i]*100))

        logger.info('PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}'.format(sPQ*100, sRQ*100, sSQ*100))
        logger.info('thing PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}'.format(thing_PQ*100, thing_RQ*100, thing_SQ*100))
        logger.info('stuff PQ / RQ / SQ : {:.3f} / {:.3f} / {:.3f}'.format(stuff_PQ*100, stuff_RQ*100, stuff_SQ*100))
        return sPQ*100, sRQ*100, sSQ*100