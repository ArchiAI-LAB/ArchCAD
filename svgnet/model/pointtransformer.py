"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/pointtransformer.py

Contains substantial modifications to the original code.

NOTICE: Original repository does not specify a license.
This derivative work is for research purposes only.
"""

import torch
import torch.nn as nn
from modules.pointtransformer_utils import PointTransformerBlock, TransitionDown, TransitionUp


class Model(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        block = PointTransformerBlock
        num_block = [2, 3, 4, 6, 3]
        self.in_planes = cfg.in_channels
        self.planes = [32*2, 64*2, 128*2, 256*2, 512*2]
        share_planes = 8
        stride, nsample = [1, 4, 4, 4, 4], [16, 16, 16, 16, 16]
        self.enc1 = self._make_enc(block, self.planes[0], num_block[0], share_planes, stride=stride[0],
                                   nsample=nsample[0])  # N/1
        self.enc2 = self._make_enc(block, self.planes[1], num_block[1], share_planes, stride=stride[1],
                                   nsample=nsample[1], num_sector=4)  # N/4
        self.enc3 = self._make_enc(block, self.planes[2], num_block[2], share_planes, stride=stride[2],
                                   nsample=nsample[2])  # N/16
        self.enc4 = self._make_enc(block, self.planes[3], num_block[3], share_planes, stride=stride[3],
                                   nsample=nsample[3])  # N/64
        self.enc5 = self._make_enc(block, self.planes[4], num_block[4], share_planes, stride=stride[4],
                                   nsample=nsample[4])  # N/256
        self.dec5 = self._make_dec(block, self.planes[4], 2, share_planes, nsample=nsample[4], is_head=True)  # transform p5
        self.dec4 = self._make_dec(block, self.planes[3], 2, share_planes, nsample=nsample[3])  # fusion p5 and p4
        self.dec3 = self._make_dec(block, self.planes[2], 2, share_planes, nsample=nsample[2])  # fusion p4 and p3
        self.dec2 = self._make_dec(block, self.planes[1], 2, share_planes, nsample=nsample[1])  # fusion p3 and p2
        self.dec1 = self._make_dec(block, self.planes[0], 2, share_planes, nsample=nsample[0])  # fusion p2 and p1
        
    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16, num_sector=1):
        layers = [TransitionDown(self.in_planes, planes * block.expansion, stride, nsample, num_sector)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def _make_dec(self, block, planes, blocks, share_planes=8, nsample=16, is_head=False):
        layers = [TransitionUp(self.in_planes, None if is_head else planes * block.expansion)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, self.in_planes, share_planes, nsample=nsample))
        return nn.Sequential(*layers)

    def forward(self, stage_list):
        
        p0 = stage_list["inputs"]["p_out"] # (n, 3)
        x0 = stage_list["inputs"]["f_out"] # (n, c), 
        o0 = stage_list["inputs"]["offset"]#  (b)
        #layerids = stage_list["inputs"]['layerids']
        x0 = p0 if self.in_planes == 3 else torch.cat((p0, x0), 1)
        [p1, x1, o1] = self.enc1[0]([p0, x0, o0])
        p1, x1, o1, knn_idx1 = self.enc1[1:]([p1, x1, o1,None])
        [p2, x2, o2], idx2 = self.enc2[0]([p1, x1, o1])
        p2, x2, o2,knn_idx2  = self.enc2[1:]([p2, x2, o2, None])
        #layerids2 = layerids[idx2]
        [p3, x3, o3], idx3 = self.enc3[0]([p2, x2, o2])
        p3, x3, o3,knn_idx3 = self.enc3[1:]([p3, x3, o3, None])
        #layerids3 = layerids2[idx3]
        [p4, x4, o4], idx4 = self.enc4[0]([p3, x3, o3])
        p4, x4, o4,knn_idx4 = self.enc4[1:]([p4, x4, o4, None])
        #layerids4 = layerids3[idx4]
        [p5, x5, o5], idx5 = self.enc5[0]([p4, x4, o4])
        p5, x5, o5,knn_idx5 = self.enc5[1:]([p5, x5, o5,None])
        #layerids5 = layerids4[idx5]
        down_list = [
            # [p0, x0, o0],  # (n, 3), (n, in_feature_dims), (b)
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # (n, 3), (n, base_fdims), (b) - default base_fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512
        ]
        stage_list['down'] = down_list
        x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5,knn_idx5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4,knn_idx4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3,knn_idx3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2,knn_idx2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1,knn_idx1])[1]
       
        up_list = [
            {'p_out': p1, 'f_out': x1, 'offset': o1},  # n_0 = n, fdims = 32
            {'p_out': p2, 'f_out': x2, 'offset': o2},  # n_1
            {'p_out': p3, 'f_out': x3, 'offset': o3},  # n_2
            {'p_out': p4, 'f_out': x4, 'offset': o4},  # n_3
            {'p_out': p5, 'f_out': x5, 'offset': o5},  # n_4 - fdims = 512 (extracted through dec5 = mlps)
        ]
        stage_list['up'] = up_list
        return stage_list
