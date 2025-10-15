"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/pointnet2.py

Contains substantial modifications to the original code.

NOTICE: Original repository does not specify a license.
This derivative work is for research purposes only.
"""

import torch
import torch.nn as nn
from modules.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.planes = [128*6, 128*6, 256*6, 256*6]
        self.sa1 = PointNetSetAbstraction(4, 32, 9 + 3, [32*6, 32*6, 64*6], num_sector=4)
        self.sa2 = PointNetSetAbstraction(4, 32, 64*6 + 3, [64*6, 64*6, 128*6])
        self.sa3 = PointNetSetAbstraction(4, 32, 128*6 + 3, [128*6, 128*6, 256*6])
        self.sa4 = PointNetSetAbstraction(4, 32, 256*6 + 3, [256*6, 256*6, 512*6])

        self.fp4 = PointNetFeaturePropagation(768*6, [256*6, 256*6])
        self.fp3 = PointNetFeaturePropagation(384*6, [256*6, 256*6])
        self.fp2 = PointNetFeaturePropagation(320*6, [256*6, 128*6])
        self.fp1 = PointNetFeaturePropagation(128*6, [128*6, 128*6, 128*6])

        

    def forward(self, stage_list):
        
        p0 = stage_list["inputs"]["p_out"] # (n, 3)
        x0 = stage_list["inputs"]["f_out"] # (n, c), 
        o0 = stage_list["inputs"]["offset"]#  (b)
        pos_feat_off0 = [p0, x0, o0]
        pos_feat_off0[1] = torch.cat([pos_feat_off0[0], pos_feat_off0[1]], 1)

        pos_feat_off1 = self.sa1(pos_feat_off0)
        pos_feat_off2 = self.sa2(pos_feat_off1)
        pos_feat_off3 = self.sa3(pos_feat_off2)
        pos_feat_off4 = self.sa4(pos_feat_off3)

        pos_feat_off3[1] = self.fp4(pos_feat_off3, pos_feat_off4)
        pos_feat_off2[1] = self.fp3(pos_feat_off2, pos_feat_off3)
        pos_feat_off1[1] = self.fp2(pos_feat_off1, pos_feat_off2)
        pos_feat_off0[1] = self.fp1([pos_feat_off0[0], None, pos_feat_off0[2]], pos_feat_off1)
      
        up_list = [
            {'p_out': pos_feat_off0[0], 'f_out': pos_feat_off0[1], 'offset': pos_feat_off0[2]},
            {'p_out': pos_feat_off1[0], 'f_out': pos_feat_off1[1], 'offset': pos_feat_off1[2]},
            {'p_out': pos_feat_off2[0], 'f_out': pos_feat_off2[1], 'offset': pos_feat_off2[2]},
            {'p_out': pos_feat_off3[0], 'f_out': pos_feat_off3[1], 'offset': pos_feat_off3[2]},
        
        ]
        stage_list['up'] = up_list

        return stage_list
