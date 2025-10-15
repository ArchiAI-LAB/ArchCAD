"""
Source: https://github.com/nicehuster/SymPointV2/blob/master/svgnet/model/blocks.py

NOTICE: Original repository does not specify a license.
This code is used for research purposes only.
Will be removed or replaced if licensing issues arise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from modules.pointops.functions import pointops
from .utils import *
from .basic_operators import *
from .basic_operators import _eps, _inf

class MLP(nn.Module):
    """ mlp(s) to generate from f_out to the desired fkey (latent/logits)
    """
    fkey_to_dims = None
    def __init__(self, fdim,config, fkey, drop=None):
        super().__init__()
        infer_list = []
        fkey = get_ftype(fkey)[0]
        valid_fkey = {
            'latent': config.base_fdim,
            'logits': config.num_class,
        }
        assert fkey in valid_fkey
        if MLP.fkey_to_dims is None:
            MLP.fkey_to_dims = valid_fkey

        if fkey in ['latent', 'logits']:
            d_out = valid_fkey['latent']
            if 'latent_ops' in config and config.latent_ops:
                ops = [MLPbyOps(config.latent_ops, fdim, d_out)]
            else:
                ops = [nn.Linear(fdim, d_out), nn.BatchNorm1d(d_out), nn.ReLU(inplace=True)]
            infer_list += ops
            fdim = d_out

        if fkey in ['logits']:
            d_out = valid_fkey['logits']
            if 'logits_ops' in config and config.logits_ops:
                ops = [MLPbyOps(config.logits_ops, fdim, d_out)]
            else:
                ops = [nn.Linear(fdim, d_out)]
            infer_list += ops
        self.infer = nn.Sequential(*infer_list)

    def forward(self, stage, k):
        return self.infer(stage[k])

class MLPbyOps(nn.Module):
    @property
    def mlp_kwargs(self):
        return {
            'activation': 'relu',
            'bias': True,
            'bn': True,
            'linear_bn': False,
        }
    def __init__(self, ops, fdim, d_mid=None, d_out=None, **kwargs):
        super().__init__()
        ops_seq = ops.split('-') if '-' in ops else [ops]
        d_mid = d_mid if d_mid else fdim
        d_out = d_out if d_out else d_mid

        ops_list = []
        for ops in ops_seq:
            assert 'mlp' in ops or ops in ['linear', 'linearbn'], f'invalid ops = {ops}'
            mlp_kwargs = self.mlp_kwargs
            mlp_kwargs.update(kwargs)

            num_mlp = re.search('\d+', ops)
            num_mlp = int(num_mlp.group()) if num_mlp else 1
            linear = 'linear' in ops or not ops.endswith('mlp')  # linear / linearbn / mlp2 to ends with linear

            def get_mlp(ops_list, din, dout, mlp_kwargs):
                ops_list += [nn.Linear(din, dout, bias=mlp_kwargs['bias'])]
                if mlp_kwargs['bn']:
                    ops_list += [nn.BatchNorm1d(dout)]
                if mlp_kwargs['activation'] == 'relu':
                    ops_list += [nn.ReLU(inplace=True)]
                elif mlp_kwargs['activation'] == '':
                    pass
                else:
                    raise ValueError(f'not support activation = ' + mlp_kwargs['activation'])
                return ops_list

            for i in range(num_mlp - 1):
                ops_list = get_mlp(ops_list, din=fdim, dout=d_mid, mlp_kwargs=mlp_kwargs)
                fdim = d_mid

            if linear:
                mlp_kwargs['activation'] = ''
                mlp_kwargs['bn'] = False
            cur_out = d_out if ops == ops_seq[-1] else d_mid
            ops_list = get_mlp(ops_list, din=fdim, dout=cur_out, mlp_kwargs=mlp_kwargs)
            fdim = cur_out
            if mlp_kwargs['linear_bn'] or 'linearbn' in ops:
                ops_list += [nn.BatchNorm1d(fdim)]

        self.ops_func = nn.Sequential(*ops_list)

    def forward(self, features):
        return self.ops_func(features)


class MLPBlock(nn.Module):
    # block-level interface of MLP / MLPbyOps

    def __init__(self, stage_n, stage_i, block_cfg, config, **kwargs):
        super(MLPBlock, self).__init__()
        self.config = config
        self.block_cfg = block_cfg
        mlp_kwargs = {'fdim': config.nsample[stage_i]}
        if block_cfg.kwargs:
            mlp_kwargs.update(block_cfg.kwargs)
        mlp_kwargs.update(**kwargs)
        self.mlp_kwargs = mlp_kwargs
        self.mlp = MLPbyOps(block_cfg.ops, **mlp_kwargs)

    def forward(self, pxo, stage_n, stage_i, stage_list, inputs):
        x = self.mlp(pxo[1])
        pxo[1] = x
        return pxo

