import numpy as np
import torch
import torch.nn as nn
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
import multiprocessing as mp
import os
import os.path as osp
import time
from functools import partial
from svgnet.data import build_dataloader, build_dataset
from svgnet.evaluation import PointWiseEval,InstanceEval
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.util  import get_root_logger, init_dist, load_checkpoint


def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--out", type=str, help="directory for output results")
    parser.add_argument("--save_lite", action="store_true")
    args = parser.parse_args()
    return args


def save_npy(root, name, scan_ids, arrs):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.npy") for i in scan_ids]
    pool = mp.Pool()
    pool.starmap(np.save, zip(paths, arrs))
    pool.close()
    pool.join()


def save_pred_instances(root, name, scan_ids, pred_insts, benchmark_sem_id):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    benchmark_sem_ids = [benchmark_sem_id] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, benchmark_sem_ids))
    pool.close()
    pool.join()


def save_gt_instances(root, name, scan_ids, gt_insts):
    root = osp.join(root, name)
    os.makedirs(root, exist_ok=True)
    paths = [osp.join(root, f"{i}.txt") for i in scan_ids]
    pool = mp.Pool()
    map_func = partial(np.savetxt, fmt="%d")
    pool.starmap(map_func, zip(paths, gt_insts))
    pool.close()
    pool.join()


def main():
    args = get_args()
    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    if args.dist:
        init_dist()
    logger = get_root_logger()

    model = svgnet(cfg.model).cuda()
    if args.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #logger.info(model)
    
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    gpu_num = dist.get_world_size()

    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)

    val_set = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(args,val_set, training=False, dist=args.dist, **cfg.dataloader.test)

    time_arr = []
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,ignore_label=32,gpu_num=dist.get_world_size())
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes,ignore_label=32,gpu_num=dist.get_world_size())
    
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            t1 = time.time()

            if i % 10 == 0:
                step = int(len(val_set)/gpu_num)
                logger.info(f"Infer  {i+1}/{step}")
            torch.cuda.empty_cache()
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                res = model(batch,return_loss=False)
            
            t2 = time.time()
            time_arr.append(t2 - t1)
            sem_preds = torch.argmax(res["semantic_scores"],dim=1).cpu().numpy()
            sem_gts = res["semantic_labels"].cpu().numpy()
            sem_point_eval.update(sem_preds, sem_gts)
            #sem_point_eval.update(sem_preds, sem_gts, lengths) #    def update(self, pred_sem, gt_sem):
            instance_eval.update(
                res["instances"],
                res["targets"],
                res["lengths"],
            )
           
    # logger.info("Evaluate semantic segmentation")
    # sem_point_eval.get_eval(logger)
    # logger.info("Evaluate panoptic segmentation")
    # instance_eval.get_eval(logger)
    
    # mean_time = np.array(time_arr).mean()
    # logger.info(f"Average run time: {mean_time:.4f}")

    # # save output
    # if not args.out:
    #     return

    # #logger.info("Save results")
        # 保存结果

    logger.info("Evaluate semantic segmentation")
    sem_point_eval.get_eval(logger)
    # sem_point_eval.get_semantic_eval(logger) #AttributeError: 'PointWiseEval' object has no attribute 'get_semantic_eval'

    logger.info("Evaluate panoptic segmentation")
    instance_eval.get_eval(logger)
    # instance_eval.get_instance_eval(logger)

    
if __name__ == "__main__":
    main()
