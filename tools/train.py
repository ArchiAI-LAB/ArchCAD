import numpy as np
import torch
import torch.nn as nn
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
import datetime
import os
import os.path as osp
import shutil
import time
# import pdb

from svgnet.data import build_dataloader, build_dataset
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.model.criterion import SetCriterion
from svgnet.model.matcher import HungarianMatcher
from svgnet.evaluation import PointWiseEval,InstanceEval
from svgnet.util import (
    AverageMeter,
    SummaryWriter,
    build_optimizer,
    checkpoint_save,
    cosine_lr_after_step,
    fixed_lr,
    custimized_lr,
    get_dist_info,
    get_max_memory,
    get_root_logger,
    init_dist,
    is_main_process,
    is_multiple,
    is_power2,
    load_checkpoint,
    set_seed,
    get_scheduler,
    build_new_optimizer
)



def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    parser.add_argument("--resume", type=str, help="path to resume from")
    parser.add_argument("--work_dir", type=str, help="working directory")
    parser.add_argument("--skip_validate", action="store_true", help="skip validation")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--exp_name", type=str, default="default")
    
    args = parser.parse_args()
    return args


def train(epoch, model, optimizer, scheduler, scaler, train_loader, cfg, logger, writer):

    model.train()
    iter_time = AverageMeter(True)
    data_time = AverageMeter(True)
    meter_dict = {}
    end = time.time()
    # 
    lr = optimizer.param_groups[0]["lr"]  # 确保lr有初值

    if train_loader.sampler is not None and cfg.dist:
        train_loader.sampler.set_epoch(epoch)

    for i, batch in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if scheduler is None:
            cosine_lr_after_step(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)
            # custimized_lr(optimizer, cfg.optimizer.lr, epoch - 1, cfg.step_epoch, cfg.epochs)

        with torch.cuda.amp.autocast(enabled=cfg.fp16):
            try:
                _, loss, log_vars = model(batch)

            except Exception as e:
                print(f"Error in file {batch[-1]}")
                print(f"Error encountered for batch: {e}")
                continue # 
                loss = torch.tensor(0.1, device=batch[0].device)
                log_vars = {}

            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            
        for k, v in log_vars.items():
            if k not in meter_dict.keys() and k != "placeholder":
                meter_dict[k] = AverageMeter()
            meter_dict[k].update(v)
        
        
        # backward  
        if loss>0:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        for name, param in model.named_parameters():
            if param.grad is None:
                print(name)
        
        # time and print
        remain_iter = len(train_loader) * (cfg.epochs - epoch + 1) - i
        iter_time.update(time.time() - end)
        end = time.time()
        remain_time = remain_iter * iter_time.avg
        remain_time = str(datetime.timedelta(seconds=int(remain_time)))
        lr = optimizer.param_groups[0]["lr"]

        if is_multiple(i, 50):
            log_str = f"Epoch [{epoch}/{cfg.epochs}][{i}/{len(train_loader)}]  "
            log_str += (
                f"lr: {lr:.2g}, eta: {remain_time}, mem: {get_max_memory()}, "
                f"data_time: {data_time.val:.2f}, iter_time: {iter_time.val:.2f}"
            )
            for k, v in meter_dict.items():
                log_str += f", {k}: {v.val:.4f}"
            logger.info(log_str)
    writer.add_scalar("train/learning_rate", lr, epoch)
    for k, v in meter_dict.items():
        writer.add_scalar(f"train/{k}", v.avg, epoch)
    checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq)


# train 改完不要忘记val

def validate(epoch, model, optimizer, val_loader, cfg, logger, writer):
    logger.info("Validation")
    sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,ignore_label=49,gpu_num=dist.get_world_size())
    instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes,ignore_label=49,gpu_num=dist.get_world_size())
    meter_dict = {}
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(val_loader):

            try:
                with torch.cuda.amp.autocast(enabled=cfg.fp16):
                    res,loss, log_vars = model(batch)
                sem_preds = torch.argmax(res["semantic_scores"],dim=1).cpu().numpy()
                sem_gts = res["semantic_labels"].cpu().numpy()
                sem_point_eval.update(sem_preds, sem_gts)
                instance_eval.update(
                    res["instances"],
                    res["targets"],
                    res["lengths"],
                )
                # meter_dict
                for k, v in log_vars.items():
                    if k not in meter_dict.keys() and k != "placeholder":
                        meter_dict[k] = AverageMeter()
                    meter_dict[k].update(v)
            except:
                print(f"Error in file {batch[-1]}")

    global best_metric

    logger.info("Evaluate semantic segmentation")
    miou,acc = sem_point_eval.get_eval(logger)
    logger.info("Evaluate panoptic segmentation")
    sPQ, sRQ, sSQ = instance_eval.get_eval(logger)
    for k, v in meter_dict.items():
        writer.add_scalar(f"val/{k}", v.avg, epoch)
        
    writer.add_scalar("val/mIoU", miou, epoch)
    writer.add_scalar("val/Acc", acc, epoch)
    writer.add_scalar("val/sPQ", sPQ, epoch)
    writer.add_scalar("val/sRQ", sRQ, epoch)
    writer.add_scalar("val/sSQ", sSQ, epoch)


    if best_metric < sPQ:
        best_metric = sPQ
        checkpoint_save(epoch, model, optimizer, cfg.work_dir, cfg.save_freq, best=True)
        logger.info(f"New best sPQ {best_metric:.3f} at {epoch} epoch" )
    
       

def main():
    args = get_args()

    cfg_txt = open(args.config, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))

    if args.dist:
        rank = init_dist()
        set_seed(args.seed + rank)    
    cfg.dist = args.dist
    
    # work_dir & logger
    if args.work_dir:
        cfg.work_dir = args.work_dir
    else:
        dataset_name = cfg.data.train.type
        cfg.work_dir = osp.join("./work_dirs", dataset_name, osp.splitext(osp.basename(args.config))[0], args.exp_name)

    os.makedirs(osp.abspath(cfg.work_dir), exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    logger.info(f"Config:\n{cfg_txt}")
    logger.info(f"Distributed: {args.dist}")
    logger.info(f"Mix precision training: {cfg.fp16}")
    shutil.copy(args.config, osp.join(cfg.work_dir, osp.basename(args.config)))
    writer = SummaryWriter(cfg.work_dir)

    logger.info(f"Save at: {cfg.work_dir}")

    # criterion
    matcher = HungarianMatcher(**cfg.matcher)
    weight_dict = {
            "loss_ce": cfg.matcher.cost_class, 
            "loss_mask": cfg.matcher.cost_mask, 
            "loss_dice": cfg.matcher.cost_dice,
            }
    criterion = SetCriterion(matcher,weight_dict,cfg.criterion).cuda()
    
    model = svgnet(cfg.model, criterion=criterion).cuda()

    if args.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
    #logger.info(model)
    
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
    
    logger.info('Total Number of Parameters: {} M'.format(str(float(total_params) / 1e6)[:5]))
    logger.info('Total Trainable Number of Parameters: {} M'.format(str(float(trainable_params) / 1e6)[:5]))
        

    if args.dist:
        model = DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()], find_unused_parameters=(trainable_params < total_params)
        )
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.fp16)

    # data
    train_set = build_dataset(cfg.data.train, logger)
    val_set = build_dataset(cfg.data.test, logger)

    train_loader = build_dataloader(args,train_set, training=True, dist=args.dist, **cfg.dataloader.train)
    val_loader = build_dataloader(args,val_set, training=False, dist=args.dist, **cfg.dataloader.test)#args.dist

    # optim
    default_lr = cfg.optimizer.lr  # default for batch 16
    _, world_size = get_dist_info()
    total_batch_size = cfg.dataloader.train.batch_size * world_size
    scaled_lr = default_lr * (total_batch_size / 16)
    cfg.optimizer.lr = scaled_lr
    logger.info(f"Scale LR from {default_lr} (batch size 16) to {scaled_lr} (batch size {total_batch_size})")
    optimizer = build_new_optimizer(model, cfg.optimizer)
    # scheduler
    #scheduler = get_scheduler(cfg.scheduler,optimizer)
    scheduler = None
    
    # pretrain, resume
    start_epoch = 1
    if args.resume:
        logger.info(f"Resume from {args.resume}")
        start_epoch = load_checkpoint(args.resume, logger, model, optimizer=optimizer)
    elif cfg.pretrain:
        logger.info(f"Load pretrain from {cfg.pretrain}")
        load_checkpoint(cfg.pretrain, logger, model)

    global best_metric
    best_metric = 0

    # if is_main_process():
    #     validate(0, model, optimizer, val_loader, cfg, logger, writer)

    # train and val
    logger.info("Training")
    for epoch in range(start_epoch, cfg.epochs + 1):
        train(epoch, model, optimizer, scheduler, scaler, train_loader, cfg, logger, writer)
        if scheduler is not None:scheduler.step()
        validate(epoch, model, optimizer, val_loader, cfg, logger, writer)
        writer.flush()

    logger.info(f"Finish!!! Model at: {cfg.work_dir}")


if __name__ == "__main__":
    main()
 

 