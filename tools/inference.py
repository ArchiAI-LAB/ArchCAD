import yaml
from munch import Munch
import glob, tqdm
import os.path as osp
import numpy as np
import torch
import torchvision.transforms as T
from typing import List, Dict
import random


from pathlib import Path
from svgnet.model.svgnet import SVGNet as svgnet
#
from svgnet.data.svg import SVGDataset  #
from svgnet.util import get_root_logger, load_checkpoint
from svgnet.evaluation import PointWiseEval, InstanceEval

import time

def imagenet_preprocess():
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    return T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def process_svgnet(config_path: str, checkpoint_path: str, datadir: str, out: str) -> None:
    """
    处理SVG文件的函数
    
    Args:
        config_path (str): 配置文件路径
        checkpoint_path (str): 模型检查点路径
        datadir (str): 数据集目录
        out (str): 输出结果保存路径
    """
    # 读取配置文件
    cfg_txt = open(config_path, "r").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger = get_root_logger()

    # 图像预处理变换
    transform = [T.Resize((cfg.data.test.img_size, cfg.data.test.img_size))]
    transform.append(T.ToTensor())
    transform.append(imagenet_preprocess())
    img_transform = T.Compose(transform)
    
    # 初始化模型
    model = svgnet(cfg.model).cuda()
   
    logger.info(f"Load state dict from {checkpoint_path}")
    load_checkpoint(checkpoint_path, logger, model)
    
    # 获取数据文件列表
    data_list = glob.glob(osp.join(datadir, "*_s2.json"))
    # random.shuffle(data_list)
    # data_list = data_list[:3000]
    # print("data_list:",data_list)
    """
    data_list: ['/home/ecadi/DPSS_for_use/dataset/0a0a6ee1-9016-47b6-b720-e70fa7db6b2e_s2.json', '/home/ecadi/DPSS_for_use/dataset/0a22ab9c-6c8e-475d-a7ee-d6c74bbc18d0_s2.json', '/home/ecadi/DPSS_for_use/dataset/0a5c21e5-6de9-4275-847f-cbef241e857c_s2.json', '/home/ecadi/DPSS_for_use/dataset/0a72accb-fde0-4fc1-ae27-b7dd9544325d_s2.json', '/home/ecadi/DPSS_for_use/dataset/0a3cae24-9a2b-4983-9c26-73af3776134f_s2.json', '/home/ecadi/DPSS_for_use/dataset/0a0c4a0b-b5c5-40b0-ab06-39e0f4d2073f_s2.json',
     '/home/ecadi/DPSS_for_use/dataset/0a4a15ed-41fc-4337-b478-4c6c869aadc0_s2.json']
    """
    logger.info(f"Load dataset: {len(data_list)} svg")

    # 初始化评估器
    sem_point_eval = PointWiseEval(
        num_classes=cfg.model.semantic_classes,
        ignore_label=cfg.model.semantic_classes,
        gpu_num=1
    )
    instance_eval = InstanceEval(
        num_classes=cfg.model.semantic_classes,
        ignore_label=cfg.model.semantic_classes,
        gpu_num=1
    )
    
    save_dicts: List[Dict] = []
    total_times: List[float] = []
    
    with torch.no_grad():
        model.eval()
        for svg_file in tqdm.tqdm(data_list):
            #coords, feats, labels, lengths, layerIds, img, bound, json_file = SVGDataset.load(file_name=svg_file, idx=1)
            #print("file_name:",svg_file) #/home/ecadi/DPSS_for_use/dataset/0a0a6ee1-9016-47b6-b720-e70fa7db6b2e_s2.json
            file_name = osp.splitext(osp.basename(svg_file))[0].replace("_s2", "")
            coords, feats, labels, lengths, layerIds, img, bound, json_file = SVGDataset.load(
                data_root=datadir, file_name=file_name, idx=1, min_points=2048
            )

            centers = coords[:,:2].copy()
            centers = centers * 2 - 1
            coords -= np.mean(coords, 0)
            offset = [coords.shape[0]]
            layerIds = torch.LongTensor(layerIds)
            offset = torch.IntTensor(offset)
            coords = torch.FloatTensor(coords).cuda()
            feats = torch.FloatTensor(feats).cuda()
            labels = torch.LongTensor(labels).cuda()
            
            img = img_transform(img)
            img = torch.FloatTensor(img).cuda()
            centers = torch.FloatTensor(centers).cuda()
            img = [img]
            centers = [centers]

            batch = (coords, feats, labels, offset, torch.FloatTensor(lengths), 
                    layerIds, img, centers, json_file)
            
            torch.cuda.empty_cache()
            
            with torch.cuda.amp.autocast(enabled=cfg.fp16):
                t1 = time.time()
                res = model(batch, return_loss=False)
                t2 = time.time()
                total_times.append(t2-t1)
                
                sem_preds = torch.argmax(res["semantic_scores"], dim=1).cpu().numpy()
                sem_gts = res["semantic_labels"].cpu().numpy()
                #sem_point_eval.update(sem_preds, sem_gts, lengths) #    def update(self, pred_sem, gt_sem):
                sem_point_eval.update(sem_preds, sem_gts) #    def update(self, pred_sem, gt_sem):
                instance_eval.update(
                    res["instances"],
                    res["targets"],
                    res["lengths"],
                )
                save_dicts.append({
                    "filepath": svg_file.replace(".json", ".svg"),
                    "sem": res["semantic_scores"].cpu().numpy(),
                    "ins": res["instances"],
                    "targets": res["targets"],
                    "lengths": res["lengths"],
                })            
    # 保存结果
    save_path = osp.join(out, 'sem_ins_split_val.npy')

    np.save(save_path, save_dicts)   
    print("保存路径为：", save_path)

    logger.info("Evaluate semantic segmentation")
    sem_point_eval.get_eval(logger)
    #sem_point_eval.get_semantic_eval(logger) #AttributeError: 'PointWiseEval' object has no attribute 'get_semantic_eval'

    logger.info("Evaluate panoptic segmentation")
    instance_eval.get_eval(logger)
    #instance_eval.get_instance_eval(logger)


if __name__ == "__main__":
    config_path = "/data/ecadi/DPSS_for_use/configs/svg/infer.yaml"
    checkpoint_path = "/data/ecadi/DPSS_for_use/best_model/epoch_72.pth"
    datadir = '/data/share/data_30_cat_486dxf_no_overlap/data'
    out = '/data/share/data_30_cat_486dxf_no_overlap/data'
    process_svgnet(config_path, checkpoint_path, datadir, out)

