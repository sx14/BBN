import _init_paths
import sys
import pickle
from loss import *
from dataset import *
from config import cfg, update_config
from utils.utils import (
    create_logger,
    get_optimizer,
    get_scheduler,
    get_model,
    get_category_list,
)
from core.function import train_model, valid_model
from core.combiner import Combiner
from core.hierarchy import Tree, Node

import torch
import numpy as np
import os, shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from sklearn.cluster import AgglomerativeClustering
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar100_baseline.yaml",
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        dest='save_dir',
        required=False,
        default="datasets/imbalance_cifar10/cifar-100-cache",
        type=str,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= False,
    )

    parser.add_argument('--level_num',
                        dest='level_num',
                        default=2,
                        type=int)
    parser.add_argument('--cluster_num',
                        dest='cluster_num',
                        default=2,
                        type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True
    auto_resume = args.auto_resume

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    # ----- BEGIN MODEL BUILDER -----
    model = get_model(cfg, num_classes, device, logger)
    # ----- END MODEL BUILDER -----

    # ----- BEGIN RESUME ---------
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    resume_model = os.path.join(model_dir, "best_model.pth")

    logger.info("Loading checkpoint from {}...".format(resume_model))
    if cfg.CPU_MODE:
        model.load_model(resume_model)
        next_level_centers = model.extract_classifier_weight()
    else:
        model.module.load_model(resume_model)
        next_level_centers = model.module.extract_classifier_weight()
        next_level_centers = next_level_centers.cpu()
    # ----- END RESUME ---------
    cluster_num = args.cluster_num
    level_num = args.level_num
    level_ranges = []
    if level_num == 2:
        # num_per_cls_dict = train_set.num_per_cls_dict
        # min_diff = num_per_cls_dict[0]
        # flag = -1
        # for i in range(num_classes-1):
        #     part1 = sum([num_per_cls_dict[j] for j in range(0, i+1)])
        #     part2 = sum([num_per_cls_dict[j] for j in range(i+1, num_classes)])
        #     if abs(part1 - part2) < min_diff:
        #         min_diff = abs(part1 - part2)
        #         flag = i
        # level_ranges = [[0, flag], [flag + 1, num_classes]]
        level_ranges = [(0, int(num_classes * 0.6)),
                        (int(num_classes * 0.6), num_classes)]
    else:
        raise ValueError('Level num = 2 only.')

    cid_to_lcid = np.ones((num_classes, level_num)) * (-1)
    cid_to_lcid = cid_to_lcid.astype(int)

    curr_lcid_to_next_lcid = [None] * (level_num - 1)
    for level in range(level_num):
        curr_level_start, curr_level_end = level_ranges[level]
        curr_level_class_num = curr_level_end - curr_level_start
        for class_id in range(curr_level_start, curr_level_end):
            level_class_id = class_id - curr_level_start
            cid_to_lcid[class_id, level] = level_class_id

        if level < level_num - 1:
            next_level_start, next_level_end = level_ranges[level+1]
            next_level_class_num = next_level_end - next_level_start
            curr_lcid_to_next_lcid[level] = np.zeros((curr_level_class_num + cluster_num,
                                                      next_level_class_num)).astype(int)
            next_level_centers = next_level_centers[next_level_start: next_level_end]
            clustering = AgglomerativeClustering(
                linkage='ward',
                n_clusters=cluster_num,
                affinity='euclidean').fit(next_level_centers)

            cluster_logits = clustering.labels_
            for i in range(cluster_num):
                lcids = np.where(cluster_logits == i)[0]
                cids = lcids + next_level_start
                cid_to_lcid[cids, level] = curr_level_class_num + i
                curr_lcid_to_next_lcid[level][curr_level_class_num + i, lcids] = 1

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, 'cid_to_lcid.bin')
    with open(save_path, 'wb') as f:
        pickle.dump(cid_to_lcid, f)
    print('Class map is saved at %s.' % save_path)

    save_path = os.path.join(args.save_dir, 'curr_lcid_to_next_lcid.bin')
    with open(save_path, 'wb') as f:
        pickle.dump(curr_lcid_to_next_lcid, f)
    print('Class map is saved at %s.' % save_path)