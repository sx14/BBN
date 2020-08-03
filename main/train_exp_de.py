import _init_paths
import sys
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

from core.function_exp import train_model, valid_model
from core.combiner import Combiner

import torch
import os, shutil
from torch.utils.data import DataLoader
import argparse
import warnings
import click
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="codes for BBN")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar100_exp5.yaml",
        type=str,
    )

    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= False,
    )

    parser.add_argument(
        "--cache_dir",
        dest='cache_dir',
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

    args = parser.parse_args()
    return args


def load_label_map(cache_dir, head_ratio, cluster_num):
    import pickle
    save_path = os.path.join(cache_dir, 'cid_to_lcid_%d_%d.bin' % (head_ratio, cluster_num))
    with open(save_path, 'rb') as f:
        label_map = pickle.load(f)

    # without virtual categories
    label_map_l1 = label_map[:, 0]
    label_map[label_map_l1 >= head_ratio, 0] = -1
    label_map = torch.Tensor(label_map).long()

    save_path = os.path.join(cache_dir, 'curr_lcid_to_next_lcid_%d_%d.bin' % (head_ratio, cluster_num))
    with open(save_path, 'rb') as f:
        level_label_maps = pickle.load(f)
    for level in range(len(level_label_maps)):
        level_label_maps[level] = torch.from_numpy(level_label_maps[level]).byte()
    return label_map, level_label_maps


def load_last_stage(model, weight_path):
    checkpoint = torch.load(weight_path)
    pre_state_dict = checkpoint['state_dict']
    mod_state_dict = model.state_dict()
    for k in mod_state_dict:
        if k in pre_state_dict:
            mod_state_dict[k] = pre_state_dict[k]
    model.load_state_dict(mod_state_dict)
    print('Load weight from %s' % weight_path)


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    logger, log_file = create_logger(cfg)
    warnings.filterwarnings("ignore")
    cudnn.benchmark = True
    auto_resume = args.auto_resume
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")

    train_set = eval(cfg.DATASET.DATASET)("train", cfg)
    valid_set = eval(cfg.DATASET.DATASET)("valid", cfg)

    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    num_class_list, cat_list = get_category_list(annotations, num_classes, cfg)

    para_dict = {
        "num_classes": num_classes,
        "num_class_list": num_class_list,
        "cfg": cfg,
        "device": device,
    }

    criterion = eval(cfg.LOSS.LOSS_TYPE)(para_dict=para_dict)
    epoch_number = cfg.TRAIN.MAX_EPOCH

    # ----- BEGIN MODEL BUILDER -----
    label_map, level_label_maps = load_label_map(args.cache_dir, cfg.HEAD_RATIO, cfg.CLUSTER_NUM)
    l1_cls_num = label_map[:, 0].max().item() + 1
    l2_cls_num = label_map[:, 1].max().item() + 1

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    code_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "codes")

    model = get_model(cfg, [l1_cls_num, l2_cls_num], device, logger)
    if cfg.TRAIN_STAGE == 2:
        # last_stage_weight_path = 'output/cifar100/CIFAR100.res32.200epoch/models/best_model.pth'
        last_stage_weight_path = os.path.join(model_dir, 'best_model_stage1.pth')
        load_last_stage(model, last_stage_weight_path)
        model.module.freeze_backbone()
        train_set.gen_class_balance_data()

    # load_pretrained_weight(model, args.pretrained_path)
    combiner = Combiner(cfg, device)
    optimizer = get_optimizer(cfg, model)
    scheduler = get_scheduler(cfg, optimizer)
    # ----- END MODEL BUILDER -----

    trainLoader = DataLoader(
        train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True
    )

    validLoader = DataLoader(
        valid_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )

    # close loop
    tensorboard_dir = (
        os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "tensorboard")
        if cfg.TRAIN.TENSORBOARD.ENABLE
        else None
    )

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    else:
        logger.info(
            "This directory has already existed, Please remember to modify your cfg.NAME"
        )
        if not click.confirm(
            "\033[1;31;40mContinue and override the former directory?\033[0m",
            default=False,
        ):
            exit(0)
        shutil.rmtree(code_dir)
        if tensorboard_dir is not None and os.path.exists(tensorboard_dir):
            shutil.rmtree(tensorboard_dir)
    print("=> output model will be saved in {}".format(model_dir))
    this_dir = os.path.dirname(__file__)
    ignore = shutil.ignore_patterns(
        "*.pyc", "*.so", "*.out", "*pycache*", "*.pth", "*build*", "*output*", "*datasets*"
    )
    shutil.copytree(os.path.join(this_dir, ".."), code_dir, ignore=ignore)

    if tensorboard_dir is not None:
        dummy_input = torch.rand((1, 3) + cfg.INPUT_SIZE).to(device)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        writer.add_graph(model if cfg.CPU_MODE else model.module, (dummy_input,))
    else:
        writer = None

    best_result, best_epoch, start_epoch = 0, 0, 1
    # ----- BEGIN RESUME ---------
    all_models = os.listdir(model_dir)
    if len(all_models) <= 1 or auto_resume == False:
        auto_resume = False
    else:
        all_models.remove("best_model.pth")
        resume_epoch = max([int(name.split(".")[0].split("_")[-1]) for name in all_models])
        resume_model_path = os.path.join(model_dir, "epoch_{}.pth".format(resume_epoch))

    if cfg.RESUME_MODEL != "" or auto_resume:
        if cfg.RESUME_MODEL == "":
            resume_model = resume_model_path
        else:
            resume_model = cfg.RESUME_MODEL if '/' in cfg.RESUME_MODEL else os.path.join(model_dir, cfg.RESUME_MODEL)
        logger.info("Loading checkpoint from {}...".format(resume_model))
        checkpoint = torch.load(
            resume_model, map_location="cpu" if cfg.CPU_MODE else "cuda"
        )
        if cfg.CPU_MODE:
            model.load_model(resume_model)
        else:
            model.module.load_model(resume_model)
        if cfg.RESUME_MODE != "state_dict":
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            best_result = checkpoint['best_result']
            best_epoch = checkpoint['best_epoch']
    # ----- END RESUME ---------

    logger.info(
        "-------------------Train start :{}  {}  {}-------------------".format(
            cfg.BACKBONE.TYPE, cfg.MODULE.TYPE, cfg.TRAIN.COMBINER.TYPE
        )
    )

    for epoch in range(start_epoch, epoch_number + 1):
        scheduler.step()
        train_model(
            trainLoader,
            model,
            epoch,
            epoch_number,
            optimizer,
            combiner,
            criterion,
            cfg,
            logger,
            label_map,
            stage=cfg.TRAIN_STAGE,
            writer=writer,
        )
        model_save_path = os.path.join(
            model_dir,
            "epoch_{}_stage{}.pth".format(epoch, cfg.TRAIN_STAGE),
        )
        if epoch % cfg.SAVE_STEP == 0:
            torch.save({
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'best_result': best_result,
                'best_epoch': best_epoch,
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict()
            }, model_save_path)

        # loss_dict, acc_dict = {"train_loss": train_loss}, {"train_acc": train_acc}
        if cfg.VALID_STEP != -1 and epoch % cfg.VALID_STEP == 0:
            valid_acc = valid_model(
                validLoader,
                epoch,
                model,
                logger,
                device,
                label_map,
                level_label_maps,
                stage=cfg.TRAIN_STAGE,
            )
            # loss_dict["valid_loss"], acc_dict["valid_acc"] = valid_loss, valid_acc
            if valid_acc > best_result:
                best_result, best_epoch = valid_acc, epoch
                torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'best_result': best_result,
                        'best_epoch': best_epoch,
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                }, os.path.join(model_dir, "best_model_stage%d.pth" % cfg.TRAIN_STAGE)
                )
            logger.info(
                "--------------Best_Epoch:{:>3d}    Best_Acc:{:>5.4f}--------------".format(
                    best_epoch, best_result
                )
            )
        # if cfg.TRAIN.TENSORBOARD.ENABLE:
        #     writer.add_scalars("scalar/acc", acc_dict, epoch)
        #     writer.add_scalars("scalar/loss", loss_dict, epoch)
    if cfg.TRAIN.TENSORBOARD.ENABLE:
        writer.close()
    logger.info(
        "-------------------Train Finished :{}-------------------".format(cfg.NAME)
    )
