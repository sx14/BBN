import _init_paths
from net import Network1, Network2
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix, AverageMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        default="configs/cifar100_exp2.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument(
        "--cache_dir",
        dest='cache_dir',
        required=False,
        default="datasets/imbalance_cifar10/cifar-100-cache",
        type=str,
    )
    args = parser.parse_args()
    return args


def tor_norm(model, tor=1):
    fc = model.classifiers.state_dict()
    fc_w1 = fc['0.weight']
    fc_w_norm1 = torch.norm(fc_w1, p=2, dim=1).unsqueeze(1)
    fc_w_norm1_rep = fc_w_norm1.repeat((1, fc_w1.shape[1]))
    fc_w1 = fc_w1 / torch.pow(fc_w_norm1_rep, tor)
    fc['0.weight'] = fc_w1

    fc_w2 = fc['1.weight']
    fc_w_norm2 = torch.norm(fc_w2, p=2, dim=1).unsqueeze(1)
    fc_w_norm2_rep = fc_w_norm2.repeat((1, fc_w2.shape[1]))
    fc_w2 = fc_w2 / torch.pow(fc_w_norm2_rep, tor)
    fc['1.weight'] = fc_w2

    fc_b1 = fc['0.bias']
    fc_b1[:] = 0
    fc['0.bias'] = fc_b1

    fc_b2 = fc['1.bias']
    fc_b2[:] = 0
    fc['1.bias'] = fc_b2

    model.classifiers.load_state_dict(fc)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([i for i in range(fc_w_norm1.shape[0])], [fc_w_norm1[i, 0].item() for i in range(fc_w_norm1.shape[0])])
    plt.show()


def valid_model(
        dataLoader,
        model,
        device,
        label_map,
        level_label_maps):

    model.eval()
    num_levels = label_map.shape[1]
    num_classes = dataLoader.dataset.get_num_classes()

    l1_cls_num = label_map[:, 0].max().item() + 1
    l2_cls_num = label_map[:, 1].max().item() + 1
    virtual_cls_num = l1_cls_num + l2_cls_num - num_classes
    l1_raw_cls_num = l1_cls_num - virtual_cls_num

    fusion_matrix1 = FusionMatrix(num_classes)
    fusion_matrix2 = FusionMatrix(l1_cls_num)
    func = torch.nn.Softmax(dim=1)

    # 20 + 80
    acc1 = AverageMeter()
    p1_acc1 = AverageMeter()
    p2_acc1 = AverageMeter()
    level_accs = [AverageMeter() for _ in range(num_levels)]
    all_labels1 = []
    all_result1 = []

    # 20 + 2
    acc2 = AverageMeter()
    p1_acc2 = AverageMeter()
    p2_acc2 = AverageMeter()
    all_labels2 = []
    all_result2 = []

    with torch.no_grad():
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            batch_size = label.shape[0]
            level_scores = []
            level_probs = []
            for level in range(num_levels):

                level_score = model(image, level)
                level_scores.append(level_score)

                # for each level
                level_label = label_map[label, level]
                level_mask = level_label >= 0
                level_label1 = level_label[level_mask]
                level_score1 = level_score[level_mask]
                level_top1 = torch.argmax(level_score1, 1)
                level_acc1, level_cnt1 = accuracy(level_top1.cpu().numpy(), level_label1.cpu().numpy())
                level_accs[level].update(level_acc1, level_cnt1)

                if level == 0:
                    level_prob = func(level_score)
                    level_probs.append(level_prob)
                else:
                    high_lcid_to_curr_lcid = level_label_maps[level-1]
                    level_prob = torch.zeros(level_score.shape).cuda()
                    for high_lcid in range(high_lcid_to_curr_lcid.shape[0]):
                        curr_lcid_mask = high_lcid_to_curr_lcid[high_lcid]
                        if curr_lcid_mask.sum().item() > 0:
                            level_prob[:, curr_lcid_mask] = func(level_score[:, curr_lcid_mask])
                    level_probs.append(level_prob)

            # =================== 20 + 80 ============================
            all_probs = torch.ones((batch_size, num_classes)).cuda()
            for level in range(num_levels):
                level_prob = level_probs[level]
                related_lcids = label_map[:, level]
                related_lcids = related_lcids[related_lcids >= 0]
                unrelated_class_num1 = (label_map[:, level] < 0).sum().item()
                unrelated_class_num2 = label_map.shape[0] - related_lcids.shape[0]
                assert unrelated_class_num1 == unrelated_class_num2
                all_probs[:, unrelated_class_num1:] *= level_prob[:, related_lcids]

            l1_mask1 = label < l1_raw_cls_num
            l1_scores1 = all_probs[l1_mask1]
            l1_labels1 = label[l1_mask1]
            l1_result1 = torch.argmax(l1_scores1, 1)
            l1_now_acc1, l1_cnt1 = accuracy(l1_result1.cpu().numpy(), l1_labels1.cpu().numpy())
            p1_acc1.update(l1_now_acc1, l1_cnt1)

            l2_mask1 = label >= l1_raw_cls_num
            l2_scores1 = all_probs[l2_mask1]
            l2_labels1 = label[l2_mask1]
            l2_result1 = torch.argmax(l2_scores1, 1)
            l2_now_acc1, l2_cnt1 = accuracy(l2_result1.cpu().numpy(), l2_labels1.cpu().numpy())
            p2_acc1.update(l2_now_acc1, l2_cnt1)

            now_result = torch.argmax(all_probs, 1)
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc1.update(now_acc, cnt)
            fusion_matrix1.update(now_result.cpu().numpy(), label.cpu().numpy())
            all_labels1.extend(label.cpu().numpy().tolist())
            all_result1.extend(now_result.cpu().numpy().tolist())
            # ====================================================================

            # ===================20 + 2 =================================
            l1v_scores = level_probs[0]
            l1v_labels = label_map[label, 0]

            l1_mask2 = l1v_labels < l1_raw_cls_num
            l1_scores2 = l1v_scores[l1_mask2]
            l1_labels2 = l1v_labels[l1_mask2]
            l1_result2 = torch.argmax(l1_scores2, 1)
            l1_now_acc2, l1_cnt2 = accuracy(l1_result2.cpu().numpy(), l1_labels2.cpu().numpy())
            p1_acc2.update(l1_now_acc2, l1_cnt2)

            l2_mask2 = l1v_labels >= l1_raw_cls_num
            l2_scores2 = l1v_scores[l2_mask2]
            l2_labels2 = l1v_labels[l2_mask2]
            l2_result2 = torch.argmax(l2_scores2, 1)
            l2_now_acc2, l2_cnt2 = accuracy(l2_result2.cpu().numpy(), l2_labels2.cpu().numpy())
            p2_acc2.update(l2_now_acc2, l2_cnt2)

            l1v_result = torch.argmax(l1v_scores, 1)
            l1v_now_acc, l1v_cnt = accuracy(l1v_result.cpu().numpy(), l1v_labels.cpu().numpy())
            acc2.update(l1v_now_acc, l1v_cnt)
            fusion_matrix2.update(l1v_result.cpu().numpy(), l1v_labels.cpu().numpy())
            all_labels2.extend(l1v_labels.cpu().numpy().tolist())
            all_result2.extend(l1v_result.cpu().numpy().tolist())
            # ====================================================================

    print('Acc (head+tail): %.4f %d' % (acc1.avg, acc1.count))
    print('Acc P1         : %.4f %d' % (p1_acc1.avg, p1_acc1.count))
    print('Acc P2         : %.4f %d' % (p2_acc1.avg, p2_acc1.count))
    print('Acc L1         : %.4f %d' % (level_accs[0].avg, level_accs[0].count))
    print('Acc L2         : %.4f %d' % (level_accs[1].avg, level_accs[1].count))
    print('=' * 23)
    print('Acc (head+v)   : %.4f %d' % (acc2.avg, acc2.count))
    print('Acc P1         : %.4f %d' % (p1_acc2.avg, p1_acc2.count))
    print('Acc Pv         : %.4f %d' % (p2_acc2.avg, p2_acc2.count))
    print('=' * 23)
    return fusion_matrix1, fusion_matrix2, all_labels1, all_result1, all_labels2, all_result2


def load_label_map(cache_dir, head_ratio, cluster_num):
    import pickle
    save_path = os.path.join(cache_dir, 'cid_to_lcid_%d_%d.bin' % (head_ratio, cluster_num))
    with open(save_path, 'rb') as f:
        label_map = pickle.load(f)
    label_map = torch.Tensor(label_map).long()

    save_path = os.path.join(cache_dir, 'curr_lcid_to_next_lcid_%d_%d.bin' % (head_ratio, cluster_num))
    with open(save_path, 'rb') as f:
        level_label_maps = pickle.load(f)
    for level in range(len(level_label_maps)):
        level_label_maps[level] = torch.from_numpy(level_label_maps[level]).byte()
    return label_map, level_label_maps


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    label_map, level_label_maps = load_label_map(args.cache_dir, cfg.HEAD_RATIO, cfg.CLUSTER_NUM)
    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    l1_cls_num = label_map[:, 0].max().item() + 1
    l2_cls_num = label_map[:, 1].max().item() + 1
    virtual_cls_num = l1_cls_num + l2_cls_num - num_classes
    l1_raw_cls_num = l1_cls_num - virtual_cls_num
    l2_raw_cls_num = l2_cls_num
    if cfg.MULTI_BRANCH:
        model = Network2(cfg, mode="test", num_classes=[l1_cls_num, l2_cls_num])
    else:
        model = Network1(cfg, mode="test", num_classes=[l1_cls_num, l2_cls_num])

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)

    # tor_norm(model)

    if cfg.CPU_MODE:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model).cuda()

    testLoader = DataLoader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    matrix1, matrix2, all_labels1, all_result1, all_labels2, all_result2 = valid_model(testLoader, model, device, label_map, level_label_maps)
    head_rec = matrix1.get_rec_in_range(0, l1_raw_cls_num-1)
    head_pre = matrix1.get_pre_in_range(0, l1_raw_cls_num-1)
    tail_rec = matrix1.get_rec_in_range(l1_raw_cls_num, num_classes-1)
    tail_pre = matrix1.get_pre_in_range(l1_raw_cls_num, num_classes-1)
    print('Rec [%d - %d]: %.4f' % (0, l1_raw_cls_num-1, head_rec))
    print('Pre [%d - %d]: %.4f' % (0, l1_raw_cls_num-1, head_pre))
    print('Rec [%d - %d]: %.4f' % (l1_raw_cls_num, num_classes-1, tail_rec))
    print('Pre [%d - %d]: %.4f' % (l1_raw_cls_num, num_classes-1, tail_pre))
    print('Prediction ratio [%d-%d]: %f' % (
        0, l1_raw_cls_num - 1,
        head_rec * (l1_raw_cls_num * 1.0 / num_classes) / head_pre))

    print('=' * 30)

    head_rec = matrix2.get_rec_in_range(0, l1_raw_cls_num-1)
    head_pre = matrix2.get_pre_in_range(0, l1_raw_cls_num-1)
    tail_rec = matrix2.get_rec_in_range(l1_raw_cls_num, l1_cls_num-1)
    tail_pre = matrix2.get_pre_in_range(l1_raw_cls_num, l1_cls_num-1)
    print('Rec [%d - %d]: %.4f' % (0, l1_raw_cls_num-1, head_rec))
    print('Pre [%d - %d]: %.4f' % (0, l1_raw_cls_num-1, head_pre))
    print('Rec [%d - %d]: %.4f' % (l1_raw_cls_num, l1_cls_num-1, tail_rec))
    print('Pre [%d - %d]: %.4f' % (l1_raw_cls_num, l1_cls_num-1, tail_pre))
    print('Prediction ratio [%d-%d]: %f' % (
        0, l1_raw_cls_num - 1,
        head_rec * (l1_raw_cls_num * 1.0 / num_classes) / head_pre))

    output1 = {'labels': all_labels1, 'result': all_result1}
    output2 = {'labels': all_labels2, 'result': all_result2}

    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "analysis")
    save_path = os.path.join(save_dir, 'label_result1.bin')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'wb') as f:
        pickle.dump(output1, f)
    print('Evaluation result is saved at %s' % save_path)

    save_path = os.path.join(save_dir, 'label_result2.bin')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'wb') as f:
        pickle.dump(output2, f)
    print('Evaluation result is saved at %s' % save_path)


