import _init_paths
from net import Network
from config import cfg, update_config
from dataset import *
import numpy as np
import torch
import os
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar100_exp1.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--head-ratio",
        dest='head_ratio',
        required=False,
        default=0,
        type=int)

    parser.add_argument('--start', dest='start', default=0, type=int)
    parser.add_argument('--end', dest='end', default=21, type=int)

    args = parser.parse_args()
    return args


def load_label_map(cache_dir, head_ratio):
    import pickle
    save_path = os.path.join(cache_dir, 'cid_to_lcid_%d.bin' % (head_ratio))
    with open(save_path, 'rb') as f:
        label_map = pickle.load(f)
    return label_map


def valid_model(dataLoader, model, cfg, device, num_classes, label_map=None):

    all_labels = []
    all_result = []

    if label_map is not None:
        num_classes = label_map[:, 0].max() + 1

    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)
            result = func(output)
            top1_res = result.cpu().numpy().argmax(axis=1)
            image_labels = image_labels.numpy()

            if label_map is not None:
                image_labels = label_map[image_labels, 0]
                top1_res = label_map[top1_res, 0]

            fusion_matrix.update(top1_res, image_labels)
            all_labels.extend(image_labels.tolist())
            all_result.extend(top1_res.tolist())

            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                top1_count += [image_labels[i] == top1_res[i]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)
    pbar.close()
    top1_acc = float(np.sum(top1_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%".format(
            top1_acc * 100
        )
    )

    return fusion_matrix, all_labels, all_result


def tor_norm(model, tor=1):
    fc = model.classifier.state_dict()
    fc_w = fc['weight']
    fc_w_norm = torch.norm(fc_w, p=2, dim=1).unsqueeze(1)
    fc_w_norm_rep = fc_w_norm.repeat((1, fc_w.shape[1]))
    fc_w = fc_w / torch.pow(fc_w_norm_rep, tor)
    fc['weight'] = fc_w

    fc_b = fc['bias']
    fc_b[:] = 0
    fc['bias'] = fc_b
    model.classifier.load_state_dict(fc)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([i for i in range(fc_w.shape[0])], [fc_w_norm[i, 0].item() for i in range(fc_w.shape[0])])
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cpu" if cfg.CPU_MODE else "cuda")
    model = Network(cfg, mode="test", num_classes=num_classes)

    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "models")
    model_file = cfg.TEST.MODEL_FILE
    if "/" in model_file:
        model_path = model_file
    else:
        model_path = os.path.join(model_dir, model_file)
    model.load_model(model_path)

    head_ratio = args.head_ratio
    label_map = None
    if head_ratio > 0:
        cache_dir = 'datasets/imbalance_cifar10/cifar-100-cache'
        label_map = load_label_map(cache_dir, head_ratio)

    # fc normalization
    tor_norm(model)

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
    matrix, all_labels, all_result = valid_model(testLoader, model, cfg, device, num_classes, label_map=label_map)
    print('Pre from %d to %d: %.4f' % (args.start, args.end, matrix.get_pre_in_range(args.start, args.end)))
    print('Rec from %d to %d: %.4f' % (args.start, args.end, matrix.get_rec_in_range(args.start, args.end)))

    output = {'labels': all_labels, 'result': all_result}
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "analysis")
    save_path = os.path.join(save_dir, 'label_result.bin')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'wb') as f:
        pickle.dump(output, f)
    print('Evaluation result is saved at %s' % save_path)
