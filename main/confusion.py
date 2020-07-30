import _init_paths
from config import cfg, update_config
import argparse
import os
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns


def load_label_map(cache_dir, head_ratio):
    import pickle
    save_path = os.path.join(cache_dir, 'cid_to_lcid_%d.bin' % (head_ratio))
    with open(save_path, 'rb') as f:
        label_map = pickle.load(f)
    return label_map


def parse_args():
    parser = argparse.ArgumentParser(description="BBN evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default="configs/cifar100_baseline.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix', show_value=False):
    plt.figure(figsize=(40, 30))
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)

    if show_value:
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.001:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=10, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


# def plot_confusion_matrix2(cm, save_path, classes, title='confusion matrix', show_value=False):
#     sns.set()
#     f, ax = plt.subplots(figsize=(50, 40))
#     if show_value:
#         sns.heatmap(cm, annot=True, ax=ax)
#     else:
#         sns.heatmap(cm, ax=ax)
#     ax.set_title(title)
#     ax.set_xlabel('predict')
#     ax.set_ylabel('true')
#     plt.savefig(save_path, format='png')
    # plt.show()

# def convert_label_and_result(label_map, labels, result):
#     return label_map[labels, 0].tolist(), label_map[result, 0].tolist()


# head_ratio = 60
# cache_dir = 'datasets/imbalance_cifar10/cifar-100-cache'
# label_map = load_label_map(cache_dir, head_ratio)

args = parse_args()
update_config(cfg, args)
save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "analysis")
res_path = os.path.join(save_dir, 'label_result.bin')
sav_path = os.path.join(save_dir, 'confusion_matrix.png')
with open(res_path, 'rb') as f:
    res = pickle.load(f)
labels = res['labels']
result = res['result']

# labels, result = convert_label_and_result(label_map, labels, result)

num_classes = max(labels) + 1
cm = confusion_matrix(labels, result)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plot_confusion_matrix(cm_normalized, sav_path, [str(i) for i in range(num_classes)], title='confusion matrix', show_value=False)
# plot_confusion_matrix2(cm_normalized, sav_path, classes, title='confusion matrix', show_value=True)
