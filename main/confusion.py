from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix', show_value=False):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)

    if show_value:
        for x_val, y_val in zip(x.flatten(), y.flatten()):
            c = cm[y_val][x_val]
            if c > 0.001:
                plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

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


def plot_confusion_matrix2(cm, save_path, classes, title='confusion matrix', show_value=False):
    sns.set()
    f, ax = plt.subplots(figsize=(50, 40))
    if show_value:
        sns.heatmap(cm, annot=True, ax=ax)
    else:
        sns.heatmap(cm, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    plt.savefig(save_path, format='png')
    # plt.show()


tag = 'exp1'
infer = 'flat'
level = 3
classes_dict = [13, 37, 122, 200]
classes = [i for i in range(classes_dict[level])]

res_path = 'output/tiny-imagenet-200/exp_hier/results_%s_%s.bin' % (tag, infer)
sav_path = 'cache/CUB_200_2011/exp_hier/cm_%s_%s_%d.png' % (tag, infer, level)

import pickle
with open(res_path, 'rb') as f:
    all_res = pickle.load(f)

preds, labels = all_res[level]
tp_cnt = [1 for i in range(len(preds)) if preds[i] == labels[i]]
prec = sum(tp_cnt) * 1.0 / len(preds)
print(prec)
# plt.hist(labels)
# plt.show()


cm = confusion_matrix(labels, preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plot_confusion_matrix(cm_normalized, sav_path, classes, title='confusion matrix', show_value=False)
plot_confusion_matrix2(cm_normalized, sav_path, classes, title='confusion matrix', show_value=True)
