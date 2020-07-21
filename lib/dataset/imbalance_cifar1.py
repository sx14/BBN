# To ensure fairness, we use the same code in LDAM (https://github.com/kaidic/LDAM-DRW) to produce long-tailed CIFAR datasets.

import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random


class IMBALANCECIFAR10S(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, mode, cfg, root = './datasets/imbalance_cifar10', imb_type='exp',
                 transform=None, target_transform=None, download=True):
        train = True if mode == "train" else False
        super(IMBALANCECIFAR10S, self).__init__(root, train, transform, target_transform, download)
        self.cfg = cfg
        self.level = cfg.LEVEL
        self.train = train
        self.dual_sample = True if cfg.TRAIN.SAMPLER.DUAL_SAMPLER.ENABLE and self.train else False
        rand_number = cfg.DATASET.IMBALANCECIFAR.RANDOM_SEED
        if self.train:
            np.random.seed(rand_number)
            random.seed(rand_number)
            imb_factor = self.cfg.DATASET.IMBALANCECIFAR.RATIO
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_list)
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.level_to_cls = self._split_levels(img_num_list)
            self.cls_to_level = []
            for level in range(len(self.level_to_cls)):
                self.cls_to_level += [level] * len(self.level_to_cls[level])
            for level in range(len(self.level_to_cls)):
                print('Level %d: %d (%d)' % (level+1, sum([img_num_list[cls] for cls in self.level_to_cls[level]]),
                                             len(self.level_to_cls[level])))
            self._augment_level_labels()
            self.cls_num = len(self.level_to_cls)
        else:
            self.transform = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                            ])
        print("{} Mode: Contain {} images".format(mode, len(self.data)))
        if self.dual_sample or (self.cfg.TRAIN.SAMPLER.TYPE == "weighted sampler" and self.train):
            self.class_weight, self.sum_weight = self.get_weight(self.get_annotations(), self.cls_num)
            self.class_dict = self._get_class_dict()

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.cls_num):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i

    def reset_epoch(self, cur_epoch):
        self.epoch = cur_epoch

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.get_annotations()):
            cat_id = anno["category_id"]
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def __getitem__(self, index):
        img, target = self.data[index], self.level_targets[index]

        return img, target

    def get_num_classes(self):
        return self.cls_num

    def reset_epoch(self, epoch):
        self.epoch = epoch

    @staticmethod
    def _split_levels(img_num_list, level_num=2):
        if level_num == 2:
            # 2-8 rule
            class_num = len(img_num_list)
            return [[i for i in range(int(class_num * 0.2))],
                    [i for i in range(int(class_num * 0.2), class_num)]]
        elif level_num == 3:
            min_diff = sum(img_num_list)
            best_flags = []
            for flag1 in range(0, len(img_num_list)-2):
                level1 = sum(img_num_list[:flag1+1])
                for flag2 in range(flag1+1, len(img_num_list)-1):
                    level2 = sum(img_num_list[flag1+1:flag2+1])
                    level3 = sum(img_num_list[flag2+1:])
                    flag_list = [level1, level2, level3]
                    curr_diff = max(flag_list) - min(flag_list)
                    if curr_diff < min_diff:
                        min_diff = curr_diff
                        best_flags = flag_list
            return [[i for i in range(0, best_flags[1]+1)],
                    [i for i in range(best_flags[1]+1, best_flags[2]+1)],
                    [i for i in range(best_flags[2]+1, len(img_num_list))]]
        else:
            raise ValueError('level num is 2 or 3')

    def _augment_level_labels(self):
        cls_num_before_level = [0]
        for level in range(1, len(self.level_to_cls)):
            cls_num_before_level.append(cls_num_before_level[level-1] + len(self.level_to_cls[level-1]))

        self.level_targets = []
        for label in self.targets:
            level = self.cls_to_level[label]
            label_in_level = label - cls_num_before_level[level]
            self.level_targets.append([label, level, label_in_level])

    def get_annotations(self):
        annos = []
        for level_target in self.level_targets:
            annos.append({'category_id': int(level_target[2])})
        return annos

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([the_class, ] * the_img_num)
        new_data = np.vstack(new_data)
        self.data = new_data
        self.targets = new_targets
        
    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

class IMBALANCECIFAR100S(IMBALANCECIFAR10S):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }
    cls_num = 100


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = IMBALANCECIFAR100S(root='./data', train=True,
                    download=True, transform=transform)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    import pdb; pdb.set_trace()
