import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
from core.evaluate import accuracy

import numpy as np
import torch
import time


def train_model(
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
    stage=0,
    **kwargs):

    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_level_cnt = [0] * label_map.shape[1]
    all_level_acc = [0] * label_map.shape[1]

    batch_cnt = 0
    batch_loss = 0
    batch_level_loss = [0] * label_map.shape[1]
    batch_level_acc = [0] * label_map.shape[1]
    batch_level_size = [0] * label_map.shape[1]
    for i, (image, label, meta) in enumerate(trainLoader):

        if stage == 1 or stage == 0:
            level_loss_list = [0] * label_map.shape[1]
            for level in range(label_map.shape[1]):

                # if stage == 1 and level == 1:
                #     continue
                #
                # if stage == 2 and level == 0:
                #     continue

                level_label = label_map[label, level]
                level_mask = level_label >= 0
                level_image = image[level_mask]
                level_label = level_label[level_mask]
                level_image = level_image.cuda()
                level_label = level_label.cuda()

                level_score = model(level_image, level)
                level_loss = criterion(level_score, level_label)

                level_res = torch.argmax(level_score, 1)
                level_acc = accuracy(level_res.cpu().numpy(), level_label.cpu().numpy())[0]
                level_loss_list[level] = level_loss

                batch_level_acc[level] += level_acc
                batch_level_size[level] += level_label.shape[0]
                all_level_cnt[level] += level_label.shape[0]
                all_level_acc[level] += level_acc * level_label.shape[0]

            for level, level_loss in enumerate(level_loss_list):
                batch_level_loss[level] += level_loss

            loss = sum(level_loss_list)

        elif stage == 2:
            image = image.cuda()
            label = label.cuda()
            scores = model(image)
            loss = criterion(scores, label)
            result = torch.argmax(scores, 1)
            acc = accuracy(result.cpu().numpy(), label.cpu().numpy())[0]

        batch_cnt += 1
        batch_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Loss:{:>5.3f}[{:>5.3f}/{:>5.3f}]  L1/L2:{}/{}  ".format(
                epoch, i, number_batch,
                batch_loss / batch_cnt,
                batch_level_loss[0] / batch_cnt,
                batch_level_loss[1] / batch_cnt,
                int(batch_level_size[0] * 1.0 / batch_cnt),
                int(batch_level_size[1] * 1.0 / batch_cnt))
            for level, acc in enumerate(batch_level_acc):
                pbar_str += 'Level %d Acc: %.4f  ' % (level + 1, acc / batch_cnt)
            logger.info(pbar_str)
            batch_cnt = 0
            batch_loss = 0
            batch_level_loss = [0] * label_map.shape[1]
            batch_level_acc = [0] * label_map.shape[1]
            batch_level_size = [0] * label_map.shape[1]


def valid_model(
        dataLoader,
        epoch_number,
        model,
        logger,
        device,
        label_map,
        level_label_maps,
        stage=0):

    model.eval()
    num_levels = label_map.shape[1]
    num_classes = dataLoader.dataset.get_num_classes()

    l1_cls_num = label_map[:, 0].max().item() + 1
    l2_cls_num = label_map[:, 1].max().item() + 1
    virtual_cls_num = l1_cls_num + l2_cls_num - num_classes
    l1_raw_cls_num = l1_cls_num - virtual_cls_num
    l2_raw_cls_num = l2_cls_num

    fusion_matrix = FusionMatrix(num_classes)
    func = torch.nn.Softmax(dim=1)
    acc = AverageMeter()
    l1_acc = AverageMeter()
    l2_acc = AverageMeter()

    with torch.no_grad():
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            batch_size = label.shape[0]

            if stage == 1 or stage == 0:
                level_scores = []
                level_probs = []
                for level in range(num_levels):
                    level_score = model(image, level)
                    level_scores.append(level_score)

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

                all_probs = torch.ones((batch_size, num_classes)).cuda()
                for level in range(num_levels):
                    level_prob = level_probs[level]
                    related_lcids = label_map[:, level]
                    related_lcids = related_lcids[related_lcids >= 0]
                    unrelated_class_num1 = (label_map[:, level] < 0).sum().item()
                    unrelated_class_num2 = label_map.shape[0] - related_lcids.shape[0]
                    assert unrelated_class_num1 == unrelated_class_num2
                    all_probs[:, unrelated_class_num1:] *= level_prob[:, related_lcids]

            elif stage == 2:
                all_probs = model(image)

            else:
                print('ERROR STAGE: %d' % stage)
                exit(-1)

            # if stage == 1:
            #     all_probs = level_probs[0]
            #     label = label_map[label, 0]

            l1_mask = label < l1_raw_cls_num
            l1_scores = all_probs[l1_mask]
            l1_labels = label[l1_mask]
            l1_result = torch.argmax(l1_scores, 1)
            l1_now_acc, l1_cnt = accuracy(l1_result.cpu().numpy(), l1_labels.cpu().numpy())
            l1_acc.update(l1_now_acc, l1_cnt)

            l2_mask = label >= l1_raw_cls_num
            l2_scores = all_probs[l2_mask]
            l2_labels = label[l2_mask]
            l2_result = torch.argmax(l2_scores, 1)
            l2_now_acc, l2_cnt = accuracy(l2_result.cpu().numpy(), l2_labels.cpu().numpy())
            l2_acc.update(l2_now_acc, l2_cnt)

            now_result = torch.argmax(all_probs, 1)
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Acc:{:>5.4f}  P1_Acc:{:>5.4f}  P2_Acc:{:>5.4f}-------".format(
            epoch_number, acc.avg, l1_acc.avg, l2_acc.avg)
        logger.info(pbar_str)
    return acc.avg
