import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix

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
    **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    combiner.reset_epoch(epoch)

    if cfg.LOSS.LOSS_TYPE in ['LDAMLoss', 'CSCE']:
        criterion.reset_epoch(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()
    for i, (image, label, meta) in enumerate(trainLoader):
        cnt = label.shape[0]
        loss, now_acc = combiner.forward(model, criterion, image, label, meta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)
    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    logger.info(pbar_str)
    return acc.avg, all_loss.avg


def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, **kwargs
):
    model.eval()
    num_classes = dataLoader.dataset.get_num_classes()
    fusion_matrix = FusionMatrix(num_classes)


    with torch.no_grad():
        all_loss = AverageMeter()
        acc = AverageMeter()
        l1_acc = AverageMeter()
        l2_acc = AverageMeter()
        func = torch.nn.Softmax(dim=1)
        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True)
            loss = criterion(output, label)
            score_result = func(output)

            l1_mask = label < 20
            l1_scores = score_result[l1_mask]
            l1_labels = label[l1_mask]
            l1_result = torch.argmax(l1_scores, 1)
            l1_now_acc, l1_cnt = accuracy(l1_result.cpu().numpy(), l1_labels.cpu().numpy())
            l1_acc.update(l1_now_acc, l1_cnt)

            l2_mask = label >= 20
            l2_scores = score_result[l2_mask]
            l2_labels = label[l2_mask]
            l2_result = torch.argmax(l2_scores, 1)
            l2_now_acc, l2_cnt = accuracy(l2_result.cpu().numpy(), l2_labels.cpu().numpy())
            l2_acc.update(l2_now_acc, l2_cnt)

            now_result = torch.argmax(score_result, 1)
            all_loss.update(loss.data.item(), label.shape[0])
            fusion_matrix.update(now_result.cpu().numpy(), label.cpu().numpy())
            now_acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())
            acc.update(now_acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}  Valid_Acc:{:>5.2f}%  L1_Acc:{:>5.2f}%  L2_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc.avg * 100, l1_acc.avg * 100, l2_acc.avg * 100
        )
        logger.info(pbar_str)
    return acc.avg, all_loss.avg
