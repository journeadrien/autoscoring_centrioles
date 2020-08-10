# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:55:29 2020

@author: journe
"""

import math
import sys
import time
import torch

import torchvision.models.detection.mask_rcnn

# from coco_utils import get_coco_api_from_dataset
# from coco_eval import CocoEvaluator
import utils
from pprint import PrettyPrinter
pp = PrettyPrinter()



def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            # sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.no_grad()
def evaluate(model, data_loader, device):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        # torch.cuda.synchronize()
        model_time = time.time()
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(image, targets)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        model_time = time.time() - model_time
        metric_logger.update(model_time=model_time)
    print("Averaged stats:", metric_logger)
    
@torch.no_grad()
def evaluate2(model, data_loader, device):
    model.eval()
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()
    for image, targets in data_loader:
        # torch.cuda.synchronize()
        image = list(img.to(device) for img in image)
        boxes = [t['boxes'].to(device) for t in targets]
        labels = [t['labels'].to(device) for t in targets]
        difficulties = [torch.zeros_like(t['labels']).to(device) for t in targets]
        # Forward prop.
        predictions = model(image)  # (N, 8732, 4), (N, 8732, n_classes)
          
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
        det_boxes_batch = [t['boxes'] for t in predictions]
        det_labels_batch = [t['labels'] for t in predictions]
        det_scores_batch =  [t['scores'] for t in predictions]
        det_boxes.extend(det_boxes_batch)
        det_labels.extend(det_labels_batch)
        det_scores.extend(det_scores_batch)
        true_boxes.extend(boxes)
        true_labels.extend(labels)
        true_difficulties.extend(difficulties)
    APs, mAP = utils.calculate_mAP(device,det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties,n_classes =2)
    pp.pprint(APs)
    print('\nMean Average Precision (mAP): %.3f' % mAP)
    # # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
