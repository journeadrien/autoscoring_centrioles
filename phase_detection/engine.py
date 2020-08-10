# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:28:07 2020

@author: journe
"""
import torch
import time
import datetime

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, percent = False):
        self.reset()
        self.m = 1
        if percent:
            self.m = 100

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum * self.m / self.count
        
        
def validate(val_loader, model, criterion, args):
    confusion_matrix = torch.zeros(args.nb_classes, args.nb_classes)
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    start = time.time()
    with torch.no_grad():
        end = time.time()
        for i, (input, target, sample_weight) in enumerate(val_loader):
            data_time.update(time.time() - end)

            input = input.float().to(args.device,non_blocking=True )
            target = target.to(args.device,non_blocking=True )
            sample_weight = sample_weight.to(args.device,non_blocking=True )

            ## 1. forward propagation
            output = model(input)

            loss = criterion(output, target)
            loss =(loss * sample_weight / sample_weight.sum()).sum()
            loss = loss.mean()


            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            top1.update(acc1[0], input.size(0))

            losses.update(loss.item(), input.size(0))

            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
            batch_time.update(time.time() - end)
            end = time.time()
            #import  pdb; pdb.set_trace()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time,
                       data_time=data_time,top1=top1, loss=losses))
    print('End of Testing, Took {total_time}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}'
                  .format(
                   total_time=str(datetime.timedelta(seconds=int(time.time()-start))),
                   top1=top1, loss=losses))

    print(confusion_matrix)
    perf_metric = losses.avg
    return perf_metric


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()
    start = time.time()
    end = time.time()
    for i, (input, target, sample_weight) in enumerate(train_loader):
        data_time.update(time.time() - end)

        #print(torch.isnan(input).sum(axis=1))
        input = input.float().to(args.device,non_blocking=True )
        target = target.to(args.device,non_blocking=True )
        sample_weight = sample_weight.to(args.device,non_blocking=True )
        ## 1. forward propagation
        output = model(input)

        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        top1.update(acc1[0], input.size(0))
        

        ## 2. loss calculation
        loss = criterion(output, target)
        loss =(loss * sample_weight / sample_weight.sum()).sum()
        loss = loss.mean()
        ## 3. compute gradient and do SGD step
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # every args.print_freq batch
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Lr: {lr:.5f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'
                  .format(
                   epoch, i, len(train_loader),
                   lr = optimizer.param_groups[0]['lr'],
                   batch_time=batch_time,
                   data_time=data_time,top1=top1, loss=losses))

    print('End of Training Epoch: {0}\t'
                  'Took {total_time}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}'
                  .format(
                   epoch,
                   total_time=str(datetime.timedelta(seconds=int(time.time()-start))),
                   top1=top1, loss=losses))