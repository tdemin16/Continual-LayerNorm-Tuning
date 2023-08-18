# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
from typing import Iterable
import wandb
import datetime

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from timm.utils import accuracy

import couple.utils as utils

def train_key_one_epoch(model: nn.Module, original_model: nn.Module, data_loader: Iterable,
                        optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, 
                        max_norm: float=0, task_id=-1, run=None, args=None,):
    assert args.split_train
    assert original_model is not None

    original_model.eval()
    model.train()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.no_grad():
            _, original_feats, _ = original_model(input)
            original_feats = original_feats[:, 0]
        
        model_without_ddp = model
        if args.distributed:
            model_without_ddp = model.module
        
        _, metrics = model_without_ddp.wsm(feats=original_feats, task=task_id)
        loss = -metrics['sim']

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.meters['Sim'].update(metrics['sim'].item(), n=input.shape[0])
    
    metric_logger.synchronize_between_processes()
    print(f"Similarity Epoch {epoch}: ", metric_logger)
    wandb_dict = {}
    if run is not None:
        wandb_dict = {'train/sim': metric_logger.meters['Sim'].global_avg}
    return wandb_dict


def train_one_epoch_unbiased(model: nn.Module, original_model: nn.Module, 
                             criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                             device: torch.device, epoch: int, task_id=-1, class_mask=None, 
                             run=None, args=None,):
    
    if original_model is not None:
        original_model.eval()
    model.train()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if original_model is not None:
            with torch.no_grad():
                _, original_feats, _ = original_model(input)
                original_feats = original_feats[:, 0]

                logits, feats, metrics = model(input, original_feats, task_eval=task_id, fake=True)
        else:
            with torch.no_grad():
                logits, feats, metrics = model(input, task_eval=task_id, fake=True)

        logits = model.module.forward_head(feats)
        
        # here is the trick to mask out classes of non-current tasks
        logits = utils.mask_logits(logits, class_mask, args.nb_classes, task_id)
        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"Train Unbiased Epoch {epoch}:", metric_logger)

    wandb_dict = {}
    if run is not None:
        wandb_dict = {
            'train/acc1_unbiased': metric_logger.meters['Acc@1'].global_avg,
            'train/acc5_unbiased': metric_logger.meters['Acc@5'].global_avg,
            'train/loss_unbiased': metric_logger.meters['Loss'].global_avg,
            'train/lr_unbiased': metric_logger.meters['Lr'].global_avg,
        }
    return wandb_dict


def train_one_epoch(model: nn.Module, original_model: nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, run=None, args=None,):

    if original_model is not None:
        original_model.eval()
    model.train(set_training_mode)

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)
    
    # wheter to use the fake classifier or not
    fake = False
    if args.unbiased:
        fake=True

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if original_model is not None and not args.split_train:
            with torch.no_grad():
                _, original_feats, _ = original_model(input)
                original_feats = original_feats[:, 0]

            logits, feats, metrics = model(input, original_feats, task_id, fake=fake)
        else:
            logits, feats, metrics = model(input, task=task_id, fake=fake)

        # here is the trick to mask out classes of non-current tasks
        logits = utils.mask_logits(logits, class_mask, args.nb_classes, task_id)

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)

        if 'sim' in metrics.keys():
            loss = loss - metrics['sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        if 'sim' in metrics.keys():
            metric_logger.meters['Sim'].update(metrics['sim'].item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    wandb_dict = {}
    if run is not None:
        wandb_dict = {
            'train/acc1': metric_logger.meters['Acc@1'].global_avg,
            'train/acc5': metric_logger.meters['Acc@5'].global_avg,
            'train/loss': metric_logger.meters['Loss'].global_avg,
            'train/lr': metric_logger.meters['Lr'].global_avg,
        }
        if 'sim' in metrics.keys():
            wandb_dict['train/sim'] = metric_logger.meters['Sim'].global_avg
    return wandb_dict


@torch.no_grad()
def evaluate(model: nn.Module, original_model: nn.Module, data_loader: Iterable, 
             device, task_id=-1, last_task=-1, class_mask=None, args=None,):
    criterion = nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)
    
    # store values for Confusion Matrix
    task_pred = []
    task_target = []

    # switch to evaluation mode
    if original_model is not None:
        original_model.eval()
    model.eval()

    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if original_model is not None:
            _, original_feats, _ = original_model(input)
            original_feats = original_feats[:, 0]
            logits, _, metrics = model(input, original_feats, task_eval=task_id)
        
        else:
            logits, _, metrics = model(input, task_eval=task_id)

        loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        metric_logger.meters['Loss'].update(loss.item())
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        if 'acc_sim' in metrics.keys():
            metric_logger.meters['Acc_Task'].update(metrics['acc_sim'].item(), n=input.shape[0])
        if 'logits_acc' in metrics.keys():
            metric_logger.meters['Logits_Acc'].update(metrics['logits_acc'].item(), n=input.shape[0])
        
        # store in two lists task predictions and ground truths iff the prediction has been computed
        if 'pred' in metrics.keys():
            task_pred.append(metrics['pred'])
            task_target.append(torch.full(metrics['pred'].size(), task_id))

    cm = None
    # store confusion matrix information 
    if len(task_pred) > 0 and len(task_target) > 0:
        assert args.world_size == 1
        task_pred = torch.cat(task_pred)
        task_target = torch.cat(task_target)
        cm = [task_pred, task_target]

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} Acc_Task {task.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], 
                task=metric_logger.meters['Acc_Task'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cm


@torch.no_grad()
def evaluate_till_now(model: nn.Module, original_model: nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, run=None, args=None,):
    stat_matrix = np.zeros((4, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss, Acc_task

    task_pred = []
    task_target = []
    for i in range(task_id+1):
        test_stats, cm = evaluate(model=model, original_model=original_model, 
                              data_loader=data_loader[i]['val'], device=device, task_id=i, 
                              last_task=task_id, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        if 'Acc_Task' in test_stats.keys():
            stat_matrix[3, i] = test_stats['Acc_Task']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
        # retrieve task predictions and ground truths
        if cm is not None:
            p, t = cm
            task_pred.append(p)
            task_target.append(t)
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    cm = None
    if len(task_pred) > 1 and len(task_target) > 1:
        task_pred = torch.cat(task_pred)
        task_target = torch.cat(task_target)
        cm = (task_pred, task_target)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tAcc_Task: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[3], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    wandb_dict = {}
    if run is not None:
        for task in range(task_id+1):
            wandb_dict.update({
                f'test/task{task}_acc': stat_matrix[0][task],
                f'test/task{task}_acc_task': stat_matrix[3][task],
            })
        wandb_dict.update({
            'test/avg_acc': avg_stat[0],
        })
        if task_id > 0:
            wandb_dict.update({
                'test/forgetting': forgetting,
            })

    return wandb_dict, cm

def train_and_evaluate(model: nn.Module, model_without_ddp: nn.Module, 
                       criterion, data_loader: Iterable, 
                       optimizer: torch.optim.Optimizer, lr_scheduler, 
                       device: torch.device, original_model: nn.Module=None, 
                       class_mask=None, args=None):
    
    if args.wandb and args.rank == 0:
        wandb_name = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}-{args.name}'
        run = wandb.init(
            project='C-LN',
            entity='entity',
            name=wandb_name,
            notes=args.notes,
            config=args,
        )
    else:
        run = None

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        # transfer parameters to new task
        if task_id > 0:
            if args.distributed:
                model.module.next_task(task_id)
            else:
                model.next_task(task_id)

        # Create new optimizer for each task to clear optimizer status
        if task_id > 0:
            optimizer = utils.get_optimizer(args, model_without_ddp)
        
        train_dataloader = data_loader[task_id]['train']
        
        wandb_dict = {}
        # Similarity training when split_train is on
        if args.split_train and original_model is not None:
            for epoch in range(args.epochs):
                stats = train_key_one_epoch(model=model, original_model=original_model, 
                                    data_loader=train_dataloader, optimizer=optimizer, device=device,
                                    epoch=epoch, max_norm=args.clip_grad, task_id=task_id, run=run, 
                                    args=args)
            wandb_dict.update(stats)

        # LN training (head training too in case unbiased is false)
        for epoch in range(args.epochs):
            stats = train_one_epoch(model=model, original_model=original_model, 
                                    criterion=criterion, data_loader=train_dataloader,
                                    optimizer=optimizer, device=device, epoch=epoch,
                                    max_norm=args.clip_grad, set_training_mode=True, 
                                    task_id=task_id, class_mask=class_mask, run=run,
                                    args=args,)
        
            if lr_scheduler:
                lr_scheduler.step(epoch)
        wandb_dict.update(stats)

        # classifier correction
        if args.unbiased:
            params = []
            for n, p in model_without_ddp.named_parameters():
                p.requires_grad = True if n.startswith('head') else False
                if p.requires_grad:
                    params.append(p)
                    
            params = [{'params': params, 'lr': args.lr_unbiased}]
            optimizer = utils.create_optimizer(args, params)
            
            if args.full_match:
                model_without_ddp.set_custom_match(True)
            for epoch in range(args.epochs_unbiased):
                stats = train_one_epoch_unbiased(model=model, original_model=original_model,
                                                criterion=criterion, data_loader=train_dataloader,
                                                optimizer=optimizer, device=device, epoch=epoch,
                                                task_id=task_id, class_mask=class_mask, run=run,
                                                args=args,)
            if args.full_match:
                model_without_ddp.set_custom_match(False)
            wandb_dict.update(stats) 
            for n, p in model_without_ddp.named_parameters():
                p.requires_grad = True if utils.is_trainable(args, n) else False
        
        if args.full_match:
            model_without_ddp.set_custom_match(True)
        stats, cm = evaluate_till_now(model=model, original_model=original_model,\
                                       data_loader=data_loader, device=device, task_id=task_id, 
                                       class_mask=class_mask, acc_matrix=acc_matrix, run=run, 
                                       args=args)
        if args.full_match:
            model_without_ddp.set_custom_match(False)    
    
        wandb_dict.update(stats)
        if run is not None:
            run.log(wandb_dict)
        
        if args.save_cm and task_id > 0 and cm is not None:
            cm_path = os.path.join(args.output_dir, f'cm/{args.name}/task_{task_id}.png')
            utils.save_confusion_matrix(cm, cm_path)
