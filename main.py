# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler

from couple.blocks import BlockLN
from couple.configs import CONFIGS
from couple.datasets import build_continual_dataloader
from couple.engine import *
import couple.models # used to register custom vit architecture
from couple import utils

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')

def main(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    # train_dataloader, val_dataloader = tmp_dl(args)
    data_loader, class_mask = build_continual_dataloader(args)

    # whether original model is used for weight selection
    if args.method == 'single_stage':
        args.original_model = False
    else:
        args.original_model = True
    
    original_model = None
    if args.original_model:
        original_model = create_model(
            args.model,
            pretrained=True,
            num_classes=args.nb_classes,
        )
        original_model.to(device)

        # freeze original model parameters
        for p in original_model.parameters():
            p.requires_grad = False

    model = create_model(
        args.model,
        pretrained=True,
        num_classes=args.nb_classes,
        block_fn=BlockLN,
    )
    model.init(args)
    model.to(device)
        
    # freeze everything except head and layernorm
    learnable_params = []
    for n, p in model.named_parameters():
        if utils.is_trainable(args, n):
            p.requires_grad = True
            learnable_params.append((n, p))
        else:
            p.requires_grad = False
        
    n_parameters = sum(p.numel() for _, p in learnable_params)
    print(f'Name: {args.name}')
    print(f'Description: {args.notes}')
    print(f"Learnable Parameters {[n for n, _ in learnable_params]}")
    print('Number of params:', n_parameters)
    print(args)

    if args.eval:
        raise NotImplementedError("Use the evaluation script.")

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = utils.get_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + args.name \
           if args.name else None
    
    train_and_evaluate(model=model, 
                       model_without_ddp=model_without_ddp, 
                       original_model=original_model, 
                       criterion=criterion, 
                       data_loader=data_loader, 
                       optimizer=optimizer, 
                       lr_scheduler=lr_scheduler, 
                       device=device,
                       class_mask=class_mask, 
                       args=args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    
    subparser = parser.add_subparsers(dest='subparser_name')
    config_parser = subparser.add_parser(config)
    
    get_args_parser = CONFIGS[config]
    get_args_parser(config_parser)

    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
    
    sys.exit(0)