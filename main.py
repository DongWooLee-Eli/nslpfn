import os
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt
import json
import time

import torch
import torch.nn as nn

from pfn.prior import create_get_batch_func, sample_from_prior
from pfn.bar_distribution import get_bucket_limits, FullSupportBarDistribution
from pfn.utils import get_cosine_schedule_with_warmup, PriorDL
from pfn.transformer import TransformerModel
from utils import Logger
from inference import test
import matplotlib
import warnings
matplotlib.use('Agg') 
warnings.filterwarnings("ignore")

def main(args):
    os.environ["WANDB_SILENT"] = "true"
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)

    # prior data
    batch_func = create_get_batch_func(
        prior=sample_from_prior,
        var_lnloc=args.var_lnloc,
        var_lnscale=args.var_lnscale,
    )
    prior_dl = PriorDL(batch_func, args.iteration, args.batch_size)

    if args.debug:
        args.exp_name = "debug"
        args.print_every = 1
        args.eval_every = 1

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    borders = get_bucket_limits(num_outputs=args.d_output, full_range=(0, 1.2), ys=None)
    criterion = FullSupportBarDistribution(borders)
    criterion_checkpoint = torch.load(os.path.join(f"{args.save_dir}/{args.exp_name}/criterion_init.pt"))
    criterion.load_state_dict(criterion_checkpoint)
    criterion = criterion.to(device)

    # model and opt
    model = TransformerModel(
        d_output=args.d_output,
        d_model=args.d_model,
        dim_feedforward=2*args.d_model,
        nlayers=args.nlayers,
        dropout=args.dropout,
        activation="gelu",
        y_stats=None,
    ).to(device)
    model.train()

        
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.wd,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    sch = get_cosine_schedule_with_warmup(opt, args.iteration//4, args.iteration)

    # logger
    logger = Logger(
        args.exp_name,
        save_dir=f"{args.save_dir}/{args.exp_name}",
        save_only_last=True,
        print_every=args.print_every,
        save_every=args.save_every,
        total_step=args.iteration,
        print_to_stdout=True,
        wandb_project_name=f"nsl-pfn",
        wandb_config=args
    )
    logger.register_model_to_save(model, "model")
    logger.register_model_to_save(criterion, "criterion")
    logger.start()
    
    # outer loop
    for step, (x, y, context_size) in enumerate(prior_dl):
        # data
        x, y = x.to(device), y.to(device)
        xc, yc = x[:, :context_size, :], y[:, :context_size]

        # train
        y_pred = model(xc, yc, x)
        losses = criterion(y_pred, y.contiguous())
        loss = losses.mean()
        opt.zero_grad()
        loss.backward()
        if args.grad_norm > 0.:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        opt.step()
        sch.step()

        with torch.no_grad():
            if args.logit_strategy == 'mean':
                y_pred = criterion.mean(y_pred)
            elif args.logit_strategy == 'median':
                y_pred = criterion.median(y_pred)
            elif args.logit_strategy == 'mode':
                y_pred = criterion.mode(y_pred)

            # rmse
            rmse = torch.sqrt(((y_pred - y) ** 2).sum() / y.numel()).item()
            
            # rmsle
            rmsle = torch.sqrt(((torch.log(y_pred) - torch.log(y)) ** 2).sum() / y.numel()).item()

        logger.meter("train", "loss", loss)
        logger.meter("train", "rmse", rmse)
        logger.meter("train", "rmsle", rmsle)

        # test
        if (step+1) % args.eval_every == 0 or (step+1) == args.iteration:
            model.eval()
            test(logger, device, model, criterion, cutoff=-1, data_dir=args.data_dir, save_dir=f"{args.save_dir}/{args.exp_name}/{step+1}")
            model.train()

        logger.step()

    logger.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=1)

    # dir
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--save_dir', type=str, default="./pretrained_surrogate_results")
    parser.add_argument('--exp_name', type=str, default="debug")

    # hparams for data
    parser.add_argument('--batch_size', type=int, default=16)

    # hparams for model
    parser.add_argument('--d_output', type=int, default=1000)
    parser.add_argument('--nlayers', type=int, default=12)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.2)

    # hparms for training
    parser.add_argument('--iteration', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--wd', type=float, default=0.0)
    parser.add_argument('--grad_norm', type=float, default=1.0)

    # hparams for test
    parser.add_argument('--logit_strategy', type=str, choices=['mean', 'median', 'mode'], default='median')
    
    # hparams for logger
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=2000)

    # hparams for prior
    parser.add_argument('--var_lnloc', type=float, default=-4.0, help='mean of the normal distribution for the variance')
    parser.add_argument('--var_lnscale', type=float, default=1.0, help='std of the normal distribution for the variance')

    # gpus
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action="store_true")
    args = parser.parse_args()
    
    main(args)
