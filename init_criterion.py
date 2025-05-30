import os
import random
import argparse
import numpy as np

import torch

from pfn.prior import create_get_batch_func, sample_from_prior
from pfn.bar_distribution import get_bucket_limits, FullSupportBarDistribution

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main(args):
    # prior data
    batch_func = create_get_batch_func(
        prior=sample_from_prior,
        var_lnloc=args.var_lnloc,
        var_lnscale=args.var_lnscale,
    )

    # seed
    if args.seed is None:
        args.seed = random.randint(0, 9999)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Start to sampling priors..")
    _, ys, _ = batch_func(batch_size=100_000, seq_len=100)
    borders = get_bucket_limits(num_outputs=args.d_output, full_range=None, ys=ys)
    criterion = FullSupportBarDistribution(borders)
    os.makedirs(f"{args.save_dir}/{args.exp_name}", exist_ok=True)
    criterion_path = os.path.join(f"{args.save_dir}/{args.exp_name}/criterion_init.pt")
    torch.save(criterion.state_dict(), criterion_path)
    print(f"Criterion is saved in {criterion_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # seed
    parser.add_argument('--seed', type=int, default=42)

    # dir
    parser.add_argument('--data_dir', type=str, default="./data")
    parser.add_argument('--save_dir', type=str, default="./pretrained_surrogate_results")
    parser.add_argument('--exp_name', type=str, default="debug")

    # hparams for model
    parser.add_argument('--d_output', type=int, default=1000)

    # hparams for prior
    parser.add_argument('--var_lnloc', type=float, default=-4.0, help='mean of the normal distribution for the variance')
    parser.add_argument('--var_lnscale', type=float, default=1.0, help='std of the normal distribution for the variance')

    args = parser.parse_args()
    main(args)
