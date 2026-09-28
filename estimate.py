"""Estimate what a training run will cost before you rent any GPUs.

  python estimate.py configs/gpt2_124m.json
  python estimate.py configs/gpt2_124m.json --tokens 10e9 --efficiency 0.3

Formula: training FLOPs ~= 6 x parameters x tokens (+ an attention term).
Real speed depends on your code, GPU, and network; treat this as a rough guide.
"""

import argparse
import math

import torch

from llm.config import load_config
from llm.model import GPT, flops_per_token

# Peak dense bf16 throughput (FLOP/s) and an assumed rental price in USD per GPU-hour.
# Prices change often and vary by provider: check yours and pass --price.
GPUS = {
    "H100": (989e12, 2.50),
    "A100": (312e12, 1.50),
    "RTX 4090": (165e12, 0.40),
}
LLAMA_31_405B_FLOPS = 3.8e25  # reported by Meta in the Llama 3 paper


def human(n: float) -> str:
    return f"{n / 1e9:,.2f}B" if n >= 1e9 else f"{n / 1e6:,.1f}M"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("--tokens", type=float, help="training tokens (default: max_steps x total_batch_tokens)")
    p.add_argument("--vocab", type=int, default=100277, help="tokenizer vocab size (default: cl100k_base)")
    p.add_argument("--efficiency", type=float, default=0.4, help="fraction of peak GPU speed you reach (0.3-0.5)")
    p.add_argument("--price", type=float, help="override USD per GPU-hour for every GPU type")
    args = p.parse_args(argv)

    mcfg, tcfg = load_config(args.config, [])
    mcfg.vocab_size = 128 * math.ceil(args.vocab / 128)
    with torch.device("meta"):  # count parameters without allocating memory
        n_params = GPT(mcfg).num_params()
    tokens = args.tokens or tcfg.max_steps * tcfg.total_batch_tokens
    flops = flops_per_token(mcfg, n_params) * tokens

    print(f"parameters:        {n_params / 1e6:,.1f}M")
    print(f"training tokens:   {human(tokens)} ({tokens / n_params:.1f} per parameter)")
    print(f"compute-optimal:   ~{human(20 * n_params)} tokens (20 per parameter, Chinchilla rule of thumb)")
    print(f"training compute:  {flops:.2e} FLOPs")
    print(f"vs Llama 3.1 405B: 1/{LLAMA_31_405B_FLOPS / flops:,.0f} of its training compute\n")
    print(f"{'GPU':<10} {'GPU-hours':>10} {'hours on 8 GPUs':>16} {'rough cost':>12}")
    for name, (peak, price) in GPUS.items():
        hours = flops / (peak * args.efficiency) / 3600
        cost = hours * (args.price or price)
        print(f"{name:<10} {hours:>10,.2f} {hours / 8:>16,.2f} {'$' + format(cost, ',.2f'):>12}")
    print("\nCost excludes failed runs, experiments, data prep and storage. Budget 2-3x for those.")


if __name__ == "__main__":
    main()
