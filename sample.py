"""Generate text or code from a trained model.

  python sample.py runs/gpt2_124m --prompt "def quicksort(arr):"
  python sample.py runs/gpt2_124m --prompt "The history of Rome" --num-samples 3 --temperature 0.9
"""

import argparse
import os

import torch

from llm.config import ModelConfig
from llm.model import GPT
from llm.tokenizer import Tokenizer
from train import pick_device


def load_model(run_dir: str, device: str):
    path = run_dir if run_dir.endswith(".pt") else os.path.join(run_dir, "ckpt.pt")
    ckpt = torch.load(path, map_location="cpu")
    model = GPT(ModelConfig(**ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), Tokenizer(ckpt["tokenizer"]), ckpt["step"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="training output folder (or a path to ckpt.pt)")
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8, help="0 = always pick the most likely token")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model, tok, step = load_model(args.run_dir, device)
    print(f"Loaded model from step {step} ({model.num_params() / 1e6:.1f}M params)\n")

    prompt_ids = [tok.eot] + tok.encode(args.prompt)  # EOT = "a new document starts here"
    idx = torch.tensor([prompt_ids] * args.num_samples, device=device)
    out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k, args.top_p, stop_token=tok.eot)
    samples = []
    for row in out.tolist():
        new = row[len(prompt_ids) :]
        if tok.eot in new:
            new = new[: new.index(tok.eot)]
        samples.append(args.prompt + tok.decode(new))
    for i, text in enumerate(samples):
        print(f"--- sample {i + 1} ---\n{text}\n")
    return samples


if __name__ == "__main__":
    main()
