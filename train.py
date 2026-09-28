"""Pretrain the model on data made by prepare_data.py.

  python train.py configs/tiny_cpu.json                                      # quick test on a CPU
  python train.py configs/gpt2_124m.json                                     # one GPU
  torchrun --standalone --nproc_per_node=8 train.py configs/gpt2_124m.json   # 8 GPUs, one machine
  python train.py configs/gpt2_124m.json --train.lr=3e-4 --model.n_layer=8   # change any setting

Run the same command again after a crash or stop: it resumes from the last checkpoint.
"""

import contextlib
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from llm.config import load_config, to_dict
from llm.data import BatchLoader, check_same_tokenizer, parse_sources
from llm.model import GPT, flops_per_token
from llm.tokenizer import Tokenizer

H100_BF16_FLOPS = 989e12  # peak dense bf16, used only for the cost estimate we print


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str, device: str) -> str:
    if name != "auto":
        return name
    if device.startswith("cuda"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def lr_at(step: int, tcfg) -> float:
    """Linear warmup, then cosine decay down to lr * min_lr_ratio."""
    if step < tcfg.warmup_steps:
        return tcfg.lr * (step + 1) / tcfg.warmup_steps
    min_lr = tcfg.lr * tcfg.min_lr_ratio
    progress = min(1.0, (step - tcfg.warmup_steps) / max(1, tcfg.max_steps - tcfg.warmup_steps))
    return min_lr + 0.5 * (tcfg.lr - min_lr) * (1 + math.cos(math.pi * progress))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    config_path = argv[0] if argv and not argv[0].startswith("--") else None
    mcfg, tcfg = load_config(config_path, argv[1:] if config_path else argv)

    # --- devices -------------------------------------------------------------
    ddp = "RANK" in os.environ  # torchrun sets this
    if ddp:
        use_cuda = torch.cuda.is_available()  # without GPUs, torchrun still works on CPU (for testing)
        dist.init_process_group(backend="nccl" if use_cuda else "gloo")
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}" if use_cuda else "cpu"
        if use_cuda:
            torch.cuda.set_device(device)
    else:
        rank, world, local_rank = 0, 1, 0
        device = pick_device(tcfg.device)
    master = rank == 0
    device_type = "cuda" if device.startswith("cuda") else device
    dtype = pick_dtype(tcfg.dtype, device)
    autocast = (
        contextlib.nullcontext()
        if dtype == "float32"
        else torch.autocast(device_type=device_type, dtype=getattr(torch, dtype))
    )
    torch.manual_seed(tcfg.seed)  # same init on every GPU
    torch.set_float32_matmul_precision("high")

    # --- data and batch size ---------------------------------------------------
    sources = parse_sources(tcfg.data)
    meta = check_same_tokenizer(sources)
    tokenizer = Tokenizer(meta["tokenizer"])
    mcfg.vocab_size = 128 * math.ceil(meta["vocab_size"] / 128)  # pad for faster GPU kernels

    tokens_per_micro_step = tcfg.batch_size * mcfg.max_seq_len * world
    if tcfg.total_batch_tokens % tokens_per_micro_step:
        raise ValueError(
            f"total_batch_tokens ({tcfg.total_batch_tokens}) must be a multiple of "
            f"batch_size x max_seq_len x GPUs ({tokens_per_micro_step})"
        )
    grad_accum = tcfg.total_batch_tokens // tokens_per_micro_step

    # --- model, optimizer, resume ---------------------------------------------
    model = GPT(mcfg).to(device)
    optimizer = model.make_optimizer(tcfg.lr, tcfg.weight_decay, (tcfg.beta1, tcfg.beta2), device_type)
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == "float16"))
    ckpt_path = os.path.join(tcfg.out_dir, "ckpt.pt")
    start_step, best_val = 0, float("inf")
    if tcfg.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if ckpt["config"]["model"] != asdict(mcfg):
            raise ValueError(f"{ckpt_path} was trained with different model settings. Use a new out_dir.")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step, best_val = ckpt["step"], ckpt["best_val_loss"]
        del ckpt

    raw_model = model
    if tcfg.compile:
        model = torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[local_rank] if device.startswith("cuda") else None)

    if master:
        os.makedirs(tcfg.out_dir, exist_ok=True)
        with open(os.path.join(tcfg.out_dir, "config.json"), "w") as f:
            json.dump(to_dict(mcfg, tcfg), f, indent=2)
        log_file = open(os.path.join(tcfg.out_dir, "log.jsonl"), "a")

    def log(record: dict, text: str):
        if master:
            print(text, flush=True)
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

    n_params = raw_model.num_params()
    total_tokens = tcfg.max_steps * tcfg.total_batch_tokens
    train_tokens = sum(check_same_tokenizer([s])["train_tokens"] for s in sources)
    flops = flops_per_token(mcfg, n_params) * total_tokens
    if master:
        print(f"model:   {n_params / 1e6:.1f}M params ({raw_model.num_params(False) / 1e6:.1f}M without embeddings), "
              f"vocab {mcfg.vocab_size}, context {mcfg.max_seq_len}")
        print(f"batch:   {tcfg.total_batch_tokens:,} tokens/step = {tcfg.batch_size} seqs x {mcfg.max_seq_len} tokens "
              f"x {world} device(s) x {grad_accum} accumulation steps")
        print(f"plan:    {tcfg.max_steps:,} steps = {total_tokens / 1e9:.2f}B tokens "
              f"({total_tokens / n_params:.1f} tokens per parameter; ~20 is compute-optimal)")
        print(f"data:    {train_tokens / 1e9:.3f}B train tokens -> {total_tokens / train_tokens:.2f} passes over the data")
        print(f"compute: {flops:.2e} FLOPs = about {flops / (0.4 * H100_BF16_FLOPS) / 3600:.1f} H100-hours at 40% efficiency")
        print(f"device:  {device} x{world}, {dtype}, compile={tcfg.compile}, resuming from step {start_step}")
        if total_tokens > 4 * train_tokens:
            print("warning: you will repeat the data more than 4 times. Expect overfitting; prepare more data.")

    # --- evaluation and checkpoints -------------------------------------------
    @torch.no_grad()
    def evaluate() -> dict:
        """Validation loss per data source, then the weighted mean."""
        model.eval()
        losses, weighted = {}, 0.0
        for i, (path, weight) in enumerate(sources):
            # A fixed seed gives the same val batches every time, so losses are comparable.
            loader = BatchLoader(path, "val", tcfg.batch_size, mcfg.max_seq_len, device, seed=[tcfg.seed, rank, i])
            total = torch.zeros((), device=device)
            for _ in range(tcfg.eval_steps):
                x, y = loader.next_batch()
                with autocast:
                    _, loss = model(x, y)
                total += loss.float()
            total /= tcfg.eval_steps
            if ddp:
                dist.all_reduce(total)
                total /= world
            name = os.path.basename(os.path.normpath(path))
            losses[name if name not in losses else path] = total.item()
            weighted += weight * total.item()
        losses["val"] = weighted / sum(w for _, w in sources)
        model.train()
        return losses

    def save(step: int):
        if not master:
            return
        state = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val,
            "config": to_dict(mcfg, tcfg),
            "tokenizer": tokenizer.name,
        }
        torch.save(state, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)  # never leave a half-written checkpoint

    def sample(prompt: str) -> str:
        raw_model.eval()
        ids = torch.tensor([[tokenizer.eot] + tokenizer.encode(prompt)], device=device)
        with autocast:
            out = raw_model.generate(ids, 60, temperature=0.8, top_k=50, stop_token=tokenizer.eot)
        raw_model.train()
        return tokenizer.decode(out[0, 1:].tolist())

    # --- training loop ---------------------------------------------------------
    train_loader = BatchLoader(tcfg.data, "train", tcfg.batch_size, mcfg.max_seq_len, device,
                               seed=[tcfg.seed, rank, start_step])
    x, y = train_loader.next_batch()
    model.train()
    val_losses = {}
    timer = time.time()

    for step in range(start_step, tcfg.max_steps + 1):  # step = optimizer updates done so far
        last = step == tcfg.max_steps
        if step % tcfg.eval_interval == 0 or last:
            val_losses = evaluate()
            best_val = min(best_val, val_losses["val"])
            parts = " | ".join(f"{k} {v:.4f}" for k, v in val_losses.items())
            log({"step": step, **{f"loss_{k}": v for k, v in val_losses.items()}}, f"step {step:>6} | eval | {parts}")
            if tcfg.sample_prompt and master:
                print(f"sample: {sample(tcfg.sample_prompt)!r}", flush=True)
            timer = time.time()  # don't count eval time in throughput
        if (step > start_step and step % tcfg.save_interval == 0) or last:
            save(step)
        if last:
            break

        lr = lr_at(step, tcfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        for micro in range(grad_accum):
            if ddp:
                model.require_backward_grad_sync = micro == grad_accum - 1  # sync grads once per step
            with autocast:
                _, loss = model(x, y)
            loss = loss / grad_accum
            loss_sum += loss.detach()
            x, y = train_loader.next_batch()  # load the next batch while the GPU works
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if (step + 1) % tcfg.log_interval == 0:
            if ddp:
                dist.all_reduce(loss_sum)
                loss_sum /= world
            train_loss = loss_sum.item()
            if not math.isfinite(train_loss):
                raise RuntimeError(f"loss became {train_loss} at step {step + 1}. Lower --train.lr or use bfloat16.")
            elapsed = time.time() - timer
            tok_per_sec = tcfg.log_interval * tcfg.total_batch_tokens / elapsed
            timer = time.time()
            log(
                {"step": step + 1, "loss": train_loss, "lr": lr, "grad_norm": grad_norm.item(), "tok_per_sec": tok_per_sec},
                f"step {step + 1:>6} | loss {train_loss:.4f} | lr {lr:.2e} | grad norm {grad_norm.item():.2f} | "
                f"{tok_per_sec / 1e3:,.1f}K tok/s",
            )

    if master:
        log_file.close()
    if ddp:
        dist.destroy_process_group()
    return {"step": tcfg.max_steps, "val_losses": val_losses, "best_val_loss": best_val}


if __name__ == "__main__":
    main()
