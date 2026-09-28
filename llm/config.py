"""Model and training settings.

Settings live in a JSON file with two sections, "model" and "train". Any setting
can be overridden on the command line, e.g. `--train.lr=3e-4 --model.n_layer=6`.
"""

import json
from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    vocab_size: int = 0  # filled in from the dataset's tokenizer at train time
    max_seq_len: int = 1024  # longest text (in tokens) the model can read at once
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int = 12  # fewer than n_head = grouped-query attention (smaller, faster)
    d_model: int = 768
    rope_theta: float = 10000.0
    qk_norm: bool = True  # normalize queries/keys; makes training more stable
    dropout: float = 0.0  # keep 0 for pretraining on large data
    tie_embeddings: bool = True  # share input and output token tables (saves params)


@dataclass
class TrainConfig:
    # One or more tokenized datasets made by prepare_data.py. Mix them with weights:
    # "data/web:0.7,data/code:0.3". All must use the same tokenizer.
    data: str = "data/web"
    out_dir: str = "runs/default"
    seed: int = 1337

    # Batch size. total_batch_tokens is what the optimizer sees per step; we reach it
    # with gradient accumulation, so it stays the same on 1 GPU or 8.
    batch_size: int = 16  # sequences per device per micro-step
    total_batch_tokens: int = 524288
    max_steps: int = 5000

    # Optimizer (AdamW) and learning-rate schedule (warmup, then cosine decay).
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 700
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Evaluation, logging and checkpoints.
    eval_interval: int = 250
    eval_steps: int = 20
    log_interval: int = 10
    save_interval: int = 1000
    sample_prompt: str = ""  # if set, print a sample from the model at each eval
    resume: bool = True  # continue from out_dir/ckpt.pt if it exists

    # Hardware.
    device: str = "auto"  # auto | cuda | mps | cpu
    dtype: str = "auto"  # auto | bfloat16 | float16 | float32
    compile: bool = False  # torch.compile: ~1.3-2x faster on modern GPUs, slow first step


def _cast(value: str, current):
    if isinstance(current, bool):
        if value.lower() in ("1", "true", "yes"):
            return True
        if value.lower() in ("0", "false", "no"):
            return False
        raise ValueError(f"expected true/false, got {value!r}")
    if isinstance(current, int):
        return int(float(value))  # allows 5e3
    if isinstance(current, float):
        return float(value)
    return value


def load_config(path: str | None, overrides: list[str]) -> tuple[ModelConfig, TrainConfig]:
    """Read a JSON config file (optional), then apply `--section.key=value` overrides."""
    model, train = ModelConfig(), TrainConfig()
    sections = {"model": model, "train": train}

    raw = {}
    if path:
        with open(path) as f:
            raw = json.load(f)
    for section, values in raw.items():
        if section not in sections:
            raise ValueError(f"unknown config section {section!r} (use 'model' or 'train')")
        _apply(sections[section], values)

    for arg in overrides:
        if not arg.startswith("--") or "=" not in arg or "." not in arg.split("=")[0]:
            raise ValueError(f"bad override {arg!r}; use --train.lr=3e-4 or --model.n_layer=6")
        key, value = arg[2:].split("=", 1)
        section, name = key.split(".", 1)
        if section not in sections:
            raise ValueError(f"unknown config section {section!r} in {arg!r}")
        obj = sections[section]
        _check_field(obj, name)
        setattr(obj, name, _cast(value, getattr(obj, name)))

    return model, train


def _apply(obj, values: dict):
    for name, value in values.items():
        _check_field(obj, name)
        setattr(obj, name, value)


def _check_field(obj, name: str):
    known = {f.name for f in fields(obj)}
    if name not in known:
        raise ValueError(f"unknown setting {name!r} for {type(obj).__name__}; known: {sorted(known)}")


def to_dict(model: ModelConfig, train: TrainConfig) -> dict:
    return {"model": asdict(model), "train": asdict(train)}
