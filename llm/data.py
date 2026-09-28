"""Reads the token files made by prepare_data.py and serves random training batches."""

import json
import os

import numpy as np
import torch


def load_meta(data_dir: str) -> dict:
    path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run prepare_data.py first.")
    with open(path) as f:
        return json.load(f)


def parse_sources(spec: str) -> list[tuple[str, float]]:
    """'data/web:0.7,data/code:0.3' -> [('data/web', 0.7), ('data/code', 0.3)]. Weight defaults to 1."""
    sources = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        path, sep, weight = part.rpartition(":")
        if sep and path and _is_number(weight):
            sources.append((path, float(weight)))
        else:
            sources.append((part, 1.0))
    if not sources:
        raise ValueError("no data sources given")
    if any(w <= 0 for _, w in sources):
        raise ValueError(f"data weights must be positive: {spec!r}")
    return sources


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def check_same_tokenizer(sources) -> dict:
    metas = [load_meta(path) for path, _ in sources]
    names = {m["tokenizer"] for m in metas}
    if len(names) > 1:
        raise ValueError(f"all data sources must use the same tokenizer, found {sorted(names)}")
    return metas[0]


class BatchLoader:
    """Samples random windows of seq_len+1 tokens from one or more weighted sources."""

    def __init__(self, spec: str, split: str, batch_size: int, seq_len: int, device: str, seed):
        self.sources = parse_sources(spec)
        meta = check_same_tokenizer(self.sources)
        self.dtype = np.dtype(meta["dtype"])
        self.batch_size, self.seq_len, self.device = batch_size, seq_len, device
        self.files = []
        for path, _ in self.sources:
            file = os.path.join(path, f"{split}.bin")
            n = os.path.getsize(file) // self.dtype.itemsize
            if n <= seq_len + 1:
                raise ValueError(f"{file} has {n} tokens, fewer than seq_len+1={seq_len + 1}")
            self.files.append((file, n))
        weights = np.array([w for _, w in self.sources], dtype=np.float64)
        self.probs = weights / weights.sum()
        self.rng = np.random.default_rng(seed)

    def next_batch(self):
        which = self.rng.choice(len(self.files), size=self.batch_size, p=self.probs)
        rows = []
        for i in which:
            file, n = self.files[i]
            # Re-open the memmap per batch; holding one open leaks memory over long runs.
            tokens = np.memmap(file, dtype=self.dtype, mode="r")
            start = self.rng.integers(0, n - self.seq_len - 1)
            rows.append(tokens[start : start + self.seq_len + 1].astype(np.int64))
        batch = torch.from_numpy(np.stack(rows))
        x, y = batch[:, :-1], batch[:, 1:]
        if self.device.startswith("cuda"):
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)
