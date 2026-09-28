"""Download internet text or code, clean it, tokenize it, and save it for training.

  # Web pages (FineWeb-Edu: Common Crawl filtered for educational quality)
  python prepare_data.py --source web --max-tokens 3e9 --out data/web

  # Python code from GitHub (permissive licenses only, by default)
  python prepare_data.py --source code --max-tokens 1e9 --out data/code

  # Your own folder of code or text
  python prepare_data.py --source local --path ~/projects --out data/mine

  # Any Hugging Face dataset
  python prepare_data.py --source hf --dataset NAME --subset SUBSET --text-field text --out data/other

Output: <out>/train.bin, <out>/val.bin (raw token ids) and <out>/meta.json.
Press Ctrl+C any time; everything written so far is kept and usable.
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

from llm.quality import PERMISSIVE_LICENSES, Deduplicator, goes_to_val, keep_code, keep_text
from llm.tokenizer import SUPPORTED, Tokenizer

PRESETS = {
    "web": {"dataset": "HuggingFaceFW/fineweb-edu", "subset": "sample-10BT", "text_field": "text", "kind": "text"},
    "code": {"dataset": "codeparrot/codeparrot-clean", "subset": None, "text_field": "content", "kind": "code"},
}

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh", ".sql", ".html", ".css", ".yaml", ".yml", ".toml",
}
TEXT_EXTENSIONS = {".md", ".txt", ".rst"}
SKIP_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".tox", ".mypy_cache"}


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["web", "code", "local", "hf"])
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--max-tokens", type=float, default=1e9, help="stop after about this many tokens (default 1e9)")
    p.add_argument("--tokenizer", default="cl100k_base", choices=SUPPORTED)
    p.add_argument("--val-permille", type=int, default=5, help="documents per 1000 held out for validation")
    p.add_argument("--dedup", choices=["auto", "on", "off"], default="auto",
                   help="drop exact duplicates (auto: on for code/local, off for FineWeb which is already deduped)")
    p.add_argument("--all-licenses", action="store_true", help="code source: also keep GPL and other copyleft files")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    p.add_argument("--overwrite", action="store_true")
    # --source hf
    p.add_argument("--dataset")
    p.add_argument("--subset")
    p.add_argument("--split", default="train")
    p.add_argument("--text-field", default="text")
    p.add_argument("--kind", choices=["text", "code"], default="text", help="which quality filter to apply")
    # --source local
    p.add_argument("--path", help="folder to read for --source local")
    args = p.parse_args(argv)

    if args.source in PRESETS:
        preset = PRESETS[args.source]
        args.dataset = args.dataset or preset["dataset"]
        args.subset = args.subset or preset["subset"]
        args.text_field = preset["text_field"]
        args.kind = preset["kind"]
    if args.source == "hf" and not args.dataset:
        p.error("--source hf needs --dataset")
    if args.source == "local" and not args.path:
        p.error("--source local needs --path")
    if args.dedup == "auto":
        args.dedup = "off" if args.source == "web" else "on"
    return args


def hf_documents(args, stats):
    from datasets import load_dataset

    ds = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True)
    check_license = args.source == "code" and not args.all_licenses
    for row in ds:
        if check_license and row.get("license") not in PERMISSIVE_LICENSES:
            stats["seen"] += 1
            stats["skipped_license"] += 1
            continue
        yield row[args.text_field], args.kind


def local_documents(args, stats):
    for root, dirs, files in os.walk(os.path.expanduser(args.path)):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext in CODE_EXTENSIONS:
                kind = "code"
            elif ext in TEXT_EXTENSIONS:
                kind = "text"
            else:
                continue
            try:
                with open(os.path.join(root, name), encoding="utf-8") as f:
                    yield f.read(), kind
            except (UnicodeDecodeError, OSError):
                stats["skipped_unreadable"] += 1


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tok = Tokenizer(args.tokenizer)
    os.makedirs(args.out, exist_ok=True)
    paths = {split: os.path.join(args.out, f"{split}.bin") for split in ("train", "val")}
    if any(os.path.exists(p) for p in paths.values()) and not args.overwrite:
        sys.exit(f"{args.out} already has data. Use --overwrite or pick another --out.")

    stats = Counter()
    dedup = Deduplicator() if args.dedup == "on" else None
    documents = local_documents(args, stats) if args.source == "local" else hf_documents(args, stats)
    files = {split: open(path, "wb") for split, path in paths.items()}
    max_tokens = int(args.max_tokens)
    written = 0
    start = last_report = time.time()

    def flush(batch):
        nonlocal written
        out = {"train": [], "val": []}
        for text, ids in zip(batch, tok.encode_batch(batch, num_threads=args.threads)):
            split = "val" if goes_to_val(text, args.val_permille) else "train"
            out[split].append(np.array([tok.eot] + ids, dtype=tok.dtype))  # EOT marks a new document
        for split, arrays in out.items():
            if arrays:
                chunk = np.concatenate(arrays)
                chunk.tofile(files[split])
                written += len(chunk)

    batch = []
    try:
        for text, kind in documents:
            stats["seen"] += 1
            if not (keep_code(text) if kind == "code" else keep_text(text)):
                stats["skipped_quality"] += 1
                continue
            if dedup and not dedup.is_new(text):
                stats["skipped_duplicate"] += 1
                continue
            stats["kept"] += 1
            batch.append(text)
            if len(batch) >= 1024:
                flush(batch)
                batch = []
                if time.time() - last_report > 10:
                    last_report = time.time()
                    rate = written / (last_report - start)
                    print(f"{written / 1e6:,.1f}M tokens | {stats['kept']:,} docs kept of {stats['seen']:,} | "
                          f"{rate / 1e3:,.0f}K tokens/s", flush=True)
                if written >= max_tokens:
                    break
        if batch:
            flush(batch)
    except KeyboardInterrupt:
        print("\nStopped early. Keeping what was written so far.")
    finally:
        for f in files.values():
            f.close()

    counts = {}
    itemsize = np.dtype(tok.dtype).itemsize
    for split, path in paths.items():
        size = os.path.getsize(path) // itemsize
        os.truncate(path, size * itemsize)  # drop a half-written token if interrupted mid-write
        counts[split] = size
    meta = {
        "tokenizer": tok.name,
        "vocab_size": tok.vocab_size,
        "eot_token": tok.eot,
        "dtype": np.dtype(tok.dtype).name,
        "train_tokens": counts["train"],
        "val_tokens": counts["val"],
        "source": {k: getattr(args, k) for k in ("source", "dataset", "subset", "path", "kind", "dedup")},
        "stats": dict(stats),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done: {counts['train']:,} train + {counts['val']:,} val tokens in {args.out}")
    print(f"Documents: {dict(stats)}")
    if counts["val"] == 0:
        print("Warning: no validation tokens. Add more data or raise --val-permille.")
    return meta


if __name__ == "__main__":
    main()
    # Stopping a streamed Hugging Face dataset early leaves a download thread running,
    # which can crash Python during shutdown. Everything is saved and closed by now,
    # so exit right away instead.
    sys.stdout.flush()
    os._exit(0)
