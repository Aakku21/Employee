"""Look inside prepared training data: where it came from, how big it is, and real examples.

  python inspect_data.py data/web
  python inspect_data.py data/code --samples 5 --chars 800
  python inspect_data.py data/web data/code          # several folders at once
"""

import argparse
import json
import os

import numpy as np

from llm.data import load_meta
from llm.tokenizer import Tokenizer

CHUNK = 50_000_000  # tokens scanned at a time, keeps memory use small


def document_lengths(tokens, eot: int, max_docs: int = 200_000) -> np.ndarray:
    """Token length of each document (documents start with EOT). Stops after max_docs for speed."""
    starts, complete = [], True
    for i in range(0, len(tokens), CHUNK):
        starts.extend((np.flatnonzero(tokens[i : i + CHUNK] == eot) + i).tolist())
        if len(starts) > max_docs:
            complete = False
            break
    starts = np.array(starts, dtype=np.int64)
    if complete:
        ends = np.append(starts[1:], len(tokens))
    else:  # the last document found may continue past where we stopped scanning
        starts, ends = starts[:-1], starts[1:]
    return ends - starts - 1


def random_documents(tokens, eot: int, n: int, seed: int) -> list[np.ndarray]:
    """Pick random spots in the data and return the whole document around each."""
    rng = np.random.default_rng(seed)
    docs = []
    for _ in range(n * 3):  # a few extra tries in case two picks land in the same document
        pos = int(rng.integers(0, len(tokens)))
        start = pos
        while start > 0 and tokens[start] != eot:
            start -= 1
        end = pos + 1
        while end < len(tokens) and tokens[end] != eot:
            end += 1
        doc = np.asarray(tokens[start + 1 : end])
        if len(doc) and not any(len(d) == len(doc) and (d == doc).all() for d in docs):
            docs.append(doc)
        if len(docs) == n:
            break
    return docs


def inspect(data_dir: str, split: str, samples: int, chars: int, seed: int):
    meta = load_meta(data_dir)
    tok = Tokenizer(meta["tokenizer"])
    tokens = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=meta["dtype"], mode="r")
    source = meta.get("source", {})
    stats = meta.get("stats", {})

    print("=" * 80)
    print(f"{data_dir}  ({split} split)")
    print("=" * 80)
    where = source.get("dataset") or source.get("path") or "unknown"
    print(f"source:     {source.get('source')} -> {where}" + (f" [{source['subset']}]" if source.get("subset") else ""))
    print(f"tokenizer:  {meta['tokenizer']} (vocab {meta['vocab_size']:,})")
    print(f"tokens:     {meta['train_tokens']:,} train + {meta['val_tokens']:,} validation")
    if stats:
        seen = stats.get("seen", 0)
        kept = stats.get("kept", 0)
        print(f"documents:  {kept:,} kept out of {seen:,} looked at ({100 * kept / max(seen, 1):.0f}%)")
        for key in ("skipped_license", "skipped_quality", "skipped_duplicate", "skipped_unreadable"):
            if stats.get(key):
                print(f"            {stats[key]:,} dropped: {key.replace('skipped_', '')}")

    lengths = document_lengths(tokens, tok.eot)
    if len(lengths):
        p50, p90 = np.percentile(lengths, [50, 90])
        print(f"doc length: median {p50:,.0f} tokens, 90% under {p90:,.0f}, longest {lengths.max():,} "
              f"(from {len(lengths):,} documents)")
        size_mb = meta["train_tokens"] * np.dtype(meta["dtype"]).itemsize / 1e6
        print(f"on disk:    {size_mb:,.0f} MB for train.bin")

    for i, doc in enumerate(random_documents(tokens, tok.eot, samples, seed), 1):
        text = tok.decode(doc.tolist())
        cut = text[:chars] + (" [...]" if len(text) > chars else "")
        print(f"\n--- random document {i} of {samples} ({len(doc):,} tokens) ---")
        print(cut)
    print()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data_dirs", nargs="+")
    p.add_argument("--split", choices=["train", "val"], default="train")
    p.add_argument("--samples", type=int, default=3, help="how many random documents to show")
    p.add_argument("--chars", type=int, default=500, help="show at most this many characters per document")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    for data_dir in args.data_dirs:
        inspect(data_dir, args.split, args.samples, args.chars, args.seed)


if __name__ == "__main__":
    main()
