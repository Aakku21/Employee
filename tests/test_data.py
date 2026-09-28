import json
import os

import numpy as np
import pytest

from llm.config import load_config
from llm.data import BatchLoader, parse_sources
from llm.quality import Deduplicator, goes_to_val, keep_code
from prepare_data import main as prepare


def write_fake_dataset(path, start, n=5000, tokenizer="cl100k_base"):
    os.makedirs(path, exist_ok=True)
    for split in ("train", "val"):
        np.arange(start, start + n, dtype=np.uint32).tofile(os.path.join(path, f"{split}.bin"))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump({"tokenizer": tokenizer, "vocab_size": 100277, "eot_token": 100257, "dtype": "uint32",
                   "train_tokens": n, "val_tokens": n}, f)


def test_targets_are_inputs_shifted_by_one(tmp_path):
    write_fake_dataset(tmp_path / "a", start=0)
    loader = BatchLoader(str(tmp_path / "a"), "train", 4, 16, "cpu", seed=0)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (4, 16)
    assert (y == x + 1).all()


def test_mixing_follows_weights(tmp_path):
    write_fake_dataset(tmp_path / "web", start=0)
    write_fake_dataset(tmp_path / "code", start=50_000)
    spec = f"{tmp_path / 'web'}:0.8,{tmp_path / 'code'}:0.2"
    loader = BatchLoader(spec, "train", 1000, 8, "cpu", seed=0)
    x, _ = loader.next_batch()
    share_code = (x[:, 0] >= 50_000).float().mean().item()
    assert 0.15 < share_code < 0.25


def test_mixing_different_tokenizers_fails(tmp_path):
    write_fake_dataset(tmp_path / "a", start=0)
    write_fake_dataset(tmp_path / "b", start=0, tokenizer="gpt2")
    with pytest.raises(ValueError, match="same tokenizer"):
        BatchLoader(f"{tmp_path / 'a'},{tmp_path / 'b'}", "train", 2, 8, "cpu", seed=0)


def test_parse_sources():
    assert parse_sources("data/web:0.7, data/code:0.3") == [("data/web", 0.7), ("data/code", 0.3)]
    assert parse_sources("data/web") == [("data/web", 1.0)]
    with pytest.raises(ValueError):
        parse_sources("data/web:0")


def test_config_overrides(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"model": {"n_layer": 3}, "train": {"lr": 0.1}}))
    model, train = load_config(str(path), ["--train.max_steps=5e3", "--train.compile=true", "--model.d_model=96"])
    assert (model.n_layer, model.d_model, train.lr, train.max_steps, train.compile) == (3, 96, 0.1, 5000, True)
    with pytest.raises(ValueError, match="unknown setting"):
        load_config(None, ["--train.learning_rate=1"])


def test_code_filter():
    good = "def add(a, b):\n    return a + b\n" * 10
    assert keep_code(good)
    assert not keep_code("x = 1")  # too short
    assert not keep_code("a" * 5000)  # one enormous line: minified or data
    assert not keep_code("# auto-generated file, do not edit\n" + good)
    assert not keep_code("{}[]();,.:=+-*/" * 50)  # mostly symbols


def test_dedup_and_split_are_deterministic():
    d = Deduplicator()
    assert d.is_new("hello world") and not d.is_new("  hello world\n")
    texts = [f"document {i}" for i in range(2000)]
    val = [t for t in texts if goes_to_val(t, 50)]
    assert val == [t for t in texts if goes_to_val(t, 50)]
    assert 50 < len(val) < 150  # about 5%


def test_prepare_local_folder(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "node_modules").mkdir()
    code = "def add(a, b):\n    return a + b\n\n" * 20
    (src / "pkg" / "a.py").write_text(code)
    (src / "pkg" / "copy.py").write_text(code)  # exact duplicate, should be dropped
    (src / "pkg" / "b.py").write_text("def mul(a, b):\n    return a * b\n\n" * 20)
    (src / "README.md").write_text("This project adds and multiplies numbers. " * 20)
    (src / "node_modules" / "skip.js").write_text("var x = 1;\n" * 50)
    (src / "image.png").write_bytes(b"\x89PNG not text")

    out = tmp_path / "out"
    meta = prepare(["--source", "local", "--path", str(src), "--out", str(out), "--val-permille", "300"])
    assert meta["stats"]["kept"] == 3
    assert meta["stats"]["skipped_duplicate"] == 1
    assert meta["train_tokens"] + meta["val_tokens"] > 0

    from llm.tokenizer import Tokenizer

    tok = Tokenizer(meta["tokenizer"])
    ids = np.concatenate([np.fromfile(out / f"{s}.bin", dtype=meta["dtype"]) for s in ("train", "val")])
    text = tok.decode([int(i) for i in ids if i != tok.eot])
    assert "def add(a, b):" in text and "def mul(a, b):" in text and "skip" not in text

    with pytest.raises(SystemExit):  # refuses to overwrite by accident
        prepare(["--source", "local", "--path", str(src), "--out", str(out)])
