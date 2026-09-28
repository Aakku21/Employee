import json

from test_data import write_fake_dataset

import sample
import train


def test_train_resume_and_sample(tmp_path, capsys):
    # A repeating pattern is easy to learn, so loss must drop fast.
    data = tmp_path / "data"
    write_fake_dataset(data, start=0, n=4000)
    out = tmp_path / "run"
    args = [
        f"--train.data={data}", f"--train.out_dir={out}", "--train.device=cpu",
        "--model.n_layer=2", "--model.n_head=2", "--model.n_kv_head=1", "--model.d_model=32", "--model.max_seq_len=16",
        "--train.batch_size=4", "--train.total_batch_tokens=128", "--train.warmup_steps=2", "--train.lr=1e-2",
        "--train.eval_interval=10", "--train.eval_steps=2", "--train.log_interval=5", "--train.save_interval=10",
    ]

    first = train.main(args + ["--train.max_steps=10"])
    assert first["step"] == 10
    assert (out / "ckpt.pt").exists()

    # Running again with a higher step count must continue from step 10, not restart.
    capsys.readouterr()
    second = train.main(args + ["--train.max_steps=20"])
    printed = capsys.readouterr().out
    assert "resuming from step 10" in printed
    assert second["best_val_loss"] < first["val_losses"]["val"]

    records = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert any(r.get("step") == 20 and "loss_val" in r for r in records)

    texts = sample.main([str(out), "--prompt", "hi", "--max-new-tokens", "5", "--num-samples", "2", "--device", "cpu"])
    assert len(texts) == 2 and all(t.startswith("hi") for t in texts)
