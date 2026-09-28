import math

import pytest
import torch

from llm.config import ModelConfig
from llm.model import GPT


def tiny(**overrides):
    cfg = dict(vocab_size=256, max_seq_len=32, n_layer=2, n_head=4, n_kv_head=2, d_model=64)
    cfg.update(overrides)
    return ModelConfig(**cfg)


def test_initial_loss_is_close_to_uniform_guess():
    torch.manual_seed(0)
    model = GPT(tiny())
    x, y = torch.randint(0, 256, (4, 32)), torch.randint(0, 256, (4, 32))
    logits, loss = model(x, y)
    assert logits.shape == (4, 32, 256)
    # A fresh model should be unsure about everything: loss ~= ln(vocab).
    assert abs(loss.item() - math.log(256)) < 0.3


def test_model_cannot_see_future_tokens():
    torch.manual_seed(0)
    model = GPT(tiny()).eval()
    x = torch.randint(0, 256, (1, 32))
    changed = x.clone()
    changed[0, 20:] = (changed[0, 20:] + 1) % 256
    a, _ = model(x, x)
    b, _ = model(changed, changed)
    assert torch.allclose(a[0, :20], b[0, :20], atol=1e-5)
    assert not torch.allclose(a[0, 20:], b[0, 20:], atol=1e-5)


@pytest.mark.parametrize("n_kv_head, qk_norm", [(4, True), (2, True), (1, False)])
def test_attention_variants_run(n_kv_head, qk_norm):
    model = GPT(tiny(n_kv_head=n_kv_head, qk_norm=qk_norm))
    _, loss = model(torch.randint(0, 256, (2, 16)), torch.randint(0, 256, (2, 16)))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())


def test_learns_to_memorize_one_batch():
    torch.manual_seed(0)
    model = GPT(tiny())
    opt = model.make_optimizer(3e-3, 0.0, (0.9, 0.95), "cpu")
    x = torch.randint(0, 256, (4, 33))
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5


def test_tied_embeddings_share_memory():
    model = GPT(tiny())
    assert model.lm_head.weight is model.tok_emb.weight
    untied = GPT(tiny(tie_embeddings=False))
    assert untied.num_params() > model.num_params()


def test_weight_decay_skips_norms():
    model = GPT(tiny())
    decay, no_decay = model.make_optimizer(1e-3, 0.1, (0.9, 0.95), "cpu").param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])


def test_generate_respects_length_and_context_limit():
    model = GPT(tiny()).eval()
    out = model.generate(torch.zeros(2, 5, dtype=torch.long), max_new_tokens=40, top_k=10, top_p=0.9)
    assert out.shape == (2, 45)  # longer than max_seq_len: context gets cropped
    greedy_a = model.generate(torch.zeros(1, 5, dtype=torch.long), 10, temperature=0)
    greedy_b = model.generate(torch.zeros(1, 5, dtype=torch.long), 10, temperature=0)
    assert torch.equal(greedy_a, greedy_b)


def test_too_long_input_is_rejected():
    with pytest.raises(ValueError, match="max_seq_len"):
        GPT(tiny())(torch.zeros(1, 33, dtype=torch.long))
