# Train your own LLM from scratch on internet text and code

A small, readable codebase that pretrains a GPT-style language model on web pages
and GitHub code. The full pipeline is here:

**download data → clean it → tokenize → train (1 or many GPUs) → generate text**

The model uses the same main building blocks as current open models (Llama, Qwen, OLMo):
rotary position embeddings, RMSNorm, SwiGLU, grouped-query attention, QK-norm,
flash attention, bf16 mixed precision, and multi-GPU training.

## Read this first: what you will and will not get

- **This will not beat GPT, Claude, Gemini, Llama or Qwen.** The `small` model here
  uses about **1/12,000,000** of the compute Meta used for Llama 3.1 405B. The same code
  at a bigger size does not close that gap. Money, data and team size do.
- **After pretraining, the model continues text. It does not chat or follow instructions.**
  A chat model needs a second stage (instruction tuning), which is not built yet.
- **What this is good for:** learning exactly how LLMs are built, research
  experiments, and small models for narrow uses.
- **If you want the strongest coding assistant you can own,** start from an existing
  open model (for example Qwen2.5-Coder) and fine-tune it on your data. On any budget
  you can afford, that will beat anything you pretrain from scratch.

| Config | Size | Training tokens | Time on 8×H100* | Compute cost* | What to expect |
|---|---|---|---|---|---|
| `tiny_cpu` | 13.6M | 0.6M | ~8 min on a laptop CPU | $0 | Proves the pipeline works. Output is mostly gibberish. |
| `colab_t4` | 69M | 131M | ~1–1.5 hours on 1 free Colab T4 | $0 | Learns word and code shapes. Rarely correct. Good for learning. |
| `small` | 153M | 3.1B | ~20–40 min | ~$6–20 | Roughly GPT-2-small (2019) quality: fluent-looking text, simple code patterns, often wrong. |
| `medium` | 369M | 7.3B | ~2–4 hours | ~$30–100 | Noticeably better, but still weak at real coding tasks. |

\* Estimates from `estimate.py` at 40% GPU efficiency and assumed rental prices. Real
cost is higher: budget 2–3× for failed runs, experiments and data prep.

## Setup

```bash
pip install -r requirements.txt
pytest                      # 19 tests, ~10 seconds on CPU
```

## Quickest path: train for free on Google Colab

No GPU of your own? Google Colab gives you a free T4 GPU for a few hours per session.

1. Open https://colab.research.google.com, create a new notebook, and pick
   **Runtime → Change runtime type → T4 GPU**.
2. Paste each block below into its own cell and run them in order.

```python
# 1. Get the code (private repo? use https://<your-github-token>@github.com/...)
!git clone -b claude/beautiful-mendel-w30nmm https://github.com/Aakku21/Employee.git
%cd Employee
!pip install -q tiktoken datasets          # PyTorch is already installed on Colab

# 2. Save checkpoints to Google Drive, so a disconnect doesn't lose your training
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 3. Download and prepare data (~20-40 minutes)
!python prepare_data.py --source web  --max-tokens 100e6 --out data/web
!python prepare_data.py --source code --max-tokens 50e6  --out data/code
```

```python
# 4. Train (~1-1.5 hours). If Colab disconnects: re-run cells 1-3, then this cell. It resumes.
!python train.py configs/colab_t4.json --train.out_dir=/content/drive/MyDrive/llm/colab
```

```python
# 5. Use it
!python sample.py /content/drive/MyDrive/llm/colab --prompt "def is_prime(n):" --num-samples 3
```

## Step 1: get data from the internet

Two ready-made sources:

- **Web:** [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
  Common Crawl web pages, filtered by Hugging Face for educational quality and already
  deduplicated.
- **Code:** [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean).
  Python files from GitHub. By default only permissively licensed files (MIT, Apache,
  BSD, ...) are kept, which drops about 40% of files. Add `--all-licenses` to keep GPL
  code too, but check the legal risk first.

```bash
# Enough for the small config (70% web, 30% code)
python prepare_data.py --source web  --max-tokens 2.5e9 --out data/web
python prepare_data.py --source code --max-tokens 1e9   --out data/code

# Your own code or notes
python prepare_data.py --source local --path ~/projects --out data/mine

# Any Hugging Face dataset
python prepare_data.py --source hf --dataset NAME --text-field text --out data/other
```

Things to know:

- Data is streamed. Nothing is downloaded in full, and you can stop with Ctrl+C
  at any time and keep what was written.
- It is not fast: about 90K tokens/second on a 4-core machine, so 2.5B tokens takes
  hours. Run it on a machine with many cores and a fast network, and set `HF_TOKEN`
  to avoid Hugging Face rate limits.
- Tokenizer: `cl100k_base` by default. It stores code in **~1.8× fewer tokens** than the
  GPT-2 tokenizer (measured: 4.18 vs 2.28 bytes per token on this code data), so
  training on code is cheaper and the model sees more code per step.
- Quality filters drop minified files, data blobs, auto-generated files and exact
  duplicates (see `llm/quality.py`).
- Train/validation split is decided by each document's content hash, so the same
  document never lands in both.

## Step 2: estimate the cost before you pay

```bash
python estimate.py configs/small.json
```

This prints parameter count, training tokens, the compute-optimal token count
(~20 tokens per parameter), and GPU-hours and cost on H100 / A100 / RTX 4090.

## Step 3: train

```bash
python train.py configs/tiny_cpu.json                                      # test on a CPU first
python train.py configs/small.json                                         # one GPU
torchrun --standalone --nproc_per_node=8 train.py configs/small.json       # 8 GPUs
python train.py configs/small.json --train.lr=3e-4 --model.n_layer=8       # change any setting
```

- **Resuming:** re-running the same command continues from `runs/<name>/ckpt.pt`.
  This is useful on cheap "spot" GPUs that can be taken away.
- **Batch size:** `total_batch_tokens` stays the same on 1 or 8 GPUs; the script
  adjusts gradient accumulation for you.
- **Out of GPU memory?** Lower `--train.batch_size`. The effective batch does not change.
  On a 24 GB card (RTX 4090) use `--train.batch_size=4` for `small`.
- **Mixing data:** `"data": "data/web:0.7,data/code:0.3"` samples 70% web, 30% code.
  Change the mix without re-tokenizing.
- **What to watch:** loss is the model's error score; lower is better.
  - It starts near 11.5 (a pure guess) and should drop fast in the first few hundred
    steps, then keep falling slowly.
  - Loss goes up, or becomes `nan`: the learning rate is too high. Lower `--train.lr`.
  - Training loss keeps falling but validation loss rises: the model is memorizing.
    Get more data.
  - Validation loss is shown per source (`web`, `code`), so you can see which one is weak.
- Logs go to `runs/<name>/log.jsonl`.

## Step 4: use the model

**From the command line:**

```bash
python sample.py runs/small --prompt "def quicksort(arr):" --num-samples 3
python sample.py runs/small --prompt "The history of Rome" --temperature 0.9 --max-new-tokens 300
```

| Setting | What it does |
|---|---|
| `--temperature` | Randomness. `0` = always the most likely word (repeats itself), `0.7–0.9` = balanced, `>1` = wild. |
| `--top-k`, `--top-p` | Only pick from the most likely next tokens. Cuts nonsense. |
| `--max-new-tokens` | How much text to write. |
| `--num-samples` | Several different answers to the same prompt. |

**From your own Python code:**

```python
import torch
from sample import load_model

device = "cuda" if torch.cuda.is_available() else "cpu"
model, tok, step = load_model("runs/small", device)

prompt = 'def is_prime(n):\n    """Return True if n is a prime number."""\n'
ids = torch.tensor([[tok.eot] + tok.encode(prompt)], device=device)
out = model.generate(ids, max_new_tokens=100, temperature=0.8, top_k=50, stop_token=tok.eot)

new = out[0, ids.shape[1]:].tolist()
if tok.eot in new:
    new = new[: new.index(tok.eot)]  # stop at "end of document"
print(prompt + tok.decode(new))
```

**How to write prompts for this model.** It has only learned to *continue documents*.
It has not learned to answer questions. So write the start of the document you want:

| Works | Doesn't work |
|---|---|
| `def is_prime(n):\n    """Return True if n is prime."""` | `Write a function that checks if a number is prime` |
| `import pandas as pd\n\ndf = pd.read_csv(` | `How do I read a CSV file?` |
| `Photosynthesis is the process by which` | `Explain photosynthesis to me` |

Making it answer questions needs instruction tuning (see below).

## What the test run on this repo showed

`configs/tiny_cpu.json` on 3M web + 4.7M code tokens, CPU only, 7.5 minutes:

- Validation loss went from 11.54 to 7.31. The starting value equals ln(100,352), what
  a model that knows nothing should score, which confirms the starting weights are right.
- Samples went from random tokens to code-shaped output (`def`, `return`, indentation,
  docstrings, `self.assertEqual`). They are still not correct code, as expected at
  this size.
- Also checked on CPU: 2-process `torchrun`, `torch.compile`, and checkpoint resume.
- **Not tested here (no GPU available):** CUDA, bf16, NCCL across real GPUs, and flash
  attention speed. These use standard PyTorch features but have not been run in this repo yet.

## How to make the model better (biggest impact first)

1. **Better data.** Quality beats quantity. Filtering and the web/code mix matter more
   than any architecture change.
2. **More compute, used well.** Grow model size and training tokens together
   (~20 tokens per parameter). A big model on little data wastes money.
3. **Longer context for code** (`max_seq_len` 2048–4096). Real code files are long.
4. **Instruction tuning (not built yet):** train on question → answer pairs so the
   model follows requests. Without this it only continues text.
5. **Real evaluation (not built yet):** validation loss is a start. For code, measure
   pass rate on HumanEval / MBPP by running the generated code against tests.

For a complete from-scratch pipeline that also includes chat tuning and evaluation,
see Andrej Karpathy's [nanochat](https://github.com/karpathy/nanochat).

## Project layout

```
prepare_data.py     download, filter, tokenize → data/<name>/{train,val}.bin + meta.json
train.py            training loop (CPU, 1 GPU, or many GPUs with torchrun)
sample.py           generate text from a checkpoint
estimate.py         compute and cost estimate for a config
configs/            tiny_cpu (test), colab_t4 (69M, free GPU), small (153M), medium (369M)
llm/model.py        the transformer
llm/data.py         batch loader with weighted data mixing
llm/quality.py      document filters, dedup, train/val split
llm/tokenizer.py    tiktoken wrapper
llm/config.py       settings + command-line overrides
tests/              unit tests and an end-to-end train → resume → sample test
```

## Data licenses

- FineWeb-Edu: ODC-By 1.0, and subject to Common Crawl's terms of use.
- codeparrot-clean: each file keeps its original license. Permissive licenses usually
  still require attribution. This is not legal advice; check before shipping a model.
