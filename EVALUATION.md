# Evaluate nanoGPT with lm-eval

`eval_adapter.py` loads native nanoGPT checkpoints and reuses lm-eval's Hugging
Face backend for tokenization, right-padded batching, continuation scoring, and
rolling document windows. The forward pass and weights remain nanoGPT's.

Supported: likelihood tasks such as WikiText-2, PIQA, and HellaSwag on a single
device. Generation-based tasks explicitly raise `NotImplementedError`.
Only WikiText-2 has been evaluated end-to-end so far.

## Environment

Tested on NVIDIA GB10 with Python 3.12.3, PyTorch 2.12.0+cu130,
`lm_eval==0.4.13`, `transformers==5.17.0`, and `datasets==5.0.1`.
The existing evaluation environment is:

```text
/home/tqxia/workspace/llm-evaluation/.venv
```

Use this environment for evaluation; nanoGPT training does not import lm-eval.
Its dependency lockfile is `/home/tqxia/workspace/llm-evaluation/requirements-lock.txt`.
The adapter subclasses `HFLM`, so rerun the parity checks when upgrading lm-eval.

## Evaluate a trained checkpoint

Substitute your checkpoint path below. The tokenizer must have exactly the same
token-to-ID mapping and special-token definitions used to prepare its training
data. Vocabulary bounds are checked, but equal vocabulary sizes do not establish
that the tokenizers match. Tokenizers must be loadable by Hugging Face
`AutoTokenizer`; nanoGPT's character-level `meta.pkl` alone is not supported.

```bash
cd /home/tqxia/workspace/nanoGPT
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 NVIDIA_TF32_OVERRIDE=0
run_dir=$(mktemp -d /home/tqxia/workspace/llm-evaluation/results/nanogpt-wikitext2-XXXXXX)
/home/tqxia/workspace/llm-evaluation/.venv/bin/python eval_adapter.py run \
  --model nanogpt \
  --model_args checkpoint=/absolute/path/to/ckpt.pt,tokenizer=/home/tqxia/workspace/llm-evaluation/gpt2,dtype=float32,attention=eager \
  --include_path /home/tqxia/workspace/llm-evaluation/tasks \
  --tasks wikitext2_local \
  --num_fewshot 0 --device cuda:0 --batch_size 4 \
  --seed 0,1234,1234,1234 \
  --output_path "$run_dir" --log_samples
```

`wikitext2_local` is the installed official WikiText task with local dataset paths;
its preprocessing and scoring are unchanged. All 62 test documents are scored.
Use the same task/protocol for comparisons with the saved public GPT-2 baseline.
The official `wikitext` task also works when its Hugging Face dataset is accessible.

Other adapter arguments:

- `max_length`: optional shorter evaluation window; must not exceed checkpoint
  `block_size`. Defaults to the checkpoint context size.
- `attention`: `sdpa` (default) or `eager`. Use `eager` and float32 for the public
  GPT-2 baseline parity protocol; SDPA is available for normal evaluation.
- `dtype`: `float32` (default), `bfloat16`, or `float16`. The model is cast to that
  dtype, not evaluated with training autocast. Record precision when comparing.
- `--batch_size`: integer or lm-eval's `auto` batch selection.

The checkpoint is loaded using `weights_only=True` and strict state-dict loading.
Compiled `_orig_mod.` prefixes are stripped. No training or optimizer updates
occur, and original checkpoint files are not modified. Padded vocabulary logits
remain in the softmax denominator, preserving the model's native distribution.

## Changes to model.py

`forward(..., return_all_logits=True)` returns `(B, T, V)` logits without requiring
targets or computing a training loss. The default generation path still returns
`(B, 1, V)`; supplying targets still computes cross-entropy as before.

`GPTConfig.gelu_approximate` defaults to `'none'`, preserving the activation used
by existing nanoGPT checkpoints. The imported public GPT-2 checkpoint explicitly
uses `'tanh'` to match Hugging Face's `gelu_new`. Do not change the activation of a
trained nanoGPT checkpoint just to resemble public GPT-2.

## Validate against public GPT-2

The converter imports local Hugging Face GPT-2 weights, transposes its Conv1D
weight matrices to match nanoGPT Linear layers, and preserves the public model's
activation semantics. It refuses to overwrite an existing output file.
The generated checkpoint is an evaluation reference, not a new training run.

```bash
cd /home/tqxia/workspace/nanoGPT
/home/tqxia/workspace/llm-evaluation/.venv/bin/python import_gpt2_for_eval.py \
  --source /home/tqxia/workspace/llm-evaluation/gpt2 \
  --output /home/tqxia/workspace/llm-evaluation/adapter-validation/public-gpt2.pt
```

This reference checkpoint has already been created on the GB10. Reuse it for:

```bash
HF_HUB_OFFLINE=1 NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=1 \
/home/tqxia/workspace/llm-evaluation/.venv/bin/python test_eval_adapter.py \
  --checkpoint /home/tqxia/workspace/llm-evaluation/adapter-validation/public-gpt2.pt \
  --tokenizer /home/tqxia/workspace/llm-evaluation/gpt2 \
  --output /home/tqxia/workspace/llm-evaluation/adapter-validation/checks.json
```

Checks cover legacy checkpoints, compiled prefixes, full-position logits,
unchanged default forward behavior, vocabulary/context bounds, tokenization and
logit parity with HF, continuation masking, batch invariance, and rolling
likelihood at window boundaries (including empty text).

To run full WikiText-2 parity, use the evaluation command above with
`checkpoint=/home/tqxia/workspace/llm-evaluation/adapter-validation/public-gpt2.pt`.
Results and validation logs are under
`/home/tqxia/workspace/llm-evaluation/adapter-validation/`.

## Validation results (2026-09-27)

Full WikiText-2 test split, 62 documents, float32, eager attention, 1,024-token
context, batch size 4, identical public GPT-2 weights:

| Metric | HF baseline | nanoGPT adapter | Absolute difference |
|---|---:|---:|---:|
| Word perplexity | 37.3698283902 | 37.3698293081 | 9.18e-7 |
| Byte perplexity | 1.9682003603 | 1.9682003693 | 9.04e-9 |
| Bits per byte | 0.9768770927 | 0.9768770993 | 6.63e-9 |

The targeted parity checks also passed: maximum logit absolute difference
`1.53e-4`, continuation log-probability difference `1.62e-5`, and rolling
log-probability difference `1.53e-5` on the test inputs.

The native 51M training checkpoint at
`/home/tqxia/workspace/llm-learning/gb10-sizing-20260923/smoke-checkpoint/ckpt.pt`
also loaded and scored a short document successfully using SDPA and its original
exact GELU. This was a loading/forward smoke test, not a full model benchmark.

Machine-readable evidence: `checks.json`, `baseline-comparison.json`,
`native-smoke.json`, and `wikitext2/` under the remote validation directory.
