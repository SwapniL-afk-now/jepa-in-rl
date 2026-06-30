# Reproducing this project on a fresh server

RL post-training research (verl, FSDP) on **Qwen2.5-Math-1.5B-Instruct**: a **DAPO**
baseline vs. the "ours" **JEPA / TCR-dual** auxiliary objective. Read this end-to-end
before running anything; it is the single source of truth for setup, layout, and
conventions on a machine that only has this git clone.

> This doc is committed so any agent/human on a new server can bootstrap without the
> original machine's local notes. The heavy artifacts live on HuggingFace (see §3).

---

## 1. Environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt          # verl + vLLM + torch stack
# secret: never committed. Create it by hand:
echo "HUGGINGFACE_API_KEY=hf_xxx" > .env     # used by training scripts (source .env) and HF pushes
```

Single GPU assumed (≈96 GB Blackwell here). Schedule jobs sequentially; each python
frees the GPU on exit. vLLM uses `VLLM_ATTENTION_BACKEND=FLASHINFER`.

### Gotchas that will bite on a new server
- **`unset RAY_ADDRESS`** before any verl launch. A shell with `RAY_ADDRESS=127.0.0.1`
  (no port) makes Ray die with `Invalid address format: 127.0.0.1`.
- **vLLM v1** needs eval scripts wrapped in `if __name__ == "__main__": main()`.
- **LoRA merge:** verl `model_merger` writes `adapter_config.json` with `lora_alpha=0`;
  set it to **1024** before PEFT `merge_and_unload` or the LoRA contribution is zeroed.
- Reward import path is `from verl.utils.reward_score import default_compute_score`.

---

## 2. Directory layout to recreate

Everything below `/workspace/jepa-grpo-cache` and `/workspace/models` is data/weights,
NOT in git — pull from HF (§3).

```
/workspace/
  exploration/                       # THIS repo (clone of jepa-in-rl)
  models/Qwen2.5-Math-1.5B-Instruct  # base model (public HF download)
  jepa-grpo-cache/
    data/        *.parquet           # training data (HF dataset repo)
    eval_data/   *.parquet           # 10 benchmark parquets (HF dataset repo)
    teacher_targets.pt(+.responses.pt)        # CoT teacher cache (HF dataset repo)
    code_teacher_targets.pt(+.responses.pt)   # code teacher cache (HF dataset repo)
  checkpoints/                       # training output (local; push best to HF)
```

---

## 3. Models & datasets — pull from HuggingFace

```bash
# base model (public)
hf download Qwen/Qwen2.5-Math-1.5B-Instruct --local-dir /workspace/models/Qwen2.5-Math-1.5B-Instruct

# teacher caches + reproduction parquets (one dataset repo)
hf download swapnil7777/framework-teacher-cache --repo-type dataset --local-dir /tmp/fwcache
#   -> copy *.pt to /workspace/jepa-grpo-cache/, data/ and eval_data/ into /workspace/jepa-grpo-cache/

# merged inference models (optional, for eval / as a training init)
hf download swapnil7777/framework      --local-dir /workspace/models/framework      # root=DAPO-JEPA step690 (ours); DR_GRPO_FW_QWEN_MATH_1.5B/=tcr-dual step330; DAPO_QWEN_MATH_1.5B/=baseline step170
hf download swapnil7777/dapo-baseline  --local-dir /workspace/models/dapo-baseline   # DAPO-only baseline merged
```

**Important:** only *merged inference models* exist on HF — there is **no resumable
verl checkpoint** (no optimizer/LoRA-adapter/sampler state) anywhere. A new server can
**train fresh** or **eval**, but cannot **resume** an interrupted run.

Training parquets are filtered from public sets (DAPO-Math-17k, DeepScaleR-Preview);
eval parquets are regenerable via `python3 -m verl.experimental.fepo.data` /
`...fepo.code_data`, but the exact files are on the dataset repo for byte-fidelity.

---

## 4. Training

Launch scripts in `examples/`:
- `examples/drgrpo_trainer/run_qwen_1_5b_dapo_fsdp.sh` — **pure DAPO** baseline (main_ppo + `dapo` reward manager).
- `examples/jepa_grpo_trainer/run_qwen_1_5b_dapo_ray.sh` — **DAPO + JEPA/TCR-dual** (Ray + FSDP + hybrid-engine vLLM). Set `JEPA_ENABLE=False` to run pure DAPO through this script.
- `examples/jepa_grpo_trainer/run_qwen25_1_5b_dual_ray.sh` — the lora=128 dual reference.

All scripts are env-var driven (defaults in their `### user-adjustable ###` block). Common knobs:

| Var | Meaning |
|---|---|
| `MODEL_PATH` | init checkpoint (base model, or a merged model to continue from) |
| `DAPO_USE_LORA`/`USE_LORA`, `LORA_RANK`, `LORA_ALPHA` | LoRA (rank 512 / alpha 1024 used here) |
| `MAX_OPTIMIZER_STEPS` | run length (800 = full schedule) |
| `BEST_CKPT_METRIC`/`BEST_CKPT_METRICS` | best-ckpt selection: `avg@8`, `pass@8`, `combined`, … over `BEST_CKPT_SOURCES` |
| `JEPA_ENABLE`, `JEPA_LOSS_TYPE`, `ALPHA`, `TEACHER_CACHE`, `CODE_TEACHER_CACHE` | JEPA-only (tcr-dual needs both caches) |
| `KL_COEF`/`KL_LOSS_COEF` | KL fully off requires **0.0** (use_kl_loss=false alone is not enough) |

Example — pure DAPO from a merged checkpoint, best-ckpt on avg@8:
```bash
cd /workspace/exploration
unset RAY_ADDRESS
MODEL_PATH=/workspace/models/framework EXPERIMENT_NAME=DAPO-RUN \
  DAPO_USE_LORA=true LORA_RANK=512 MAX_OPTIMIZER_STEPS=800 BEST_CKPT_METRIC=avg@8 \
  bash examples/drgrpo_trainer/run_qwen_1_5b_dapo_fsdp.sh 2>&1 | tee dapo_run.log
```

Run long jobs in **tmux** (detach/reattach), never blocking foreground. After a run:
push artifacts to HuggingFace, then clean local disk.

---

## 5. Evaluation

Scripts (repo root): `eval_math_benchmarks.py`, `eval_code_benchmarks.py`, fast code
judge `code_judge_worker.py` (one subprocess per program, SIGALRM timeout — do not
regress to verl's per-test subprocess scorer). `eval_code_fast.py` is a faster
single-generate n-split variant (NOT the canonical protocol — std understates seed
variance).

**Conventions (match exactly):**
- **3 fixed seeds**, default `1234,2025,7777`. Report **mean ± std** across seeds.
- Default rollout budget **n=8** (sometimes 16). At n=8, `pass@16` is meaningless — report only `pass@8` and `avg@1`.
- Math: amc23, aime24/25/26 (+ math500, minervamath, olympiadbench extended). Code: humanevalplus, mbppplus, livecodebench.
- Model context **4096** → `max_tokens 3072`.

```bash
python eval_code_benchmarks.py --model /workspace/models/framework \
  --benchmarks humanevalplus --seeds 1234,2025,7777 --n 8 --out result.json
```

---

## 6. Output: ICLR paper

Results feed `/workspace/paper/iclr2025_conference.tex` (+ `iclr2026_*`). LaTeX format
for a result: `$VAL${\scriptsize$\,\pm X.X$}`. Put exact numbers, right seeds, honest
framing; do not average old+new seed sets.
