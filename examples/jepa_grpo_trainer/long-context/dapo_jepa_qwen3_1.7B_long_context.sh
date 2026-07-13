#!/usr/bin/env bash
# DAPO + JEPA | Qwen2.5-Math-1.5B-Instruct-yarn32k | Ray + FSDP + hybrid-engine vLLM
# DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization, arXiv:2503.14476) keeps the
# GRPO group-relative advantage WITH std normalization and adds:
#   - Clip-Higher: decoupled clip epsilons (clip_ratio_low=0.2, clip_ratio_high=0.28) + clip_ratio_c
#   - Token-level policy-gradient loss (loss_agg_mode=token-mean)
#   - Overlong reward shaping via the `dapo` reward manager (soft length penalty)
#   - Dynamic sampling (group filtering) via algorithm.filter_groups
#   - KL divergence dropped entirely (no KL loss, no entropy bonus)
# combined with the JEPA cc-jepa dual-view objective: 8 CoT rollouts/prompt (CoT-only,
# no code arm). JEPA auto-off patience 10. The student rollout remains short (1024
# prompt + 3072 response). Its JEPA target view is separately allowed to read a
# complete teacher trace (up to TARGET_MAX_LENGTH) in a no-grad forward. Do not
# increase the rollout budget merely because the target encoder has a larger
# context window.
#
# CC-JEPA (CoT-only): L = (1-lambda)*L_CC-align + lambda*L_SIGReg, where L_CC-align
# is a single Bernoulli likelihood over the JEPA distance d = 1 - cos(pred, z^+):
# correct rollouts pull toward the teacher target, incorrect rollouts act as negative
# views pushed away. SIGReg (anti-collapse) applies over the full correct+wrong pool
# since JEPA_ANCHOR_SET=all. All-correct groups reduce L_CC-align exactly to ordinary
# cosine LLM-JEPA / tau.
#
# Same dataset/teacher/model recipe as drgrpo_jepo_qwen25_math_1.5b_long_context.sh
# in this directory (DAPO math 17k train file, matching teacher_targets.pt.responses.pt) —
# this script is the DAPO-algorithm counterpart to that Dr.GRPO one, so the two are
# directly comparable at matched steps.
#
# Before first run:
#   make sure the model is at /workspace/models/Qwen2.5-Math-1.5B-Instruct-yarn32k
#   (see drgrpo_jepo_qwen25_math_1.5b_long_context.sh for how the YaRN variant is built)

set -euo pipefail
if [[ "${DEBUG_LAUNCH:-0}" == "1" ]]; then
    set -x
fi

# Load local secrets/config without echoing them under `set -x`.
if [[ -f .env ]]; then
    _XTRACE_WAS_ON=0
    case $- in
        *x*) _XTRACE_WAS_ON=1; set +x ;;
    esac
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    if [[ ${_XTRACE_WAS_ON} -eq 1 ]]; then
        set -x
    fi
    unset _XTRACE_WAS_ON
fi

# Default local caches for repeatable training runs.
export HF_HOME=${HF_HOME:-/workspace/jepa-grpo-cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/jepa-grpo-cache/hf/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/workspace/jepa-grpo-cache/hf/hub}
export TORCH_HOME=${TORCH_HOME:-/workspace/jepa-grpo-cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/jepa-grpo-cache/xdg}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/jepa-grpo-cache/vllm}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"

# Keep terminal logs focused on run status, samples, metrics, and warnings that matter.
export VERL_PRINT_CONFIG=${VERL_PRINT_CONFIG:-0}
export VERL_VERBOSE_METRICS=${VERL_VERBOSE_METRICS:-0}
export VERL_CONSOLE_FULL_METRICS=${VERL_CONSOLE_FULL_METRICS:-0}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export RAY_DISABLE_DOCKER_CPU_WARNING=${RAY_DISABLE_DOCKER_CPU_WARNING:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB_MODE=${WANDB_MODE:-online}

########################### user-adjustable ###########################
# Qwen2.5-Math-1.5B-Instruct with YaRN RoPE scaling (4096 -> 32768) baked into
# config.json, so the target branch can encode the full long teacher trace
# (mean ~8k, up to ~22k tokens) in one causal forward. Weights are symlinked;
# only config.json differs. YaRN applies to ALL consumers (FSDP actor, target
# encoder, and vLLM) since they read this config.json, keeping both JEPA branches
# in the SAME representation space (required by TCR-JEPA within-space alignment).
EXPECTED_MODEL_BASENAME=${EXPECTED_MODEL_BASENAME:-Qwen2.5-Math-1.5B-Instruct-yarn32k}
MODEL_PATH=${MODEL_PATH:-/workspace/models/${EXPECTED_MODEL_BASENAME}}
ALLOW_MODEL_PATH_OVERRIDE=${ALLOW_MODEL_PATH_OVERRIDE:-false}
PYTHON_BIN=${PYTHON_BIN:-/workspace/exploration/.venv/bin/python3}
if [[ ! -x "${PYTHON_BIN}" ]]; then
    PYTHON_BIN=python3
fi
# Dataset matched to the qwen2.5-math-1.5b/run_drgrpo_jepa.sh recipe (DAPO math 17k),
# same as drgrpo_jepo_qwen25_math_1.5b_long_context.sh in this directory.
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
AMC23_ONLY=${AMC23_ONLY:-false}
# Contamination-free cutoff for LiveCodeBench: only problems released on/after this
# date are kept. Should match the base model's pretraining cutoff, not the RL
# dataset's date. DeepSeek-R1-Distill-Qwen is distilled from Qwen2.5, reported with
# a mid-2024 knowledge cutoff.
LCB_CUTOFF_DATE=${LCB_CUTOFF_DATE:-2024-08-01}
# Full 10-benchmark suite (math + code) — expensive (2405 problems x VAL_ROLLOUT_N
# samples each). Used for the one-off final evaluation after training, not every
# test_freq step. Set VAL_FILES=$FULL_VAL_FILES to run it during training instead.
FULL_VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet,${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/humanevalplus.parquet,${EVAL_DATA_DIR}/mbppplus.parquet,${EVAL_DATA_DIR}/livecodebench.parquet]"
# Fast core-math subset (AIME 24/25/26 + AMC23, ~149 problems) used for routine
# in-training validation and best-checkpoint selection (see BEST_CKPT_SOURCES below).
# The remaining benchmarks (math500, minervamath, olympiadbench, code suites) are
# held out and only evaluated once at the end, with the best checkpoint -- see
# FINAL_VAL_FILES below.
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
fi
# data_source names (see verl/experimental/fepo/data.py) used to compute the average
# accuracy that decides whether to overwrite the `best/` checkpoint. Must be a subset
# of whatever's actually in VAL_FILES, or best-checkpoint tracking silently no-ops.
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
# Best-checkpoint selection metric (verl/trainer/ppo/ray_trainer.py):
#   combined (default) = 0.5*(avg@k + pass@k) | avg | pass | avg@N | pass@N
# "pass@8" selects on the per-source val pass_at_8 averaged across BEST_CKPT_SOURCES,
# i.e. maximize MEAN pass@8 (best-of-8 success rate). VAL_ROLLOUT_N must be >= 8.
BEST_CKPT_METRICS=${BEST_CKPT_METRICS:-'["avg@8","pass@8"]'}
# Held-out benchmarks for the one-off final evaluation after training, using the
# best checkpoint (not seen during training validation).
FINAL_VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet,${EVAL_DATA_DIR}/humanevalplus.parquet,${EVAL_DATA_DIR}/mbppplus.parquet,${EVAL_DATA_DIR}/livecodebench.parquet]"
if [[ "${PREPARE_EVAL_DATA}" == "true" ]]; then
    mkdir -p "${EVAL_DATA_DIR}"
    for ds in math500 olympiadbench amc23 aime24 aime25 aime26 minervamath; do
        if [[ ! -f "${EVAL_DATA_DIR}/${ds}.parquet" ]]; then
            "${PYTHON_BIN}" -m verl.experimental.fepo.data \
                --dataset "math-ai/${ds}" --split test \
                --output "${EVAL_DATA_DIR}/${ds}.parquet"
        fi
    done
    # Code benchmarks (see verl/experimental/fepo/code_data.py). LiveCodeBench is
    # restricted to its stdin/stdout-test subset and filtered to LCB_CUTOFF_DATE.
    for cb in humanevalplus mbppplus; do
        if [[ ! -f "${EVAL_DATA_DIR}/${cb}.parquet" ]]; then
            "${PYTHON_BIN}" -m verl.experimental.fepo.code_data \
                --benchmark "${cb}" --output "${EVAL_DATA_DIR}/${cb}.parquet"
        fi
    done
    if [[ ! -f "${EVAL_DATA_DIR}/livecodebench.parquet" ]]; then
        "${PYTHON_BIN}" -m verl.experimental.fepo.code_data \
            --benchmark livecodebench --cutoff-date "${LCB_CUTOFF_DATE}" \
            --output "${EVAL_DATA_DIR}/livecodebench.parquet"
    fi
fi

if [[ "${AMC23_ONLY}" == "true" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/amc23.parquet]"
    BEST_CKPT_SOURCES='["amc23"]'
fi

# DAPO uses data.train_batch_size as prompts per optimizer step.
# On the local 96GB Blackwell GPU, 64 prompts x 8 rollouts gave the best tested utilization.
TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
N_COT=${N_COT:-8}
N_CODE=${N_CODE:-0}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}   # student rollout generation budget; shared by train + eval
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-49152}   # 2x: pack more tokens per PPO-update micro-batch to raise low actor-update GPU util (~19GB headroom at 24576)
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
DAPO_USE_LORA=${DAPO_USE_LORA:-true}   # false -> full fine-tuning; set true to re-enable LoRA
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ACTOR_LR=${ACTOR_LR:-5e-7}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
# DAPO Clip-Higher: decoupled lower/upper PPO clip epsilons, plus clip_ratio_c (dual-clip).
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
# DAPO drops the KL term entirely (no KL loss, no KL-in-reward). Override to re-enable.
USE_KL_LOSS=${USE_KL_LOSS:-false}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}

# vLLM async-engine drain before back-to-back CoT/Code rollout RPCs.
JEPA_VLLM_DRAIN_S=${JEPA_VLLM_DRAIN_S:-2}
export JEPA_VLLM_DRAIN_S

# JEPA/TCR-dual auxiliary objective.
JEPA_ENABLE=${JEPA_ENABLE:-True}
ALPHA=${ALPHA:-0.03}
EMA_DECAY=${EMA_DECAY:-0.99}
# One full teacher trajectory is encoded per no-grad micro-batch.  This avoids
# padding multiple 32k+ trajectories to one width and keeps target memory bounded.
# cc-jepa uses jepa_anchor_set=all (512 anchor rows/step vs ~150 correct-only).
EMBED_MICRO_BATCH_SIZE=${EMBED_MICRO_BATCH_SIZE:-16}   # 32 measured slower (compute-bound, more mem pressure); 16 was the fastest observed
# Qwen2.5-Math YaRN variant supports 32,768 positions. Targets are truncated to
# this so no [x, y^+] exceeds the extended context (beyond which YaRN positions
# are again out-of-distribution). Covers the 1,024-token prompt + the ~22k-token
# max teacher trace with headroom; median trace is ~7k.
TARGET_MAX_LENGTH=${TARGET_MAX_LENGTH:-32768}
MIN_VALID_PAIRS=${MIN_VALID_PAIRS:-2}
# CC-JEPA (CoT-only): L = (1-lambda)*L_CC-align + lambda*L_SIGReg, where
# L_CC-align is a single Bernoulli likelihood over the JEPA distance
# d = 1 - cos(pred, z^+): correct rollouts pull toward the teacher target,
# incorrect rollouts act as negative views pushed away (replaces the old
# correct-only stratified align_cot term). SIGReg (anti-collapse) is
# unchanged in kind, now applied over the full correct+wrong pool since
# JEPA_ANCHOR_SET=all. No self/code arms, no stratified weighting on the
# align side. All-correct groups reduce L_CC-align exactly to ordinary
# cosine LLM-JEPA / tau. (Old objective: JEPA_LOSS_TYPE=jepa-tcr-cot,
# align_cot(correct-only) + TRIPLET_SIGREG_LAMBDA * SIGReg.)
JEPA_LOSS_TYPE=${JEPA_LOSS_TYPE:-cc-jepa}
CC_TAU=${CC_TAU:-0.5}   # latent correctness scale: P(correct|d) = exp(-d/tau)
# 0 = isolate the differentiable JEPA loss (align_cot + SIGReg); no reward-shaping
# advantage term, so any effect is attributable to representation alignment alone.
JEPA_REWARD_BETA=${JEPA_REWARD_BETA:-0.0}
JEPA_SIGMA_FLOOR=${JEPA_SIGMA_FLOOR:-0.1}
# Teacher CoT view matched to the qwen2.5-math-1.5b/run_drgrpo_jepa.sh recipe.
TEACHER_CACHE=${TEACHER_CACHE:-/workspace/jepa-grpo-cache/teacher_targets.pt.responses.pt}
# CoT-only mode: no code teacher cache.
CODE_TEACHER_CACHE=${CODE_TEACHER_CACHE:-}
CODE_SYSTEM_PROMPT=${CODE_SYSTEM_PROMPT:-}
TCR_MATCH=${TCR_MATCH:-cycle}
SELF_CONSIST_W=${SELF_CONSIST_W:-1.0}
AUTO_OFF_ENABLE=${AUTO_OFF_ENABLE:-True}
AUTO_OFF_PATIENCE=${AUTO_OFF_PATIENCE:-10}
AUTO_OFF_MIN_DELTA=${AUTO_OFF_MIN_DELTA:-0.002}
AUTO_OFF_WARMUP=${AUTO_OFF_WARMUP:-20}
AUTO_OFF_MIN_COS=${AUTO_OFF_MIN_COS:-0.0}
# cc-jepa needs ALL rollouts: correct = positives, wrong = negative views.
JEPA_ANCHOR_SET=${JEPA_ANCHOR_SET:-all}
LLM_JEPA_PREDICTOR_K=${LLM_JEPA_PREDICTOR_K:-1}
TRIPLET_SIGREG_LAMBDA=${TRIPLET_SIGREG_LAMBDA:-0.3}   # cc-jepa: L=(1-λ)*CC-align + λ*SIGReg(pool)
# Match the student's hidden_size (1536 for Qwen2.5-Math-1.5B): enough random
# directions to span the actual embedding space without paying for 4096's
# ~2.7x oversampling; SIGReg cost scales with M.
N_PROJECTIONS=${N_PROJECTIONS:-1536}
JEPA_MAX_GRAD_NORM=${JEPA_MAX_GRAD_NORM:-0.5}
ALPHA_WARMUP_STEPS=${ALPHA_WARMUP_STEPS:-0}   # 0 = full alpha from step 1, no linear ramp

# DAPO overlong reward shaping (soft length penalty applied by the `dapo` reward manager).
# Canonical DAPO uses a buffer of ~20% of max_response_length (official recipe: 4096/20480),
# with the penalty ramping linearly from 0 to penalty_factor over that final window.
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-true}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-614}          # ~20% of MAX_RESPONSE_LENGTH=3072 (matches DAPO's 4096/20480 ratio)
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

# DAPO dynamic sampling (group filtering). Canonical DAPO oversamples the generation
# batch (official recipe: gen 1536 / train 512 = 3x) and keeps groups whose rollouts
# are non-degenerate under the configured metric (acc mixed rather than all-0/all-1).
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-true}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}
# Under main_ppo (no resample loop) gen must equal train, else extra prompts are generated
# and silently dropped. With dynamic sampling on, DAPO's 3x oversample applies instead.
if [[ "${FILTER_GROUPS_ENABLE}" == "true" ]]; then
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$(( TRAIN_BATCH_SIZE * 3 ))}
else
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
fi

# DAPO training-rollout sampling: high-exploration generation (paper §4.1). Temperature 1.0
# with unrestricted top_p/top_k is what produces the diverse groups dynamic sampling/clip-higher
# rely on. (verl's rollout defaults already match these; set explicitly for reproducibility.)
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}   # vLLM KV-cache fraction; 0.7+ OOM'd at sleep/wake weight-sync under fsdp_size=1 (full replica per GPU)
# ROLLOUT_N matches NUM_GENERATIONS for consistency
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-auto}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}   # more parallel sequences during vLLM rollout
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}    # halved vs. 1.5B to bound val-time rollout memory
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VALIDATION_SEEDS=${VALIDATION_SEEDS:-'[31415,27182,16180]'}
VAL_TOP_P=${VAL_TOP_P:-0.95}
TRAIN_SEED=${TRAIN_SEED:-31415}   # data-loader shuffle seed; same value as the first validation seed

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-800}   # fixed-length run on dapo-math-17k
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}   # skip the step-0 pre-train validation; go straight into training
EVAL_ONLY=${EVAL_ONLY:-false}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}   # only the most recent checkpoint is kept on disk

# verl's wandb logger uses trainer.project_name (=PROJECT_NAME) as the wandb
# project; the WANDB_PROJECT env var is IGNORED. So set PROJECT_NAME itself to
# the desired wandb project. Runs are distinguished by EXPERIMENT_NAME below.
PROJECT_NAME=${PROJECT_NAME:-dapo_qwen_1-5b}
export WANDB_ENTITY=${WANDB_ENTITY:-ismamnurswapnil-bangladesh-university-of-engineering-and}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
MODEL_NAME_FROM_PATH=$(basename "${MODEL_PATH}")
MODEL_NAME=${MODEL_NAME_OVERRIDE:-${MODEL_NAME_FROM_PATH}}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-dapo-jepa-cot-contrastive-${MODEL_NAME//\//-}-${RUN_TIMESTAMP}}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
########################### end user-adjustable ###########################

if [[ "$(basename "${MODEL_PATH}")" != "${EXPECTED_MODEL_BASENAME}" && "${ALLOW_MODEL_PATH_OVERRIDE}" != "true" ]]; then
    echo "Refusing to run this recipe with MODEL_PATH=${MODEL_PATH}" >&2
    echo "Expected basename: ${EXPECTED_MODEL_BASENAME}" >&2
    echo "Set ALLOW_MODEL_PATH_OVERRIDE=true only if this mismatch is intentional." >&2
    exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Missing model directory: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ "${EVAL_ONLY}" == "true" ]]; then
    VAL_BEFORE_TRAIN=true
fi

if (( N_COT + N_CODE != ROLLOUT_N )); then
    echo "N_COT (${N_COT}) + N_CODE (${N_CODE}) must equal ROLLOUT_N (${ROLLOUT_N})" >&2
    exit 1
fi

if [[ "${JEPA_ENABLE}" == "True" ]]; then
# Skip empty entries (CODE_TEACHER_CACHE may be unset if overridden back to CoT-only).
for c in "${TEACHER_CACHE}" "${CODE_TEACHER_CACHE}"; do
    [[ -z "${c}" ]] && continue
    if [[ ! -f "${c}" ]]; then
        echo "Missing JEPA teacher cache: ${c}" >&2
        echo "Download/copy it into /workspace/jepa-grpo-cache or override TEACHER_CACHE/CODE_TEACHER_CACHE." >&2
        exit 1
    fi
done
fi

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    # DAPO: keep GRPO group-relative advantage WITH std normalization (unlike Dr.GRPO).
    algorithm.norm_adv_by_std_in_grpo=True
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files=${VAL_FILES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    +data.gen_batch_size=${GEN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    # (Qwen2.5-Math has no thinking-mode template; enable_thinking kwarg omitted.)
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# DAPO dynamic sampling (group filtering).
if [[ "${FILTER_GROUPS_ENABLE}" == "true" ]]; then
    DATA+=(
        +algorithm.filter_groups.enable=True
        +algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}
        +algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}
    )
fi

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRAD_CKPT:-True}
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)

if [[ "${DAPO_USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    # DAPO token-level loss: token-mean across all tokens in the mini-batch.
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    # DAPO Clip-Higher: decoupled lower/upper clip epsilons + dual-clip constant.
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P}
    actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

JEPA=(
    jepa.enable=${JEPA_ENABLE}
    jepa.n_cot=${N_COT}
    jepa.n_code=${N_CODE}
    jepa.alpha=${ALPHA}
    jepa.ema_decay=${EMA_DECAY}
    jepa.embed_micro_batch_size=${EMBED_MICRO_BATCH_SIZE}
    jepa.target_max_length=${TARGET_MAX_LENGTH}
    jepa.min_valid_pairs=${MIN_VALID_PAIRS}
    jepa.loss_type=${JEPA_LOSS_TYPE}
    jepa.cc_tau=${CC_TAU}
    jepa.predictor_k=${LLM_JEPA_PREDICTOR_K}
    jepa.triplet_sigreg_lambda=${TRIPLET_SIGREG_LAMBDA}
    jepa.n_projections=${N_PROJECTIONS}
    jepa.alpha_warmup_steps=${ALPHA_WARMUP_STEPS}
    jepa.max_grad_norm=${JEPA_MAX_GRAD_NORM}
    jepa.teacher_cache_path=${TEACHER_CACHE}
    jepa.code_teacher_cache_path=${CODE_TEACHER_CACHE}
    jepa.code_system_prompt="${CODE_SYSTEM_PROMPT}"
    jepa.tcr_match=${TCR_MATCH}
    jepa.jepa_anchor_set=${JEPA_ANCHOR_SET}
    jepa.self_consist_w=${SELF_CONSIST_W}
    jepa.tcr_reward_beta=${JEPA_REWARD_BETA}
    jepa.tcr_reward_sigma_floor=${JEPA_SIGMA_FLOOR}
    jepa.auto_off_enable=${AUTO_OFF_ENABLE}
    jepa.auto_off_patience=${AUTO_OFF_PATIENCE}
    jepa.auto_off_min_delta=${AUTO_OFF_MIN_DELTA}
    jepa.auto_off_warmup_steps=${AUTO_OFF_WARMUP}
    jepa.auto_off_min_cos=${AUTO_OFF_MIN_COS}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# DAPO overlong reward shaping, applied by the `dapo` reward manager.
REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${ENABLE_OVERLONG_BUFFER}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    +trainer.val_only=${EVAL_ONLY}
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
    +trainer.best_ckpt_metrics=${BEST_CKPT_METRICS}
    +trainer.validation_seeds=${VALIDATION_SEEDS}
)

# Local-box performance overrides (from the validated run_gc_jepa.sh recipe):
# full model replica per GPU, Liger + fused triton kernels, fp8 KV cache,
# fused old-log-prob (valid: filtering keeps groups <= ppo_mini_batch_size).
EXTRA=(
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=98304
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=98304
    actor_rollout_ref.model.use_liger=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${VLLM_KV_CACHE_DTYPE}
    +actor_rollout_ref.actor.fuse_old_log_prob=True
)

########################### launch ###########################
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} "${PYTHON_BIN}" -m verl.experimental.jepa_grpo.main_ray \
    --config-name jepa_grpo_ray_qwen25_1_5b \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${JEPA[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
