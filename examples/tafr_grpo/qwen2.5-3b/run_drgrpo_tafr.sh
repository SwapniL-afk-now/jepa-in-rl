#!/usr/bin/env bash
set -euo pipefail

# Dr.GRPO + TAFR-GRPO | Qwen2.5-3B-Instruct | main_ppo + FSDP + hybrid-engine vLLM
#
# Dr.GRPO recipe (arXiv:2503.20783, mirrors examples/jepa_grpo_trainer/qwen2.5-3b/run_drgrpo.sh):
#   - Unnormalized group advantage (norm_adv_by_std_in_grpo=False), token-mean loss
#   - Symmetric PPO clip 0.2 (no Clip-Higher), dual-clip c=3.0 (verl default)
#   - NO overlong penalty, NO dynamic sampling (reward manager stays `dapo` for the acc metric)
# TAFR-GRPO on top: anchor + replay KL from frozen EMA/failure-SFT LoRA adapters,
# scored via vLLM multi-LoRA prefill in the awake-after-gen window.
# Speed recipe ported from the validated JEPA run_gc_jepa.sh EXTRA block:
# liger + fused triton kernels, fsdp_size=1, fp8 KV cache, fused old-log-prob
# (valid: one mini-batch per train batch -> exactly one optimizer step).

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$REPO_ROOT"

# Reduce CUDA fragmentation on the colocated (vLLM + FSDP) single GPU.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_deepscaler}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_SILENT=${WANDB_SILENT:-true}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
TAFR_VARIANT=${TAFR_VARIANT:-full}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-3B-Instruct}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-drgrpo_tafr_${TAFR_VARIANT}_qwen25_3b-${RUN_TIMESTAMP}}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}

TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
fi

# ── TAFR-GRPO hyperparameters ────────────────────────────────────────────────
TAFR_LOGPROB_BACKEND=${TAFR_LOGPROB_BACKEND:-vllm} # hf | vllm (vllm = fast multi-LoRA prefill in the awake-after-gen window)
TAFR_VLLM_SCORE_MICRO_BATCH_SIZE=${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE:-16}
TAFR_BETA=${TAFR_BETA:-0.1}
TAFR_ANCHOR_BETA=${TAFR_ANCHOR_BETA:-0.0}
TAFR_EMA_GAMMA=${TAFR_EMA_GAMMA:-0.9}
TAFR_MIX_ETA=${TAFR_MIX_ETA:-0.5}
TAFR_SFT_UPDATE_INTERVAL=${TAFR_SFT_UPDATE_INTERVAL:-2}
TAFR_CHECKPOINT_INTERVAL=${TAFR_CHECKPOINT_INTERVAL:-10}
TAFR_SFT_LR=${TAFR_SFT_LR:-5.0e-7}
TAFR_SFT_BATCH_SIZE=${TAFR_SFT_BATCH_SIZE:-8}
TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU:-}
TAFR_SAVE_TO_DISK_INTERVAL=${TAFR_SAVE_TO_DISK_INTERVAL:-0}
TAFR_SAVE_FAILURE_OPTIMIZER=${TAFR_SAVE_FAILURE_OPTIMIZER:-true}
TAFR_SFT_MAX_UPDATES=${TAFR_SFT_MAX_UPDATES:-9999}
TAFR_FAILURE_DATA_MAX_SIZE=${TAFR_FAILURE_DATA_MAX_SIZE:-null}
TAFR_FAILURE_DATA_SAMPLING=${TAFR_FAILURE_DATA_SAMPLING:-recent}

# ── Batch / lengths ──────────────────────────────────────────────────────────
TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
# One optimizer update per batch: mini batch == train batch keeps
# fuse_old_log_prob active (the trainer gate disables it otherwise).
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}   # 3B: 65536 OOM'd during actor backward (activation memory scales with this)
TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
USE_LORA=${USE_LORA:-true}
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ACTOR_LR=${ACTOR_LR:-5e-7}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
PPO_LOSS_COEF=${PPO_LOSS_COEF:-1}

# ── Dr.GRPO knobs ────────────────────────────────────────────────────────────
# Symmetric PPO clip (no DAPO Clip-Higher); clip_ratio_c is verl's dual-clip default.
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.2}
CLIP_RATIO_C=${CLIP_RATIO_C:-3.0}
# Dr.GRPO: no overlong length penalty (dapo reward manager kept for the acc metric).
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-false}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-614}
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}
# Dr.GRPO: no dynamic sampling (gen batch == train batch).
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-false}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}   # inert under main_ppo (no resample loop); oversample is one-shot via gen_batch_size
if [[ "${FILTER_GROUPS_ENABLE}" == "true" ]]; then
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$(( TRAIN_BATCH_SIZE * 3 ))}
else
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
fi
# High-exploration training rollouts (DAPO paper defaults; explicit for reproducibility).
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}

# ── Rollout / vLLM ───────────────────────────────────────────────────────────
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}   # 0.7+ OOM'd at sleep/wake weight-sync under fsdp_size=1; TAFR also hosts extra LoRA slots
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_LORAS=${ROLLOUT_MAX_LORAS:-3}           # actor + tafr_anchor + tafr_replay
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-8}    # halved vs. 1.5B to bound val-time rollout memory
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-250}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files=${VAL_FILES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    +data.gen_batch_size=${GEN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)
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
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)
if [[ "${USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_loss_coef=${PPO_LOSS_COEF}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
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
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras=${ROLLOUT_MAX_LORAS}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# DAPO overlong reward shaping, applied by the `dapo` reward manager (also
# emits the per-row `acc` metric that filter_groups and the TAFR failure
# collector prefer over the penalty-shaped reward).
REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${ENABLE_OVERLONG_BUFFER}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.resume_mode=auto
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
)

TAFR=(
    custom_tafr_grpo.enable=true
    custom_tafr_grpo.variant="${TAFR_VARIANT}"
    custom_tafr_grpo.logprob_backend="${TAFR_LOGPROB_BACKEND}"
    custom_tafr_grpo.vllm_score_micro_batch_size="${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE}"
    custom_tafr_grpo.beta="${TAFR_BETA}"
    custom_tafr_grpo.anchor_beta="${TAFR_ANCHOR_BETA}"
    custom_tafr_grpo.ema_gamma="${TAFR_EMA_GAMMA}"
    custom_tafr_grpo.mix_eta="${TAFR_MIX_ETA}"
    custom_tafr_grpo.disable_builtin_kl=true
    custom_tafr_grpo.sft_update_interval_grpo_steps="${TAFR_SFT_UPDATE_INTERVAL}"
    custom_tafr_grpo.checkpoint_interval_grpo_steps="${TAFR_CHECKPOINT_INTERVAL}"
    custom_tafr_grpo.failure_sft_lr="${TAFR_SFT_LR}"
    custom_tafr_grpo.failure_sft_batch_size="${TAFR_SFT_BATCH_SIZE}"
    custom_tafr_grpo.failure_sft_max_token_len_per_gpu="${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU}"
    custom_tafr_grpo.failure_sft_max_updates_per_interval="${TAFR_SFT_MAX_UPDATES}"
    custom_tafr_grpo.save_to_disk_interval_grpo_steps="${TAFR_SAVE_TO_DISK_INTERVAL}"
    custom_tafr_grpo.save_failure_sft_optimizer="${TAFR_SAVE_FAILURE_OPTIMIZER}"
    custom_tafr_grpo.failure_data_max_size="${TAFR_FAILURE_DATA_MAX_SIZE}"
    custom_tafr_grpo.failure_data_sampling="${TAFR_FAILURE_DATA_SAMPLING}"
)

# Local-box performance overrides (from the validated JEPA run_gc_jepa.sh recipe):
# full model replica per GPU, Liger + fused triton kernels, fp8 KV cache, fused
# old-log-prob. TAFR note: kv_cache_dtype=fp8 also quantizes the KV cache used
# by the anchor/replay multi-LoRA prefill scoring (slightly noisier KL);
# set TAFR_KV_CACHE_DTYPE=auto to score at full precision.
FUSE_OLD_LOG_PROB=${FUSE_OLD_LOG_PROB:-true}
TAFR_KV_CACHE_DTYPE=${TAFR_KV_CACHE_DTYPE:-fp8}
LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-98304}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}
EXTRA=(
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.model.use_liger=True
    actor_rollout_ref.model.use_fused_kernels=True
    actor_rollout_ref.model.fused_kernel_options.impl_backend=triton
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=${TAFR_KV_CACHE_DTYPE}
    +actor_rollout_ref.actor.fuse_old_log_prob=${FUSE_OLD_LOG_PROB}
)

VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} "$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${TAFR[@]}" \
    "${EXTRA[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
