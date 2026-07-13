#!/usr/bin/env bash
set -e
cd /workspace/exploration
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=1
SEEDS=1234,2025,7777,4242,31337

GMU=0.85

echo "=== [GPU1] DAPO best-pass@8: livecodebench ==="
python3 eval_code_multiseed_batched.py --model merged_dapo_pure_best_pass8 \
  --benchmarks livecodebench --seeds $SEEDS --n 8 --out dapo_pass8_livecodebench_5seed.json

echo "=== [GPU1] DAPO best-pass@8: math (amc23,aime24,aime25,aime26) ==="
python3 eval_math_multiseed_batched.py --model merged_dapo_pure_best_pass8 \
  --benchmarks amc23,aime24,aime25,aime26 --seeds $SEEDS --n 8 --gpu-memory-utilization $GMU --out dapo_pass8_math_5seed.json

echo "=== [GPU1] waiting for DAPO best-avg@8 LoRA merge ==="
while [ ! -f /workspace/exploration/merged_dapo_pure_best_avg8/config.json ]; do sleep 5; done
echo "=== [GPU1] DAPO best-avg@8: code (humanevalplus,mbppplus,livecodebench) ==="
python3 eval_code_multiseed_batched.py --model merged_dapo_pure_best_avg8 \
  --benchmarks humanevalplus,mbppplus,livecodebench --seeds $SEEDS --n 8 --gpu-memory-utilization $GMU --out dapo_avg8_code_5seed.json

echo "=== [GPU1] DAPO best-avg@8: math ==="
python3 eval_math_multiseed_batched.py --model merged_dapo_pure_best_avg8 \
  --benchmarks amc23,aime24,aime25,aime26 --seeds $SEEDS --n 8 --gpu-memory-utilization $GMU --out dapo_avg8_math_5seed.json

echo "=== [GPU1] DAPO SWEEP COMPLETE ==="
