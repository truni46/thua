#!/usr/bin/env bash
# Launch vLLM on an L4 (dev proxy for MiG H200) with the optimized serving flags.
# Serves directly from the HF hub (online FP8) so no offline quantize step is needed
# for a first benchmark pass. Flags mirror configs/base.yaml + exp_fp8.yaml.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/serve_l4.sh                 # serve Qwen/Qwen3.5-2B from hub
#   MODEL=/path/to/model ./scripts/serve_l4.sh            # serve a local (e.g. offline-fp8) checkpoint
#
# Then in another shell: python main.py all --config configs/experiment/exp_fp8.yaml
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"          # HF repo id or local path; verify exact id for the competition
SERVED_NAME="${SERVED_NAME:-Qwen3.5-2B}"    # must match `model:` in the YAML benchmark/accuracy blocks
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-45000}"
GPU_UTIL="${GPU_UTIL:-0.90}"
# Online FP8 weights by default. For a BF16 baseline (accuracy reference) set QUANT="".
QUANT="${QUANT:-fp8}"
# FP8 KV cache is supported on Ada (L4, cc 8.9) and Hopper. Set KV_DTYPE=auto for a BF16 KV baseline.
KV_DTYPE="${KV_DTYPE:-fp8_e4m3}"

ARGS=(
  --served-model-name "$SERVED_NAME"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --tensor-parallel-size 1
  --enable-prefix-caching
  --block-size 16
  --enable-chunked-prefill
  --max-num-batched-tokens 8192
  --max-num-seqs 32
  --scheduling-policy fcfs
  --preemption-mode recompute
)
[ -n "$QUANT" ]     && ARGS+=(--quantization "$QUANT")
[ "$KV_DTYPE" != "auto" ] && ARGS+=(--kv-cache-dtype "$KV_DTYPE")

echo "+ vllm serve $MODEL ${ARGS[*]}"
exec vllm serve "$MODEL" "${ARGS[@]}"
