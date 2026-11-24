#!/usr/bin/env bash
set -euo pipefail
: "${NABLA_PROFILE_ROOT:?Set the isolated experiment directory}"
: "${NABLA_PROFILE_PYTHON:?Set the uv-managed Python executable}"
: "${NABLA_PROFILE_MODEL:?Set the local checkpoint directory}"
: "${NABLA_PROFILE_NSYS:?Set the Nsight Systems executable}"
role=${1:?P or D}
export PYTHONPATH="$NABLA_PROFILE_ROOT:$NABLA_PROFILE_ROOT/source"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$(dirname "$NABLA_PROFILE_PYTHON"):$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="${NABLA_PROFILE_CUDA_COMPAT:-/usr/local/cuda-compat}:${LD_LIBRARY_PATH:-}"
export VLLM_USE_RUST_FRONTEND=0 VLLM_SERVER_DEV_MODE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_ENGINE_READY_TIMEOUT_S=3600 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 VLLM_PLUGINS='' UCX_NET_DEVICES=all
export NABLA_PROFILE_ROLE=$role
run=${NABLA_PROFILE_RUN:-formal}
mkdir -p "$NABLA_PROFILE_ROOT/raw/$run"
export NABLA_PROFILE_RECEIPTS="$NABLA_PROFILE_ROOT/raw/$run/receipts"
case "$role" in
  P) export CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_NIXL_SIDE_CHANNEL_PORT=5630
     port=8310; kv_role=kv_producer; cpus=0-50,104-154;;
  D) export CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_NIXL_SIDE_CHANNEL_PORT=5631
     port=8320; kv_role=kv_consumer; cpus=52-102,156-206;;
  *) exit 2;;
esac
cd "$NABLA_PROFILE_ROOT"
exec "$NABLA_PROFILE_NSYS" profile --force-overwrite=true \
  --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --trace-fork-before-exec=true --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=repeat \
  --output="$NABLA_PROFILE_ROOT/raw/$run/$role" \
  taskset -c "$cpus" "$NABLA_PROFILE_PYTHON" -m vllm.entrypoints.cli.main serve \
  "$NABLA_PROFILE_MODEL" --served-model-name qwen-profile \
  --host 127.0.0.1 --port "$port" --worker-cls profile_worker.ProfileWorker \
  --tensor-parallel-size 4 --enable-expert-parallel --renderer-num-workers 4 \
  --max-model-len 131072 --max-num-batched-tokens 4096 --max-num-seqs 128 \
  --gpu-memory-utilization 0.90 --engram-config '{"cpu_offload":true}' \
  --language-model-only --disable-prefix-caching --disable-extensible-kv-cache \
  --no-async-scheduling --moe-backend triton --kv-cache-dtype fp8 \
  --attention-config.indexer_kv_dtype fp8 \
  --profiler-config '{"profiler":"cuda","detailed_trace_annotation":true}' \
  --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$kv_role\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"enable_local_fastpath\":true}}"
