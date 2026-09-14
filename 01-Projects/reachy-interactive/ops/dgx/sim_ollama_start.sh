#!/usr/bin/env bash
# Start a separate Ollama endpoint for simulation on physical GPU 0.
# This intentionally refuses to start while another workload occupies GPU 0.
set -euo pipefail

OLLAMA_BIN="${OLLAMA_BIN:-/home/kiro-ai/ollama-old/bin/ollama}"
SIM_OLLAMA_HOST="${SIM_OLLAMA_HOST:-127.0.0.1:11435}"
SIM_OLLAMA_MODELS="${SIM_OLLAMA_MODELS:-/home/kiro-ai/ollama/models}"
MIN_FREE_MB="${SIM_OLLAMA_MIN_FREE_MB:-14000}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi를 찾을 수 없어 시뮬 Ollama를 시작하지 않습니다." >&2
  exit 2
fi
used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F, '$1 ~ /^ *0 *$/ {gsub(/ /, "", $2); print $2; exit}')
used=${used:-999999}
total=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits \
  | awk -F, '$1 ~ /^ *0 *$/ {gsub(/ /, "", $2); print $2; exit}')
free=$((total - used))
if [ "$free" -lt "$MIN_FREE_MB" ]; then
  echo "GPU0 여유 ${free}MiB < ${MIN_FREE_MB}MiB; 기존 학습을 방해하지 않고 시작을 보류합니다." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST="$SIM_OLLAMA_HOST"
export OLLAMA_MODELS="$SIM_OLLAMA_MODELS"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:--1}"
echo "시뮬 Ollama 시작: ${OLLAMA_HOST} (physical GPU0, free ${free}MiB)"
exec "$OLLAMA_BIN" serve
