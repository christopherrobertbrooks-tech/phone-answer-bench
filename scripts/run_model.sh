#!/bin/bash
# run_model.sh LABEL MODEL.gguf [answer_quiz.py options...] -- one model through the quiz with llama.cpp's llama-server.
# Set LLAMA_SERVER (default: llama-server on PATH) and pick the GPU with CUDA_VISIBLE_DEVICES as usual.
# Examples:
#   scripts/run_model.sh ministral3-3b Ministral-3-3B-Instruct-2512-Q4_K_M.gguf
#   scripts/run_model.sh qwen35-4b-s1 Qwen3.5-4B-Q4_K_M.gguf --sample 0.7,0.8,20,0.0,1.5 --seed 1
set -euo pipefail
label=$1; model=$2; shift 2
SRV=${LLAMA_SERVER:-llama-server}; P=${PORT:-8490}
cd "$(dirname "$0")/.."
$SRV -m "$model" -ngl 99 -fa on -c 8192 -np 1 --jinja --cache-ram 0 --host 127.0.0.1 --port $P > results/server-$label.log 2>&1 &
spid=$!; trap 'kill $spid 2>/dev/null' EXIT
for i in $(seq 1 90); do curl -sf -m 3 localhost:$P/health > /dev/null && break; sleep 2; done
python3 scripts/answer_quiz.py http://127.0.0.1:$P/v1 "$label" "$@"
