#!/bin/bash
# run_hf.sh LABEL MODEL_DIR [answer_quiz.py options...] -- one model llama.cpp can't run, through the quiz via
# scripts/hf_server.py (transformers, fp16). Set PYTHON to a venv with torch + transformers; pick the GPU with
# CUDA_VISIBLE_DEVICES as usual. The model is unloaded when the quiz ends (the server is stopped by its PID).
set -euo pipefail
label=$1; model=$2; shift 2
PY=${PYTHON:-python3}; P=${PORT:-8490}
cd "$(dirname "$0")/.."
if curl -sf -m 2 localhost:$P/health > /dev/null 2>&1; then echo "port $P is already in use: refusing to start"; exit 1; fi
$PY scripts/hf_server.py "$model" --port $P > results/server-$label.log 2>&1 < /dev/null &
spid=$!   # the server's own PID (started directly, not in a chain: a chain's PID is a shell, and the model outlives it)
stop() { kill $spid 2>/dev/null; for _ in $(seq 1 20); do kill -0 $spid 2>/dev/null || break; sleep 1; done
         kill -0 $spid 2>/dev/null && { echo "server $spid didn't stop: killing it"; kill -9 $spid; }; echo "model unloaded (pid $spid gone)"; }
trap stop EXIT
for i in $(seq 1 120); do curl -sf -m 3 localhost:$P/health > /dev/null && break; kill -0 $spid 2>/dev/null || { tail -20 results/server-$label.log; exit 1; }; sleep 2; done
echo "$label loaded in about $((i * 2)) s"
python3 scripts/answer_quiz.py http://127.0.0.1:$P/v1 "$label" "$@"
