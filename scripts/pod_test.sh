#!/usr/bin/env bash
# Erster Testlauf auf einer Miet-GPU, autonom (ohne SSH): läuft als Startbefehl eines Pods und schreibt alles
# Wichtige ins Log. Beispiel (Lium, Abrechnung pro Sekunde):
#
#   lium up --gpu H100 --name arena-test --ttl 1h --budget 5 -y \
#     --image nvidia/cuda:12.8.1-devel-ubuntu22.04 -e HF_TOKEN=... -e ARENA_TEST_MINUTES=40 \
#     --cmd "bash -c 'apt-get update -qq && apt-get install -y -qq git curl >/dev/null && \
#            git clone -q https://github.com/agent-arena-lab/arena /arena && /arena/scripts/pod_test.sh'"
#
# Ablauf: Setup (llama.cpp für die erkannte GPU) -> fertige GGUFs (prebuilt, schnell) -> Start-Check ->
# Pilot-Spiele bis zum Zeitbudget -> Auswertung -> Ergebnisse ins Log (summary.md, benchmark.json, Archiv base64).
# Umgebung: ARENA_TEST_MODELS (Standard: Qwen-Paar), ARENA_TEST_CONFIG (configs/pilot.yaml),
#           ARENA_TEST_MINUTES (Zeit für Spiele, Standard 30), HF_TOKEN (Gemma).
set -uo pipefail
cd "$(dirname "$0")/.."
export ARENA_QUANT=prebuilt HARNESS_MODE=native SERVER_MODE=native
export LLAMA_DIR=/opt/llama.cpp MODELS_DIR=/models HF_HOME=/models/.hf
MODELS="${ARENA_TEST_MODELS:-qwen3-30b-a3b-thinking qwen3-30b-a3b-thinking-abliterated}"
CONFIG="${ARENA_TEST_CONFIG:-configs/pilot.yaml}"
MINUTES="${ARENA_TEST_MINUTES:-30}"
t0=$(date +%s)
mark() { echo; echo "===== ARENA $* ($(( ($(date +%s) - t0) / 60 )) min) ====="; }

mark "SETUP"
nvidia-smi || true
scripts/setup_gpu.sh --skip-models --skip-docker || { mark "FEHLER setup"; exit 1; }
mark "MODELLE $MODELS"
# shellcheck disable=SC2086
scripts/prepare_models.sh $MODELS || { mark "FEHLER Modelle"; exit 1; }

mark "PILOT $CONFIG (max. $MINUTES min Spielzeit)"
# shellcheck disable=SC2086
timeout "$(( MINUTES * 60 ))" scripts/run_all.sh "$CONFIG" $MODELS
rc=$?
[[ $rc -eq 124 ]] && echo "Zeitbudget erreicht – Auswertung der fertigen Spiele" && scripts/serve.sh stop
exp="$(awk '/^name:/{print $2; exit}' "$CONFIG")"
.venv/bin/arena analyze "runs/$exp" >/dev/null 2>&1 || true
.venv/bin/arena export "runs/$exp" >/dev/null 2>&1 || true

mark "BENCHMARK"; cat "runs/$exp/benchmark.json" 2>/dev/null
mark "SUMMARY"; cat "runs/$exp/summary.md" 2>/dev/null
mark "SERVERS"; cat runs/_system/servers.json 2>/dev/null || ls "runs/$exp/_system" 2>/dev/null
archive="$(ls -t runs/_exports/*.tar.gz 2>/dev/null | head -n1)"
if [[ -n "$archive" ]]; then
  mark "ARCHIV $(basename "$archive") $(sha256sum "$archive" | cut -c1-64)"
  echo "-----BEGIN ARENA ARCHIVE-----"; base64 -w 0 "$archive"; echo; echo "-----END ARENA ARCHIVE-----"
fi
mark "FERTIG rc=$rc"
