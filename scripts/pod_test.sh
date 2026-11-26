#!/usr/bin/env bash
# Erster Testlauf auf einer Miet-GPU, autonom (ohne SSH): läuft als Startbefehl eines Pods und schreibt alles
# Wichtige ins Log. Beispiel (Lium, Abrechnung pro Sekunde):
#
#   lium up --gpu H100 --name arena-test --ttl 1h --budget 5 -y \
#     --image nvidia/cuda:12.8.1-devel-ubuntu22.04 -e ARENA_TEST_DEADLINE=52 \
#     --cmd "bash -c 'apt-get update -qq && apt-get install -y -qq git curl >/dev/null && \
#            git clone -q https://github.com/agent-arena-lab/arena /arena && /arena/scripts/pod_test.sh'"
#
# Ablauf: Setup (llama.cpp für die erkannte GPU) -> fertige GGUFs (prebuilt, schnell) -> Start-Check ->
# Spiele bis zum Zeitbudget -> Auswertung -> Ergebnisse ins Log (summary.md, benchmark.json, Thinking-Auszüge,
# Archiv base64).
# Umgebung: ARENA_TEST_MODELS (Standard: Modelle aus der Config), ARENA_TEST_CONFIG (configs/podtest.yaml: alle
#           vier Modelle, kurze Spiele), ARENA_TEST_DEADLINE (Minuten ab Start, bis dahin müssen die Spiele
#           enden, Standard 52 – passt zu --ttl 1h), ARENA_TEST_MINUTES (feste Spielzeit statt Deadline),
#           HF_TOKEN (nur für ARENA_QUANT=self bei Gemma).
set -uo pipefail
cd "$(dirname "$0")/.."
export ARENA_QUANT=prebuilt HARNESS_MODE=native SERVER_MODE=native
export LLAMA_DIR=/opt/llama.cpp MODELS_DIR=/models HF_HOME=/models/.hf
CONFIG="${ARENA_TEST_CONFIG:-configs/podtest.yaml}"
MODELS="${ARENA_TEST_MODELS:-$(awk '/^models:/{f=1;next} f&&/^ *- /{print $2;next} f&&!/^ *$/{exit}' "$CONFIG" | xargs)}"
DEADLINE="${ARENA_TEST_DEADLINE:-52}"
t0=$(date +%s)
mark() { echo; echo "===== ARENA $* ($(( ($(date +%s) - t0) / 60 )) min) ====="; }

mark "SETUP"
nvidia-smi || true
scripts/setup_gpu.sh --skip-models --skip-docker || { mark "FEHLER setup"; exit 1; }
mark "MODELLE $MODELS"
# shellcheck disable=SC2086
scripts/prepare_models.sh $MODELS || { mark "FEHLER Modelle"; exit 1; }

if [[ -n "${ARENA_TEST_MINUTES:-}" ]]; then secs=$(( ARENA_TEST_MINUTES * 60 ))
else secs=$(( DEADLINE * 60 - ($(date +%s) - t0) )); fi
(( secs < 300 )) && secs=300
mark "SPIELE $CONFIG (max. $(( secs / 60 )) min Spielzeit)"
# shellcheck disable=SC2086
timeout "$secs" scripts/run_all.sh "$CONFIG" $MODELS
rc=$?
[[ $rc -eq 124 ]] && echo "Zeitbudget erreicht – Auswertung der fertigen Spiele" && scripts/serve.sh stop
exp="$(awk '/^name:/{print $2; exit}' "$CONFIG")"
.venv/bin/arena analyze "runs/$exp" >/dev/null 2>&1 || true
.venv/bin/arena export "runs/$exp" >/dev/null 2>&1 || true

mark "BENCHMARK"; cat "runs/$exp/benchmark.json" 2>/dev/null
mark "SUMMARY"; cat "runs/$exp/summary.md" 2>/dev/null
mark "THINKING-AUSZÜGE"
# Pro Spiel: Anteil Züge mit Thinking und die ersten beiden Thinking-Texte (gekürzt). Volltext im Archiv.
for ev in runs/"$exp"/*/events.jsonl; do
  [[ -f "$ev" ]] || continue
  echo "--- $(basename "$(dirname "$ev")") ---"
  .venv/bin/python - "$ev" <<'PY' || true
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rs = [r for r in rs if r.get("event") == "llm_response"]
th = [r for r in rs if r.get("reasoning")]
print(f"Antworten: {len(rs)} · mit Thinking: {len(th)} · Limit erreicht: {sum(bool(r.get('limit_hit')) for r in rs)}")
for r in th[:2]:
    t = r["reasoning"]
    print(f"[Agent {r.get('agent')}, Aufgabe {r.get('task')}, Zug {r.get('turn')}, {len(t)} Zeichen]")
    print(t[:1500] + (" [...]" if len(t) > 1500 else ""))
PY
done
mark "SERVERS"; cat runs/_system/servers.json 2>/dev/null || ls "runs/$exp/_system" 2>/dev/null
archive="$(ls -t runs/_exports/*.tar.gz 2>/dev/null | head -n1)"
if [[ -n "$archive" ]]; then
  mark "ARCHIV $(basename "$archive") $(sha256sum "$archive" | cut -c1-64)"
  echo "-----BEGIN ARENA ARCHIVE-----"; base64 -w 0 "$archive"; echo; echo "-----END ARENA ARCHIVE-----"
fi
mark "FERTIG rc=$rc"
