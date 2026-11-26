#!/usr/bin/env bash
# Komplette Pipeline: Hardware erkennen -> llama-server starten -> Start-Check -> Spiele -> Manipulationscheck
# -> Server stoppen -> Auswertung -> Export. Fortsetzbar: fertige Spiele (result.json) werden übersprungen.
#
#   scripts/run_all.sh [config] [model ...]
#   scripts/run_all.sh configs/smoke.yaml                 Funktionstest
#   scripts/run_all.sh configs/pilot.yaml                 Pilot ohne Thinking-Limit (Verteilung messen)
#   scripts/run_all.sh configs/core.yaml                  Kerndesign, 160 Spiele
#   scripts/run_all.sh configs/extra.yaml                 feste Zusatzläufe, 40 Spiele
#   ARENA_LIMIT=3 scripts/run_all.sh configs/core.yaml    nur 3 Spiele (Laufzeit messen und hochrechnen)
#   ARENA_MANIPCHECK=0 ...                                ohne Manipulationscheck
#
# Gibt es weniger Server-Plätze (GPUs x MODELS_PER_GPU) als Modelle, laufen die Modelle in Gruppen nacheinander.
source "$(dirname "$0")/common.sh"

main() {
CONFIG="${1:-configs/core.yaml}"; shift || true
[[ -f "$ROOT_DIR/$CONFIG" ]] || die "config $CONFIG not found"
models=("$@")
if [[ ${#models[@]} -eq 0 && -n "${ARENA_MODELS:-}" ]]; then read -r -a models <<<"$ARENA_MODELS"; log "Modelle aus ARENA_MODELS: ${models[*]}"; fi
if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/$CONFIG"); fi
[[ ${#models[@]} -gt 0 ]] || die "keine Modelle in $CONFIG"
EXP_NAME="$(awk '/^name:/{print $2; exit}' "$ROOT_DIR/$CONFIG")"
SYS_DIR="$RUNS_DIR/$EXP_NAME/_system"
mkdir -p "$SYS_DIR"

detect_hardware
log "$(hardware_summary)"
export REASONING_BUDGET; REASONING_BUDGET="$(config_value "$ROOT_DIR/$CONFIG" thinking budget 16384)"
CTX_CFG="$(config_value "$ROOT_DIR/$CONFIG" thinking ctx_per_slot "$CTX_PER_SLOT")"
(( CTX_CFG > CTX_PER_SLOT )) && [[ "$HW_PROFILE" != rtx3090 ]] && export CTX_PER_SLOT="$CTX_CFG"
log "Thinking-Budget: $REASONING_BUDGET Tokens · Kontext pro Slot: $CTX_PER_SLOT"

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"; echo "host: $(hostname)"; uname -a
  hardware_summary
  nvidia-smi 2>/dev/null || true
  cat /etc/nv_tegra_release 2>/dev/null || true
  (nvpmodel -q 2>/dev/null || sudo -n nvpmodel -q 2>/dev/null) || true
  "$LLAMA_SERVER" --version 2>&1 | head -n 5 || true
  git -C "$LLAMA_DIR" rev-parse HEAD 2>/dev/null | sed 's/^/llama.cpp commit: /' || true
  free -h 2>/dev/null || true
} >"$SYS_DIR/system_$(date +%Y%m%d-%H%M%S).txt" 2>&1

trap '"$ROOT_DIR/scripts/serve.sh" stop' EXIT INT TERM
IFS=',' read -r -a gpus <<<"$GPU_IDS"
slots=$(( ${#gpus[@]} * MODELS_PER_GPU ))
limit_args=(); [[ -n "${ARENA_LIMIT:-}" ]] && limit_args=(--limit "$ARENA_LIMIT")

for (( start = 0; start < ${#models[@]}; start += slots )); do
  group=("${models[@]:start:slots}")
  log "===== Modelle: ${group[*]} ====="
  "$ROOT_DIR/scripts/serve.sh" start "${group[@]}"
  cp "$RUNS_DIR/_system/servers.json" "$SYS_DIR/servers_$(date +%Y%m%d-%H%M%S).json"
  harness benchmark --config "$CONFIG" --model "${group[@]}" || die "Start-Check fehlgeschlagen (runs/$EXP_NAME/benchmark.json)"
  cp "$RUNS_DIR/$EXP_NAME/benchmark.json" "$SYS_DIR/benchmark_$(date +%Y%m%d-%H%M%S).json"
  harness run --config "$CONFIG" --model "${group[@]}" ${limit_args[@]+"${limit_args[@]}"}
  if [[ "${ARENA_MANIPCHECK:-1}" == "1" ]] && [[ "$EXP_NAME" == core || "$EXP_NAME" == smoke ]]; then
    harness manipcheck --config "$CONFIG" --model "${group[@]}" >/dev/null && log "Manipulationscheck: runs/$EXP_NAME/_manipulation_check/"
  fi
  "$ROOT_DIR/scripts/serve.sh" stop
done

log "Auswertung ..."
harness analyze "runs/$EXP_NAME"
log "Export (Archiv + Prüfsummen) ..."
harness export "runs/$EXP_NAME"
log "fertig: runs/$EXP_NAME/{results.csv,results_tasks.csv,summary.md,benchmark.json}, Export in runs/_exports/"
}

main "$@"
