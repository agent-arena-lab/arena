#!/usr/bin/env bash
# Modelle als fertige GGUFs bereitstellen -> $MODELS_DIR/<alias>/*.Q4_K_M.gguf (+ SHA256SUMS, SOURCE.json)
#
#   scripts/prepare_models.sh [model ...]      Standard: Modelle aus configs/core.yaml
#
# Quelle: GGUF_REPO/GGUF_INCLUDE aus models/<alias>.env (mradermacher, statisch, beide Varianten je Familie vom
# selben Quantisierer). Selbst-Quantisierung gibt es nicht mehr. Vorhandene Dateien (z. B. auf dem Lium-Volume)
# werden nicht neu geladen, ihre SHA-256 wird nur einmal berechnet. Danach Paar-Prüfung über die GGUF-Metadaten.
# Pro Modell werden Dauer, Größe und MB/s ausgegeben und in runs/_system/timings.tsv festgehalten.
source "$(dirname "$0")/common.sh"

main() {
HF="$ROOT_DIR/.venv/bin/hf"
[[ -x "$HF" ]] || HF="$(command -v hf || true)"
[[ -n "$HF" ]] || die "Hugging Face CLI not found (scripts/setup_gpu.sh)"
PY="$ROOT_DIR/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
"$PY" -c 'import hf_transfer' 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0

models=("$@")
if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/configs/core.yaml"); fi
t_all=$(date +%s)

for m in "${models[@]}"; do
  load_model_env "$m"
  out_dir="$MODELS_DIR/$ALIAS"
  mkdir -p "$out_dir"
  if [[ -n "$MODEL_PATH" && -f "$out_dir/SHA256SUMS" ]]; then
    log "$m: vorhanden ($(basename "$MODEL_PATH"), $(du -BM "$MODEL_PATH" | cut -f1)) – kein Download"
    continue
  fi
  if [[ -z "$MODEL_PATH" ]]; then
    t0=$(date +%s)
    log "$m: lade $GGUF_REPO ($GGUF_INCLUDE) ..."
    retry 3 "$HF" download "$GGUF_REPO" --include "$GGUF_INCLUDE" --local-dir "$out_dir" >/dev/null \
      || die "$m: Download fehlgeschlagen"
    load_model_env "$m"
    [[ -n "$MODEL_PATH" ]] || die "$m: kein *${QUANT_TYPE}*.gguf nach dem Download – GGUF_INCLUDE prüfen"
    dt=$(( $(date +%s) - t0 )); (( dt > 0 )) || dt=1
    mb=$(( $(stat -c %s "$MODEL_PATH") / 1024 / 1024 ))
    log "$m: Download $mb MB in $dt s = $(( mb / dt )) MB/s"
    printf '%s\t%s\t%s\t0\t%s MB, %s MB/s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "download: $m" "$dt" "$mb" "$(( mb / dt ))" >>"$TIMINGS"
  fi
  t0=$(date +%s)
  (cd "$(dirname "$MODEL_PATH")" && sha256 "$(basename "$MODEL_PATH")") >"$out_dir/SHA256SUMS"
  log "$m: SHA-256 $(cut -c1-16 "$out_dir/SHA256SUMS")… ($(( $(date +%s) - t0 )) s)"
  "$PY" - "$out_dir/SOURCE.json" "$ALIAS" "prebuilt:$GGUF_REPO" "$QUANT_TYPE" "$(basename "$MODEL_PATH")" <<'PY'
import json, sys, datetime
path, alias, src, quant, gguf = sys.argv[1:6]
json.dump({"alias": alias, "source": src, "quant": quant, "gguf": gguf,
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
          open(path, "w"), indent=2)
PY
done
log "Modelle bereit nach $(( $(date +%s) - t_all )) s. Paar-Prüfung (GGUF-Metadaten: Format, Quant-Version, Kalibrierdaten) ..."
"$PY" "$ROOT_DIR/scripts/check_gguf_pairs.py" "$MODELS_DIR" --out "$RUNS_DIR/_system/gguf_pairs.json" \
  || die "Original und abliterierte Variante sind unterschiedlich quantisiert (runs/_system/gguf_pairs.json)"
}

main "$@"
