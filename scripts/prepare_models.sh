#!/usr/bin/env bash
# Modelle als GGUF bereitstellen -> $MODELS_DIR/<alias>/<alias>-Q4_K_M.gguf (+ SHA256SUMS, SOURCE.json)
#
#   scripts/prepare_models.sh [model ...]                 Standard: ARENA_MODELS aus .env, sonst Modelle aus configs/core.yaml
#   ARENA_QUANT=prebuilt scripts/prepare_models.sh ...    fertige GGUFs laden (schnell, für Pilot/Mac/Jetson)
#
# ARENA_QUANT=self (Standard, Regel für saubere Paare): Originalgewichte (safetensors) laden und Original wie
# abliterierte Variante mit IDENTISCHEM Befehl selbst quantisieren:
#   convert_hf_to_gguf.py --outtype bf16  ->  llama-quantize <bf16.gguf> <out.gguf> Q4_K_M
# Die Quantisierung ist damit als Störfaktor ausgeschlossen. Platzbedarf pro Modell kurzzeitig ca. 2 x SRC_GB.
# ARENA_KEEP_SRC=1 behält die Originalgewichte (Standard: nach erfolgreicher Quantisierung löschen).
# Gemma: Lizenz auf Hugging Face akzeptieren und HF_TOKEN setzen.
source "$(dirname "$0")/common.sh"

main() {
HF="$ROOT_DIR/.venv/bin/hf"
[[ -x "$HF" ]] || HF="$(command -v hf || command -v huggingface-cli || true)"
[[ -n "$HF" ]] || die "Hugging Face CLI not found (pip install 'huggingface_hub[cli]')"
PY="$ROOT_DIR/.venv/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"

models=("$@")
if [[ ${#models[@]} -eq 0 && -n "${ARENA_MODELS:-}" ]]; then read -r -a models <<<"$ARENA_MODELS"; fi
if [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/configs/core.yaml"); fi
LLAMA_COMMIT="$(git -C "$LLAMA_DIR" rev-parse --short HEAD 2>/dev/null || "$LLAMA_SERVER" --version 2>&1 | grep -oE 'version: [0-9]+ \([0-9a-f]+\)' | head -n1 || echo unknown)"

for m in "${models[@]}"; do
  load_model_env "$m"
  out_dir="$MODELS_DIR/$ALIAS"
  if [[ -n "$MODEL_PATH" ]]; then log "$m: vorhanden ($MODEL_PATH)"; continue; fi
  mkdir -p "$out_dir"
  if [[ "$ARENA_QUANT" == "prebuilt" ]]; then
    log "$m: lade fertiges GGUF $GGUF_REPO ($GGUF_INCLUDE) – Achtung: Quantisierer der Paare kann sich unterscheiden"
    "$HF" download "$GGUF_REPO" --include "$GGUF_INCLUDE" --local-dir "$out_dir"
    load_model_env "$m"
    [[ -n "$MODEL_PATH" ]] || die "$m: kein *${QUANT_TYPE}*.gguf nach dem Download – GGUF_INCLUDE prüfen"
    src="prebuilt:$GGUF_REPO"; cmd="hf download $GGUF_REPO --include '$GGUF_INCLUDE'"
  else
    [[ -x "$LLAMA_QUANTIZE" ]] || die "llama-quantize fehlt ($LLAMA_QUANTIZE) – scripts/setup_gpu.sh ausführen"
    [[ -f "$LLAMA_DIR/convert_hf_to_gguf.py" ]] || die "$LLAMA_DIR/convert_hf_to_gguf.py fehlt (llama.cpp-Checkout nötig)"
    src_dir="$MODELS_DIR/_src/$ALIAS"
    log "$m: lade Originalgewichte $SRC_REPO (~${SRC_GB:-?} GB) ..."
    "$HF" download "$SRC_REPO" --local-dir "$src_dir" \
      --include "*.safetensors" "*.json" "tokenizer*" "*.model" "*.tiktoken" "*.jinja" "*.txt"
    rev="$(cat "$src_dir/.cache/huggingface/download/"*.metadata 2>/dev/null | head -n1 || true)"
    bf16="$out_dir/$ALIAS-bf16.gguf"
    target="$out_dir/$ALIAS-$QUANT_TYPE.gguf"
    log "$m: konvertiere nach GGUF (bf16) ..."
    "$PY" "$LLAMA_DIR/convert_hf_to_gguf.py" "$src_dir" --outtype bf16 --outfile "$bf16"
    log "$m: quantisiere $QUANT_TYPE ..."
    "$LLAMA_QUANTIZE" "$bf16" "$target" "$QUANT_TYPE"
    rm -f "$bf16"
    [[ "${ARENA_KEEP_SRC:-0}" == "1" ]] || rm -rf "$src_dir"
    load_model_env "$m"
    src="self:$SRC_REPO${rev:+@$rev}"
    cmd="convert_hf_to_gguf.py --outtype bf16 && llama-quantize <bf16> <out> $QUANT_TYPE"
  fi
  log "$m: SHA-256 ..."
  (cd "$(dirname "$MODEL_PATH")" && sha256 "$(basename "$MODEL_PATH")") | tee "$out_dir/SHA256SUMS"
  "$PY" - "$out_dir/SOURCE.json" "$ALIAS" "$src" "$QUANT_TYPE" "$LLAMA_COMMIT" "$cmd" "$(basename "$MODEL_PATH")" <<'PY'
import json, sys, datetime
path, alias, src, quant, llama, cmd, gguf = sys.argv[1:8]
json.dump({"alias": alias, "source": src, "quant": quant, "llama_cpp": llama, "command": cmd, "gguf": gguf,
           "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")},
          open(path, "w"), indent=2)
PY
done
log "Paar-Prüfung (GGUF-Metadaten: Format, Quant-Version, Kalibrierdaten) ..."
"$PY" "$ROOT_DIR/scripts/check_gguf_pairs.py" "$MODELS_DIR" --out "$RUNS_DIR/_system/gguf_pairs.json" \
  || die "Original und abliterierte Variante sind unterschiedlich quantisiert (runs/_system/gguf_pairs.json)"
}

main "$@"
