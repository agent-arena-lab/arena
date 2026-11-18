#!/usr/bin/env bash
# Downloads GGUFs listed in models/*.env into $MODELS_DIR/<alias>/
# Usage: scripts/download_models.sh [model ...]
#   default: ARENA_MODELS from .env (scripts/models.sh choose), else the models of configs/experiment.yaml.
#   All catalog models (incl. 30B, ~19 GB each): scripts/download_models.sh --all
source "$(dirname "$0")/common.sh"

HF="$ROOT_DIR/.venv/bin/hf"
[[ -x "$HF" ]] || HF="$(command -v hf || command -v huggingface-cli || true)"
[[ -n "$HF" ]] || die "Hugging Face CLI not found (pip install 'huggingface_hub[cli]')"

models=("$@")
if [[ "${models[0]:-}" == "--all" ]]; then models=(); while IFS= read -r m; do models+=("$m"); done < <(all_models)
elif [[ ${#models[@]} -eq 0 && -n "${ARENA_MODELS:-}" ]]; then read -r -a models <<<"$ARENA_MODELS"
elif [[ ${#models[@]} -eq 0 ]]; then while IFS= read -r m; do models+=("$m"); done < <(config_models "$ROOT_DIR/configs/experiment.yaml"); fi
for m in "${models[@]}"; do
  load_model_env "$m"
  if [[ -n "$MODEL_PATH" ]]; then log "$m: present ($MODEL_PATH)"; continue; fi
  log "$m: downloading $HF_REPO ($HF_INCLUDE) ..."
  mkdir -p "$MODELS_DIR/$ALIAS"
  "$HF" download "$HF_REPO" --include "$HF_INCLUDE" --local-dir "$MODELS_DIR/$ALIAS"
  load_model_env "$m"
  [[ -n "$MODEL_PATH" ]] || die "$m: download finished but no file matches '$MODEL_GLOB' – check HF_INCLUDE/MODEL_GLOB"
  sha256 "$MODEL_PATH" | tee "$MODELS_DIR/$ALIAS/SHA256SUMS"
done
