#!/usr/bin/env bash
#
# Download llama.cpp prebuilt binary and model GGUFs, pinned by version/revision for reproducibility.
#
# To bump a pin, delete the target first:
#   rm -rf local-llm/llama.cpp      # upgrade llama.cpp
#   rm local-llm/models/<file>      # re-download a model
#
# IMPORTANT: Use only the pinned versions/revisions for reproducibility.

set -euo pipefail

##############
# Destinations

LLAMA_DIR="local-llm/llama.cpp"
MODELS_DIR="local-llm/models"
mkdir -p "$LLAMA_DIR" "$MODELS_DIR"

######################################
# Pin versions/tags for reproducibility
# WARNING: Manually delete the destinations above and re-run this script to bump versions.

# Version releases (vX.Y.Z) carry no binaries. Their release notes name the nightly
# build that holds them, under "Assets". Bump both together.
# Releases at https://github.com/ggml-org/llama.cpp/releases
LLAMA_VERSION="v0.6.0"
LLAMA_BUILD="b11429"

# Relative paths to the GGUF files in the Hugging Face repos, pinned by revision SHA.
# Format: "hf_repo|revision_sha|filename"
# Revision SHA from https://huggingface.co/<repo>/commits/main
MODELS=(
  # Gemma 4 E4B (Google). 4.5B active via Per-Layer Embeddings, 128K context.
  "bartowski/google_gemma-4-E4B-it-GGUF|c04cb322fd63e347db759a08b6249b867488ccf8|google_gemma-4-E4B-it-Q4_K_M.gguf"

  # Qwen3.5-4B (Alibaba). 4B parameters, 262K native context.
  "bartowski/Qwen_Qwen3.5-4B-GGUF|4168f45a16a1290d65a4ec0fa312ae917a4c15d6|Qwen_Qwen3.5-4B-Q4_K_M.gguf"

  # Llama 3.2 3B (Meta). 3B parameters, 128K context.
  "bartowski/Llama-3.2-3B-Instruct-GGUF|5ab33fa94d1d04e903623ae72c95d1696f09f9e8|Llama-3.2-3B-Instruct-Q4_K_M.gguf"
)

##########
# Download

# llama.cpp
if [[ ! -x "$LLAMA_DIR/llama-server" ]]; then
  archive="llama-${LLAMA_BUILD}-bin-macos-arm64.tar.gz"
  echo "==> Downloading llama.cpp ${LLAMA_VERSION} (build ${LLAMA_BUILD})"
  curl -fL --progress-bar -o "/tmp/$archive" \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${archive}"
  # The tarball nests everything inside llama-${LLAMA_BUILD}/. Strip that so the
  # binaries go directly into $LLAMA_DIR, where the check above looks for them.
  tar -xzf "/tmp/$archive" -C "$LLAMA_DIR" --strip-components=1
  rm "/tmp/$archive"
  xattr -dr com.apple.quarantine "$LLAMA_DIR" 2>/dev/null || true
fi

# Models
for entry in "${MODELS[@]}"; do
  IFS='|' read -r repo revision filename <<<"$entry"
  target="$MODELS_DIR/$filename"
  [[ -f "$target" ]] && continue

  echo "==> Downloading ${filename} from ${repo}@${revision}"
  curl -fL --progress-bar -o "${target}.partial" \
    "https://huggingface.co/${repo}/resolve/${revision}/${filename}"
  mv "${target}.partial" "$target"
done

echo "Done."
