#!/usr/bin/env bash
set -euo pipefail

TARGET="coder"
QUANT="Q4_0"
MODE="download"
OUT_DIR="models"
LLAMA_CPP=""
FP16_PATH=""
OUTPUT_NAME=""
DRY_RUN=0
FORCE=0
HF_BASE_URL="${HF_BASE_URL:-https://huggingface.co}"

usage() {
  cat <<'EOF'
Usage: scripts/prepare_qwen_model.sh [options]

Download or locally quantize the Qwen GGUF models used by TinyEngine/TinyAgent.

Defaults prepare TinyAgent's preferred local model:
  models/qwen2.5-coder-3b-instruct-q4_0-te.gguf

Options:
  --target coder             Model target. coder = Qwen2.5-Coder 3B Instruct.
                             Default: coder.
  --quant QUANT              Quantization to prepare. Default: Q4_0.
                             Common values: Q4_0, Q4_K_M, Q8_0, FP16.
  --mode download|quantize   download = fetch official pre-quantized GGUF.
                             quantize = fetch/use FP16 GGUF and run llama.cpp.
                             Default: download.
  --out-dir DIR              Destination directory. Default: models.
  --output-name NAME         Override output GGUF filename.
  --llama-cpp DIR            llama.cpp checkout/build directory for --mode quantize.
  --fp16 PATH                Existing FP16 GGUF to quantize instead of downloading it.
  --force                    Recreate output if it already exists.
  --dry-run                  Print actions without downloading or quantizing.
  --help                     Show this help.

Examples:
  scripts/prepare_qwen_model.sh
  scripts/prepare_qwen_model.sh --target coder --quant Q8_0
  scripts/prepare_qwen_model.sh --mode quantize --target coder --llama-cpp ../tools/llama.cpp --quant Q4_0

Notes:
  - Model files are not committed; .gitignore excludes *.gguf.
  - Review the upstream Qwen model card and license before downloading.
  - Local quantization requires llama.cpp's llama-quantize binary.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

log() {
  echo "==> $*"
}

run() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '+'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

lowercase() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

quant_slug() {
  case "$1" in
    fp16|FP16|F16|f16) printf 'fp16' ;;
    *) lowercase "$1" ;;
  esac
}

target_metadata() {
  case "$TARGET" in
    coder)
      REPO_ID="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
      MODEL_STEM="qwen2.5-coder-3b-instruct"
      DEFAULT_SUFFIX="-te"
      ;;
    *)
      die "unknown --target '$TARGET' (expected coder)"
      ;;
  esac
}

source_filename() {
  local slug
  slug="$(quant_slug "$1")"
  printf '%s-%s.gguf' "$MODEL_STEM" "$slug"
}

default_output_name() {
  local slug
  slug="$(quant_slug "$QUANT")"
  if [ "$TARGET" = "coder" ] && [ "$slug" = "q4_0" ]; then
    printf '%s-%s%s.gguf' "$MODEL_STEM" "$slug" "$DEFAULT_SUFFIX"
  else
    printf '%s-%s.gguf' "$MODEL_STEM" "$slug"
  fi
}

download_url() {
  local file
  file="$1"
  printf '%s/%s/resolve/main/%s?download=true' "$HF_BASE_URL" "$REPO_ID" "$file"
}

ensure_parent_dir() {
  local path
  path="$1"
  run mkdir -p "$(dirname "$path")"
}

download_file() {
  local url dest
  url="$1"
  dest="$2"

  if [ -f "$dest" ] && [ "$FORCE" -eq 0 ]; then
    log "Using existing file: $dest"
    return 0
  fi

  command -v curl >/dev/null 2>&1 || die "curl is required to download models"
  ensure_parent_dir "$dest"
  if [ -f "$dest" ] && [ "$FORCE" -eq 1 ]; then
    run rm -f "$dest"
  fi
  log "Downloading $url"
  run curl -L --fail --continue-at - --output "$dest" "$url"
}

find_llama_quantize() {
  if [ -n "$LLAMA_CPP" ]; then
    for candidate in \
      "$LLAMA_CPP/build/bin/llama-quantize" \
      "$LLAMA_CPP/build/bin/quantize" \
      "$LLAMA_CPP/llama-quantize" \
      "$LLAMA_CPP/quantize"
    do
      if [ -x "$candidate" ]; then
        printf '%s' "$candidate"
        return 0
      fi
    done
  fi

  if command -v llama-quantize >/dev/null 2>&1; then
    command -v llama-quantize
    return 0
  fi

  return 1
}

checksum_hint() {
  local path
  path="$1"
  if [ "$DRY_RUN" -eq 1 ]; then
    log "Checksum after download: shasum -a 256 '$path'"
  elif [ -f "$path" ]; then
    log "SHA-256:"
    shasum -a 256 "$path"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --target)
      [ "$#" -ge 2 ] || die "--target requires a value"
      TARGET="$2"
      shift 2
      ;;
    --quant)
      [ "$#" -ge 2 ] || die "--quant requires a value"
      QUANT="$2"
      shift 2
      ;;
    --mode)
      [ "$#" -ge 2 ] || die "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --out-dir)
      [ "$#" -ge 2 ] || die "--out-dir requires a value"
      OUT_DIR="$2"
      shift 2
      ;;
    --output-name)
      [ "$#" -ge 2 ] || die "--output-name requires a value"
      OUTPUT_NAME="$2"
      shift 2
      ;;
    --llama-cpp)
      [ "$#" -ge 2 ] || die "--llama-cpp requires a value"
      LLAMA_CPP="$2"
      shift 2
      ;;
    --fp16)
      [ "$#" -ge 2 ] || die "--fp16 requires a value"
      FP16_PATH="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown option '$1'"
      ;;
  esac
done

case "$MODE" in
  download|quantize) ;;
  *) die "unknown --mode '$MODE' (expected download or quantize)" ;;
esac

target_metadata
QUANT_SLUG="$(quant_slug "$QUANT")"
OUTPUT_NAME="${OUTPUT_NAME:-$(default_output_name)}"
OUT_PATH="$OUT_DIR/$OUTPUT_NAME"

log "Target: $TARGET ($REPO_ID)"
log "Output: $OUT_PATH"
log "Review upstream model terms before use: https://huggingface.co/$REPO_ID"

if [ "$MODE" = "download" ]; then
  SRC_FILE="$(source_filename "$QUANT")"
  URL="$(download_url "$SRC_FILE")"
  download_file "$URL" "$OUT_PATH"
  checksum_hint "$OUT_PATH"
  log "Ready. Use: TINYAGENT_MODEL='$OUT_PATH' bin/tinyagent --ask 'hello'"
  exit 0
fi

if [ "$QUANT_SLUG" = "fp16" ]; then
  die "--mode quantize requires a non-FP16 --quant value"
fi

if [ -f "$OUT_PATH" ] && [ "$FORCE" -eq 0 ]; then
  die "output already exists: $OUT_PATH (pass --force to replace)"
fi
if [ -f "$OUT_PATH" ] && [ "$FORCE" -eq 1 ]; then
  run rm -f "$OUT_PATH"
fi

if [ -n "$FP16_PATH" ]; then
  FP16_SOURCE="$FP16_PATH"
else
  FP16_SOURCE="$OUT_DIR/$(source_filename FP16)"
  FP16_URL="$(download_url "$(source_filename FP16)")"
  download_file "$FP16_URL" "$FP16_SOURCE"
fi

if [ "$DRY_RUN" -eq 1 ]; then
  if [ -n "$LLAMA_CPP" ]; then
    QUANTIZE_BIN="$LLAMA_CPP/build/bin/llama-quantize"
  else
    QUANTIZE_BIN="llama-quantize"
  fi
else
  [ -f "$FP16_SOURCE" ] || die "FP16 source does not exist: $FP16_SOURCE"
  QUANTIZE_BIN="$(find_llama_quantize)" || die "could not find llama-quantize; pass --llama-cpp /path/to/llama.cpp or add llama-quantize to PATH"
fi

ensure_parent_dir "$OUT_PATH"
log "Quantizing $FP16_SOURCE -> $OUT_PATH ($QUANT)"
run "$QUANTIZE_BIN" "$FP16_SOURCE" "$OUT_PATH" "$QUANT"
checksum_hint "$OUT_PATH"
log "Ready. Use: TINYAGENT_MODEL='$OUT_PATH' bin/tinyagent --ask 'hello'"
