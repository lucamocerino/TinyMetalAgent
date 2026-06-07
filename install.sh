#!/usr/bin/env bash
set -euo pipefail

WITH_MODEL=0
DRY_RUN=0
SKIP_TESTS=0
SKIP_PIP_INSTALL=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Build and install TinyMetalAgent locally.

Default setup:
  - builds the C/Metal TinyEngine runtime
  - runs C tests
  - compiles Python packages
  - creates a local .venv virtual environment
  - installs the Python package in editable mode inside .venv
  - verifies the tinyagent CLI starts

Options:
  --with-model             Also download the default Qwen2.5-Coder 3B Q4_0 GGUF.
                           This is intentionally opt-in because model files are large
                           and governed by upstream model terms.
  --skip-tests             Build/install without running C tests.
  --skip-pip-install       Do not create .venv or install the editable package.
  --dry-run                Print commands without executing them.
  --help                   Show this help.

Examples:
  ./install.sh
  ./install.sh --with-model
  ./install.sh --dry-run --with-model
EOF
}

log() {
  printf '==> %s\n' "$*"
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
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

need_cmd() {
  if [ "$DRY_RUN" -eq 0 ] && ! command -v "$1" >/dev/null 2>&1; then
    die "required command not found: $1"
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-model)
      WITH_MODEL=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --skip-pip-install)
      SKIP_PIP_INSTALL=1
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
      die "unknown option: $1"
      ;;
  esac
done

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT_DIR"

need_cmd make
need_cmd python3

if [ "$DRY_RUN" -eq 0 ] && [ "$(uname -s)" != "Darwin" ]; then
  die "TinyEngine currently supports macOS/Darwin with Metal only"
fi

log "Building TinyEngine C/Metal runtime"
run make -C c clean all

if [ "$SKIP_TESTS" -eq 0 ]; then
  log "Running C tests"
  run make -C c test
fi

log "Checking Python sources"
run python3 -m compileall -q python

if [ "$SKIP_PIP_INSTALL" -eq 0 ]; then
  log "Creating local Python virtual environment"
  run python3 -m venv .venv
  log "Updating pip and setuptools inside .venv"
  run .venv/bin/python -m pip install --upgrade pip setuptools
  log "Installing Python package in editable mode inside .venv"
  run .venv/bin/python -m pip install -e .
  PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_CMD="python3"
fi

log "Checking tinyagent CLI"
run env TINYENGINE_LIBRARY="$ROOT_DIR/c/build/libtinyengine.dylib" "$PYTHON_CMD" -m tinyagent --help

if [ "$WITH_MODEL" -eq 1 ]; then
  log "Preparing default local Qwen model"
  run scripts/prepare_qwen_model.sh --target coder
else
  log "Skipping model download. Use ./install.sh --with-model to fetch the default GGUF."
fi

cat <<EOF

TinyMetalAgent local setup complete.

Next steps:
  export TINYENGINE_LIBRARY="$ROOT_DIR/c/build/libtinyengine.dylib"
  source "$ROOT_DIR/.venv/bin/activate"
  bin/tinyagent --ask "hello"

If you did not use --with-model, place qwen2.5-coder-3b-instruct-q4_0-te.gguf in
models/ or ../models/, or pass --model /path/to/model.gguf.
EOF
