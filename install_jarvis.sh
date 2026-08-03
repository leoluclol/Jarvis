#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/jarvis-venv"

info() {
  printf '%s\n' "$*"
}

confirm() {
  local prompt_text="$1"
  local answer

  read -r -p "${prompt_text} [y/N] " answer
  case "${answer}" in
    y|Y|yes|YES)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

run_cmd() {
  info "--> $*"
  "$@"
}

info "Jarvis installer"
info "Repository: ${ROOT_DIR}"

if confirm "Run the system package setup for PortAudio and PyAudio? This uses sudo and may change system packages."; then
  run_cmd sudo apt update
  run_cmd sudo apt install -y portaudio19-dev python3-pyaudio
else
  info "Skipping system package setup. Run these manually if needed:"
  info "  sudo apt update && sudo apt install -y portaudio19-dev python3-pyaudio"
fi

if confirm "Create the virtual environment in ${VENV_DIR}?"; then
  run_cmd python3 -m venv --system-site-packages "${VENV_DIR}"
else
  info "Skipping virtual environment creation. Create it manually before continuing."
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

run_cmd python -m pip install --upgrade pip
run_cmd python -m pip install openwakeword==0.6.0 --no-deps
run_cmd python -m pip install scikit-learn onnxruntime requests scipy tqdm
run_cmd python -m pip install -r "${ROOT_DIR}/requirements.txt"

if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  if confirm "Create or overwrite .env with the OPENAI_API_KEY from the current environment?"; then
    printf 'OPENAI_API_KEY=%s\n' "${OPENAI_API_KEY}" > "${ROOT_DIR}/.env"
    info "Wrote ${ROOT_DIR}/.env"
  fi
else
  info "OPENAI_API_KEY is not set in the environment. Create ${ROOT_DIR}/.env manually before running Jarvis."
fi

info "Done. Activate the environment with: source ${VENV_DIR}/bin/activate"
info "Then run: python ${ROOT_DIR}/jarvis.py"