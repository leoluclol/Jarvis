#!/usr/bin/env bash
# Installs the build/runtime dependencies for the C++ Jarvis port.
#
# What it does:
#   1. apt-installs the system packages (cmake, g++, PortAudio, libcurl, nlohmann-json)
#   2. Fetches libtensorflowlite_c.so from the pymicro_wakeword wheel
#      (needed at runtime by the wake-word engine)
#   3. Downloads the official ONNX Runtime tarball into third_party/
#      (needed to build the Silero VAD)
#
# ONNX Runtime only ships prebuilt x86_64 and aarch64 builds. On a 32-bit ARM
# device (e.g. Raspberry Pi 2) you must build it from source and point CMake at
# it with -DONNXRUNTIME_ROOT=<prefix>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

info() { printf '%s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
if command -v apt-get >/dev/null 2>&1; then
  info "Installing system packages (may prompt for sudo)..."
  sudo apt-get update
  sudo apt-get install -y cmake g++ pkg-config \
    libportaudio2 portaudio19-dev \
    libcurl4-openssl-dev \
    nlohmann-json3-dev
fi

# ---------------------------------------------------------------------------
# 2. TFLite C runtime (wake-word inference)
# ---------------------------------------------------------------------------
TFLITE_LIB_DIR="${CPP_DIR}/third_party/libtensorflowlite_c"
TFLITE_LIB="${TFLITE_LIB_DIR}/libtensorflowlite_c.so"
mkdir -p "${TFLITE_LIB_DIR}"

if [ -s "${TFLITE_LIB}" ]; then
  info "libtensorflowlite_c.so already present."
else
  if command -v python3 >/dev/null 2>&1 && python3 -m pip --version >/dev/null 2>&1; then
    TMPDIR="$(mktemp -d)"
    info "Downloading libtensorflowlite_c.so from the pymicro_wakeword wheel..."
    python3 -m pip download pymicro_wakeword --no-deps --only-binary :all: \
      -d "${TMPDIR}" >/dev/null 2>&1 || true
    WHEEL="$(ls "${TMPDIR}"/*.whl 2>/dev/null | head -1 || true)"
    if [ -n "${WHEEL}" ]; then
      unzip -o -q "${WHEEL}" -d "${TMPDIR}/wheel"
      LIB="$(find "${TMPDIR}/wheel" -name 'libtensorflowlite_c*.so' | head -1 || true)"
      if [ -n "${LIB}" ]; then
        cp "${LIB}" "${TFLITE_LIB}"
      fi
    fi
    rm -rf "${TMPDIR}"
  fi

  if [ -s "${TFLITE_LIB}" ]; then
    info "libtensorflowlite_c.so installed under third_party/libtensorflowlite_c."
  else
    info "WARNING: could not fetch libtensorflowlite_c.so automatically."
    info "Copy it from your pymicro_wakeword install or set TFLITE_LIB_PATH."
  fi
fi

# ---------------------------------------------------------------------------
# 3. ONNX Runtime (Silero VAD)
# ---------------------------------------------------------------------------
ONNX_VER="${ONNX_VER:-1.31.0}"
ONNX_ROOT="${CPP_DIR}/third_party/onnxruntime"
mkdir -p "${ONNX_ROOT}"

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64)  ONNX_PLAT="linux-x64" ;;
  aarch64|arm64) ONNX_PLAT="linux-aarch64" ;;
  *)       ONNX_PLAT="" ;;
esac

if [ -s "${ONNX_ROOT}/include/onnxruntime_cxx_api.h" ]; then
  info "ONNX Runtime already present in third_party/onnxruntime."
elif [ -z "${ONNX_PLAT}" ]; then
  info "No official ONNX Runtime build for arch '${ARCH}'."
  info "Build ONNX Runtime from source, then configure CMake with:"
  info "  -DONNXRUNTIME_ROOT=<onnxruntime-prefix>"
else
  URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VER}/onnxruntime-${ONNX_PLAT}-${ONNX_VER}.tgz"
  info "Downloading ONNX Runtime ${ONNX_VER} (${ONNX_PLAT})..."
  curl -L -o /tmp/onnxruntime-${ONNX_VER}.tgz "${URL}"
  tar -xzf "/tmp/onnxruntime-${ONNX_VER}.tgz" -C "${ONNX_ROOT}" --strip-components=1
  rm -f "/tmp/onnxruntime-${ONNX_VER}.tgz"
  info "ONNX Runtime installed under third_party/onnxruntime."
fi

# ---------------------------------------------------------------------------
info ""
info "Done. Build with:"
info "  mkdir -p ${CPP_DIR}/build && cd ${CPP_DIR}/build"
info "  cmake .. && make"
info ""
info "The main binary also needs an .env file at the repo root with OPENAI_API_KEY."
