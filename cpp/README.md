# Jarvis — C++ port

A faithful C++17 rewrite of the Python voice assistant. It mirrors the OOP
structure of `src2/` (`AssistantOrchestrator` + controllers), keeps the same
runtime configuration, and produces byte-for-byte identical wake-word and VAD
results (verified by the test suite against the original Python pipeline).

## Feature parity

| Python (`src2/`)               | C++ (`cpp/`)                       |
|--------------------------------|------------------------------------|
| `main.py` (`Config`)           | `include/jarvis/config.h`          |
| `assistant_orchestrator.py`    | `include/jarvis/assistant_orchestrator.h` |
| `audio_controller.py`          | `include/jarvis/audio_controller.h` (PortAudio) |
| `wakeword_engine.py`           | `include/jarvis/wakeword_engine.h` (TFLite) |
| `vad.py`                       | `include/jarvis/vad.h` (ONNX Runtime) |
| `openai_client.py`             | `include/jarvis/openai_client.h` (libcurl) |
| `conversation_manager.py`      | `include/jarvis/conversation_manager.h` |
| `hardware_controller.py`       | `include/jarvis/hardware_controller.h` (sysfs GPIO) |
| `.env`                         | `config.cpp` (same `.env` at repo root) |

Everything else (`wav.h`, `thread_queue.h`, `signals.h`, `logger.h`, `env.h`)
is supporting glue with no Python equivalent.

## How the wake-word engine works

The original Python uses `pymicro_wakeword` (a TFLite model + the TFLite Micro
audio frontend). To stay identical, this port:

1. **Vendors the TFLite Micro audio frontend** (`third_party/microfrontend`,
   Apache-2.0, same sources/flags as upstream `pymicro_features`) for the
   40-band mel filterbank features.
2. **Loads `libtensorflowlite_c.so` at runtime** with `dlopen` — the same
   shared library the Python ctypes binding uses — and drives the quantized
   `hey_jarvis.tflite` model directly.

Because the library is `dlopen`ed rather than linked, the same binary runs on
x86_64 and ARM by pointing `TFLITE_LIB_PATH` at the right library. The library
is found in this order:

1. `$TFLITE_LIB_PATH`
2. `cpp/third_party/libtensorflowlite_c/libtensorflowlite_c.so`
3. `/usr/local/lib/libtensorflowlite_c.so`, `/usr/lib/...`

`cpp/scripts/setup_deps.sh` downloads the library for the current platform.

## Dependencies

- C++17 compiler, CMake ≥ 3.16, pkg-config
- PortAudio (`libportaudio2`, `portaudio19-dev`)
- libcurl (`libcurl4-openssl-dev`)
- nlohmann-json (`nlohmann-json3-dev`)
- ONNX Runtime (Silero VAD) — headers + `libonnxruntime.so`
- TensorFlow Lite C API (`libtensorflowlite_c.so`) — runtime only
- Linux + a working mic/speakers; Raspberry Pi GPIO for the LEDs

### ONNX Runtime on 32-bit ARM

Microsoft only publishes prebuilt ONNX Runtime for x86_64 and aarch64. On a
Raspberry Pi 2 (armv7l) you must build ONNX Runtime from source and pass
`-DONNXRUNTIME_ROOT=<prefix>` to CMake. This mirrors the Python version, which
has the same constraint on 32-bit ARM.

## Build

```bash
# One-time dependency setup (apt packages + TFLite lib + ONNX Runtime)
./cpp/scripts/setup_deps.sh

# Build
mkdir -p cpp/build && cd cpp/build
cmake .. && make -j$(nproc)

# Run the tests
ctest --output-on-failure

# Run the assistant (needs .env at the repo root with OPENAI_API_KEY)
./jarvis
```

## Layout

```
cpp/
├── CMakeLists.txt
├── include/jarvis/        # public headers
├── src/                   # implementation + main.cpp
├── models/                # hey_jarvis.tflite + hey_jarvis.json (wake word)
├── third_party/
│   ├── microfrontend/     # vendored TFLite Micro audio frontend (Apache-2.0)
│   └── libtensorflowlite_c/  # downloaded at setup time (gitignored)
├── tests/                 # unit/integration tests + golden reference data
├── scripts/setup_deps.sh
└── .env.example
```

## Configuration

All runtime settings are the same as the Python version and are defined in
`include/jarvis/config.h` (paths are resolved from the repo root at build
time). `OPENAI_API_KEY` is read from `.env` at the repo root, or from the
environment.

- Wake word: `Hey Jarvis` (hey_jarvis.tflite), activation threshold 0.75
- Barge-in threshold: 0.6, 1.0 s grace period
- Silero VAD threshold: 0.5, end-of-speech silence: 1.5 s
- Chat model `gpt-4o-mini`, TTS `gpt-4o-mini-tts` (onyx, 1.3x, 24 kHz PCM)
- LEDs: BCM 5 (thinking), BCM 13 (listening)
