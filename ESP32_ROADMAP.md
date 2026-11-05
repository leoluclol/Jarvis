# ESP32-S3 Roadmap — Jarvis

Goal: run the Jarvis voice assistant on an ESP32-S3 (always-on ears + wake word
+ VAD on the chip; Whisper/LLM/TTS stay off-chip). This roadmap is written
against the current state of the repo (Python `src/` + `src2/`, and the new C++
port in `cpp/`).

---

## 0. Current state — what already carries over

The C++ port in `cpp/` is split into a *logic layer* and a *backend layer*.
For the ESP32, the logic layer transfers almost unchanged; the backend layer is
replaced by ESP-IDF components.

**Transfers as-is (logic):**
- `MicroFrontend` — the vendored TFLite Micro audio frontend
  (`cpp/third_party/microfrontend`) is plain fixed-point C that compiles on
  the ESP32 unchanged. This is the hardest part of the wake-word pipeline and
  it's already done and verified (bit-for-bit vs. Python).
- `hey_jarvis.tflite` + `hey_jarvis.json` (`cpp/models/`) — designed for
  microcontrollers (microWakeWord targets ESP32; tensor arena ≈ 23 KB).
- Wake-word sliding-window / stride / quantization logic (`wakeword_engine.cpp`).
- VAD state machine (state/context threading, pre-roll, end-of-speech timing)
  (`vad.cpp`).
- `conversation_manager`, orchestrator flow, stop-command / hallucination
  guards, WAV writer (`wav.cpp`).

**Must be replaced (backend):**

| Current (`cpp/`) | ESP32-S3 replacement |
|---|---|
| PortAudio | ESP-IDF I2S driver + DMA ring buffers |
| libcurl | `esp_http_client` + mbedTLS (bundled) |
| ONNX Runtime (VAD) | TensorFlow Lite for Microcontrollers (TFLM) + converted Silero `.tflite` |
| `dlopen` TFLite C API | static linking of TFLM (no dynamic loading on MCU) |
| `std::thread` / mutex / `ThreadQueue` | FreeRTOS tasks / kernel queues |
| `.env` + filesystem paths | compile-time `sdkconfig` + NVS |
| GPIO sysfs | `gpio_hal` / `ledc` driver |

---

## 1. Key architecture decisions (read first — everything else follows)

### D1 — Use ESP-IDF, not Arduino
- **Choice:** ESP-IDF v5.x.
- **Why:** glitch-free audio and the decoupled pipeline (D6) depend on FreeRTOS
  priorities and kernel queues; ESP-IDF exposes them natively. TFLM, I2S DMA,
  and the official microWakeWord ESP-IDF component are first-class citizens.
  Arduino hides the scheduler and makes the audio/barge-in design we need much
  harder to express. Trade-off: steeper learning curve, but this project
  specifically needs the control.

### D2 — Brain = Raspberry Pi bridge first, direct-OpenAI as a variant
- **Choice:** two interchangeable "brain" modules behind one interface.
  Ship the **Pi bridge** first.
- **Why:** Whisper/LLM/TTS cannot run on the ESP32 (models too large). The Pi
  already runs the C++ port — we add a small server mode to it (accept a WAV
  over the network, run the existing transcribe→chat→TTS chain, stream PCM
  back). This reuses ~100% of the existing code, is dramatically easier to
  debug (the Pi logs everything), keeps the API key off the device, works
  offline, and has lower latency than a cloud round-trip. The direct-OpenAI
  module is a later swap of one module (multipart upload + JSON on the chip)
  and is needed only if you want the ESP32 to work without the Pi.

### D3 — One sample-rate domain: 16 kHz everywhere on the chip
- **Choice:** mic I2S @ 16 kHz; the brain resamples TTS from 24 kHz → 16 kHz
  before sending.
- **Why:** the wake word and VAD are already 16 kHz and the chip's I2S config,
  buffers, and DMA are simplest with a single rate. Resampling on the Pi/cloud
  side is free and keeps a resampler off the MCU. (If you later go standalone,
  you can request 24 kHz TTS and play it natively; both work — just pick one.)

### D4 — Hardware: PSRAM board + INMP441 + MAX98357A
- **Choice:** an ESP32-S3 board with **8 MB PSRAM** and ≥ 8 MB flash (e.g.
  ESP32-S3-DevKitC-1, S3-Box, or S3-DevKitM variants), an **INMP441** I2S mic,
  and a **MAX98357A** I2S amp.
- **Why PSRAM:** the int8 Silero VAD model is ~2.4 MB and cannot live in the
  512 KB SRAM; with PSRAM the model can be memory-mapped from flash or loaded
  there, and the audio/TLS ring buffers have headroom. INMP441 + MAX98357A are
  cheap, 3.3 V, I2S, 16-bit mono — a direct match for the pipeline. Avoid
  USB mics (S3 has no native USB audio class support).

### D5 — Wake word AND VAD on-chip; everything else off-chip
- **Choice:** the only always-on listeners live on the ESP32.
- **Why:** a 30 ms wake-word listener cannot tolerate network round-trips;
  VAD-on-chip also means only *speech segments* are sent to the brain, keeping
  bandwidth and brain load low. Whisper/LLM/TTS are inherently off-chip (D2).

### D6 — Decoupled FreeRTOS pipeline (fixes the crackling for good)
- **Choice:** the audio device is fed by a dedicated high-priority task from a
  kernel ring buffer; network download and barge-in inference live in separate
  lower-priority tasks. This is the same fix recommended for the Pi's playback
  stutter — on the S3 it's the native pattern (FreeRTOS priorities + I2S DMA).

---

## 2. Hardware bring-up

- [ ] **H1 — Buy/flash a dev board + wire the codecs.** INMP441 SCK/WS/SD and
      MAX98357A BCLK/LRC/DIN to the I2S pins; share one I2S peripheral if the
      board supports full-duplex, or use two.
      **Why:** nothing else can be validated without real audio hardware; this
      sandbox has no I2S/sound card.
- [ ] **H2 — Verify I2S RX loopback.** Feed a known 1 kHz tone into the mic and
      dump samples over USB-JTAG; confirm 16 kHz / 16-bit / mono and stable DMA.
      **Why:** a tone capture is the fastest way to isolate wiring/format bugs
      before any model runs.
- [ ] **H3 — Verify I2S TX.** Play a generated sine and a short WAV from flash.
      **Why:** confirms the DAC/amp path before the TTS pipeline exists.

## 3. Toolchain + project skeleton

- [ ] **T1 — Install ESP-IDF v5.x** (`idf.py`, `esp-idf` toolchain, `xtensa-esp32s3`).
      **Why:** D1. Nothing else runs until this is in place.
- [ ] **T2 — Scaffold the project** mirroring `cpp/`: `main/` (entry) +
      `components/jarvis/` (logic) + `components/jarvis_backend/` (ESP32 I/O).
      **Why:** keeps the logic/backend split from the C++ port so unit-tested
      code is never tangled with driver code.
- [ ] **T3 — Vendor TFLM + microWakeWord** as ESP-IDF components (or use
      `managed_components`).
      **Why:** microWakeWord is the same author/family as our model and already
      integrates TFLM + the exact frontend; using it avoids reinventing
      inference plumbing. We can also drop in our already-vendored frontend if
      a pure-port is preferred.
- [ ] **T4 — Port the logic headers verbatim**: `types.h`, `logger.h`,
      `conversation_manager.{h,cpp}`, `wav.{h,cpp}`, `signals.h`, and the
      `MicroFrontend` + wake-word/VAD state machines.
      **Why:** zero rework for already-verified logic (see §0).

## 4. Audio I/O (replaces PortAudio)

- [ ] **A1 — Implement `I2sAudio`** with the same interface as `AudioController`
      (`read_mic_chunk`, `write_samples`, `start/stop`): I2S RX @ 16 kHz + TX,
      DMA-driven, backed by FreeRTOS ring buffers.
      **Why:** the logic layer calls a narrow interface; swapping the backend
      keeps `record_until_silence` and barge-in logic untouched.
- [ ] **A2 — Add I2S DMA events → task notifications** so the audio task wakes
      on buffer-full/buffer-empty instead of polling.
      **Why:** zero-CPU audio I/O (the S3's real strength) and the foundation
      for D6.
- [ ] **A3 — Size the ring buffers** (mic ~320 ms, TTS ~1–1.5 s) in PSRAM.
      **Why:** the crackling fix is play-ahead; on an MCU the same math applies,
      and PSRAM makes it cheap.

## 5. Wake word on-chip (replaces dlopen + TFLite C API)

- [ ] **W1 — Run `hey_jarvis.tflite` under TFLM** with an int8 interpreter; arena
      ~23 KB (from the JSON), static allocation preferred.
      **Why:** TFLM is the MCU-optimized runtime; no dynamic loading on an MCU.
- [ ] **W2 — Measure wake-word latency** on real hardware (GPIO toggle + scope):
      target < 30–50 ms per 10 ms chunk.
      **Why:** it's the always-on cost (D5); if it misses, the whole device is
      unusable, so measure before building on top of it.
- [ ] **W3 — Add an energy pre-gate** (RMS on the mic buffer) before the neural
      path; skip TFLM on silence.
      **Why:** cuts idle CPU/battery massively — the neural net only runs when
      there's signal. Same optimization suggested for the Pi.
- [ ] **W4 — Cheap `reset_state`**: clear frontend + sliding-window buffers only,
      never reload the model.
      **Why:** TFLM state is in the arena; reloading costs flash reads + arena
      re-init on every wake/barge-in for no benefit (models are stateless conv).

## 6. VAD on-chip (replaces ONNX Runtime)

- [ ] **V1 — Convert `silero_vad.onnx` → int8 TFLite** (official Silero tflite
      builds, or `onnx2tf`/AI-Edge); diff outputs vs. the ONNX on the Pi.
      **Why:** ONNX Runtime doesn't exist for Xtensa; TFLM is the only
      practical runtime. The diff is the safety net that proves the conversion.
- [ ] **V2 — Port `record_until_silence`** state machine onto TFLM VAD inference,
      reusing the exact pre-roll/endpointing logic from `vad.cpp`.
      **Why:** the tricky part (state threading, timing) is already written and
      testable; only the inference call changes.
- [ ] **V3 — Solve VAD memory.** Map the ~2.4 MB model from flash (TFLM
      `mmap`/external data) and put the arena in PSRAM; if it doesn't fit
      comfortably, fall back to Silero **v4** (smaller) or a thinner 8-bit model.
      **Why:** 512 KB SRAM cannot hold it; flash-mmap avoids a RAM copy.
- [ ] **V4 — Gate VAD to after wake word.**
      **Why:** only the wake word runs always-on (D5); VAD is demand-driven,
      keeping idle CPU low.

## 7. Networking (replaces libcurl)

- [ ] **N1 — WiFi station + auto-reconnect** (`esp_netif`, `esp_wifi`), power-save
      off.
      **Why:** a voice assistant must not drop mid-turn; power-save adds latency
      that causes the network stalls behind playback glitches.
- [ ] **N2 — HTTPS via `esp_http_client` + mbedTLS with the CA-cert bundle**, one
      reusable connection (keep-alive) for transcribe→chat→TTS.
      **Why:** replaces libcurl; connection reuse removes 2 TLS handshakes per
      turn (the latency win from the efficiency analysis).
- [ ] **N3 — Implement `TtsChunkSource` over `esp_http_client`**: the event
      handler pushes PCM chunks into the FreeRTOS TTS queue.
      **Why:** exactly the downloader-task abstraction from D6; the audio task
      never blocks on the network.

## 8. The decoupled realtime pipeline (D6)

- [ ] **P1 — Audio-out task** (highest priority): drains the TTS ring buffer → I2S
      DMA; nothing else in this task.
      **Why:** the audio device can never be starved by inference or network
      (the crackle fix).
- [ ] **P2 — Barge-in task**: reads the mic ring buffer, runs wake-word TFLM,
      sets an atomic `interrupted` flag; keep the 1 s grace + 0.6 threshold.
      **Why:** inference moves out of the audio path (the Pi's #2 problem);
      the audio task only checks a flag.
- [ ] **P3 — Downloader task** (low priority): pulls TTS chunks (N3) into the
      queue.
      **Why:** network jitter is absorbed by the big ring buffer, never by the
      device.
- [ ] **P4 — Barge-in flush**: on `interrupted`, the audio task clears the ring
      buffer and returns to listen mode immediately.
      **Why:** playback latency of the queued audio doesn't matter — the user
      has already spoken over it; flushing keeps the device responsive.

## 9. Brain interface

- [ ] **B1 — Define `Brain` interface** (`transcribe(wav) -> text`,
      `chat(history) -> text`, `speak(text) -> TtsChunkSource`).
      **Why:** D2 — one interface, two implementations.
- [ ] **B2 — Pi bridge implementation**: ESP32 sends the WAV segment over
      TCP/WebSocket to the Pi; Pi runs the existing C++/Python chain and streams
      16 kHz PCM back (resampling 24→16 on the Pi).
      **Why:** D2/D3 — reuses the entire existing assistant and avoids an API
      key + multipart/TLS on the chip.
- [ ] **B3 — (Later) direct-OpenAI implementation**: `esp_http_client` multipart
      for Whisper, JSON for chat, streaming PCM for TTS.
      **Why:** only needed to remove the Pi dependency; it's a contained module
      swap once B1 exists.

## 10. Orchestration + peripherals (port from `cpp/`)

- [ ] **O1 — Port the orchestrator state machine** (standby → wake → record →
      transcribe → chat → speak → barge-in) onto the task layout; reuse
      `conversation_manager` + stop/hallucination logic as-is.
      **Why:** the flow logic is done; only the backend calls change.
- [ ] **O2 — Port `HardwareController`** to `gpio_hal`/`ledc` (thinking/listening
      LEDs, BCM 5/13 → ESP32 pins).
      **Why:** trivial, and keeps the same status signaling.

## 11. Integration tests on target

- [ ] **I1 — Port the golden wake-word test to the device**: feed the recorded
      `ref_wakeword.bin` PCM (from `cpp/tests/data/`) from flash and assert the
      same probabilities.
      **Why:** the strongest proof the on-chip port is bit-faithful to the
      Python/C++ pipeline (we already prove err=0 on the host).
- [ ] **I2 — Recorded-speech → VAD → transcription → chat → TTS → playback**
      end-to-end with a fixed prompt.
      **Why:** exercises every stage deterministically before live conversation.
- [ ] **I3 — Live smoke test**: wake word on real voice, 2-turn conversation,
      barge-in during TTS, stop command.
      **Why:** the acceptance test for "Jarvis works on an S3".
- [ ] **I4 — Profile**: heap (`idf.py monitor`), per-task stack high-water marks,
      wake-word CPU%.
      **Why:** catches stack overflows / memory leaks before deployment; the
      S3's constraints are real.

## 12. Hardening / deploy (optional, after it works)

- [ ] **D1 — OTA updates** (`esp_ota_ops`) and a watchdog.
      **Why:** field-updating a voice device is otherwise painful.
- [ ] **D2 — Power**: light sleep between turns with periodic mic sampling, or a
      hardware noise-gate to keep the wake word asleep.
      **Why:** an always-on neural listener prevents deep sleep; this is the
      battery-life knob for a battery-powered unit.
- [ ] **D3 — Robustness**: TLS session resumption, WiFi RSSI monitoring, NVS for
      last-known-good settings.
      **Why:** an embedded assistant must self-heal after network blips.

---

## Rough ordering + effort

| Phase | Focus | Effort |
|---|---|---|
| 1–3 | Decisions, toolchain, skeleton | ~1–2 days |
| 4 | I2S audio | ~1–2 days |
| 5 | Wake word on TFLM | ~1 day (frontend already ported) |
| 6 | VAD on TFLM | ~2–3 days (conversion + memory) |
| 7–8 | Network + realtime pipeline | ~2–3 days |
| 9 | Brain interface (Pi bridge) | ~1–2 days |
| 10–11 | Orchestration + tests | ~1–2 days |
| 12 | Hardening | as needed |

Total realistic: **~1.5–3 weeks** of focused work by one person, with hardware
in hand. The single biggest risk item is **VAD memory/performance on-device**
(§6) — everything else has a well-trodden path.
