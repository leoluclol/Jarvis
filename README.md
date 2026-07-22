# Jarvis Voice Assistant (Raspberry Pi 2 / 32-bit ARM Edition)

Jarvis is an always-listening voice assistant prototype that combines wake-word detection, speech recognition, LLM responses, and text-to-speech. Wake-word detection runs fully on-device with **Rhasspy Raven** (MFCC + Dynamic Time Warping), which needs no neural runtime at all — no ONNXRuntime, no TensorFlow Lite. Everything downstream of the wake word is an OpenAI API call.

Current language behavior:
- Wake word: whatever phrase **you record** (the code assumes "Hey Jarvis")
- Conversation: Italian (forced by the system prompt)

## Wake word: you must train it yourself

Raven ships **no pretrained models**. The wake word is defined by 3+ recordings of your own voice:

```bash
python jarvis.py --record
```

Speak the phrase, wait for the `✅ Salvato` confirmation, repeat at least 3 times, then Ctrl+C. Templates are written to `models/raven/hey_jarvis/`. Record them on the Pi with the same microphone you will use in production — templates are speaker- and mic-specific.

Then run normally:

```bash
python jarvis.py
```

## How It Works

Audio pipeline:
1. Microphone audio is captured continuously with PyAudio (60 ms chunks).
2. Raven's internal WebRTC VAD gates the expensive work — MFCC/DTW runs only on non-silent audio.
3. Raven compares a sliding window of MFCC features against your recorded templates using DTW; a cosine distance below the threshold means detection.
4. WebRTC VAD delimits the spoken command.
5. Command is transcribed with OpenAI Whisper.
6. Response is generated with OpenAI Chat Completions.
7. Response audio is generated with OpenAI TTS and played back.
8. During playback, barge-in is available but **off by default** (see below).

## Features

- Always-on microphone listener
- Wake-word activation with Rhasspy Raven (pure MFCC/DTW, no ML runtime)
- Personal wake word recorded by the user (`--record`)
- Voice activity detection with webrtcvad
- Speech-to-text with Whisper
- Conversational memory (short rolling history)
- Text-to-speech response playback
- Optional barge-in during TTS playback (`ENABLE_BARGE_IN`, off by default)
- Ambient noise floor measured at startup, so end-of-speech detection adapts to the room

## Tuning on a Raspberry Pi 2

All knobs are at the top of `jarvis.py`:

- `RAVEN_AVERAGE_TEMPLATES = True` collapses N templates into one averaged template. This is the single biggest CPU saving — DTW cost scales linearly with template count. Set to `False` only if accuracy is poor and you have headroom.
- `RAVEN_FAILED_MATCHES_TO_REFRACTORY = 10` forces a refractory pause after repeated near-misses, so background chatter cannot pin the CPU.
- `RAVEN_PROBABILITY_THRESHOLD` / `RAVEN_DISTANCE_THRESHOLD` trade false positives against missed detections. Lower distance threshold = stricter.
- `ENABLE_BARGE_IN` is **off by default**, and not just for CPU reasons. While the speakers are playing, the microphone hears Jarvis on top of you. openWakeWord tolerated that because it is a neural model trained with noise and echo; Raven compares MFCCs with DTW, so the echo shifts the features and the distance explodes. No threshold fixes it — that would need acoustic echo cancellation, which is too much for a Pi 2. If you enable it, interruption is detected by **loudness relative to the echo**, not by the wake word, and you must start speaking *after* playback begins.
- `SPEECH_SNR_RATIO` / `SPEECH_MIN_RMS` control how far above the measured room noise audio must be to count as speech. Raise `SPEECH_SNR_RATIO` if recordings keep running on background noise; lower it if Jarvis stops hearing you in a noisy room.
- `MAX_COMMAND_SEC` caps a single command recording, so a noisy room cannot produce an unbounded file.

## Project Structure

- `jarvis.py`: Main assistant loop, audio pipeline, and `--record` template trainer
- `requirements.txt`: Python dependencies, pinned to prebuilt armv7l wheels
- `requirements-raven.txt`: Wake-word stack, installed separately with `--no-deps`
- `models/raven/hey_jarvis/`: Your recorded wake-word templates (WAV)

## Requirements

- **Raspberry Pi OS 13 "Trixie" 32-bit (armv7l), Python 3.13** — the supported target
- A working microphone and speakers
- OpenAI API key and a reliable network connection (STT, LLM, and TTS are all remote)
- PortAudio and OpenBLAS system libraries

**Use Trixie's own `/usr/bin/python3` (3.13).** Do not install a separate interpreter for this project. piwheels builds one wheel set per Python ABI, matched to the OS release — `cp313` wheels are built on Trixie — so the system interpreter is what makes a Pi 2 download wheels instead of compiling numpy and scipy for hours. A hand-installed Python (pyenv, uv-managed, source build) is the fast route to an accidental multi-hour build.

```bash
python3 -m venv ~/jarvis-venv
source ~/jarvis-venv/bin/activate
```

Raspberry Pi OS 12 "Bookworm" (Python 3.11) still works — `requirements.txt` picks the numpy/scipy pair by environment marker, numpy 1.26 / scipy 1.11 below Python 3.12 and numpy 2.5 / scipy 1.18 at or above it. Both are tested, but Trixie/3.13 is the combination this project is developed against.

## Installation

```bash
sudo apt update
sudo apt install -y portaudio19-dev libopenblas-dev
```

Trixie no longer ships the piwheels configuration, so add it yourself — without this, pip falls back to source builds:

```bash
printf '[global]\nextra-index-url=https://www.piwheels.org/simple\n' | sudo tee /etc/pip.conf
```

Then install in two steps — **the second one must use `--no-deps`**:

```bash
pip install -r requirements.txt
pip install --no-deps -r requirements-raven.txt
```

### If you insist on uv

Not recommended here, but if you use `uv`, be aware it does **not** read `/etc/pip.conf`, so the piwheels line above is silently ignored and every armv7l wheel is missed:

```bash
export UV_EXTRA_INDEX_URL=https://www.piwheels.org/simple
uv venv --system-site-packages --python /usr/bin/python3
```

Always point it at the system interpreter. A uv-downloaded Python has a different ABI tag than the distro's, and piwheels may have no matching wheels for it at all.

### Why the install is split in two

`rhasspy-wake-raven` hardcodes `scipy==1.5.1` in its metadata. scipy 1.5.1 only ever published cp36/cp37/cp38 wheels, so on Python 3.9+ pip falls back to building it from source, and that build pulls an ancient numpy whose distutils shim is broken on modern setuptools. The result is the confusing error:

```
NameError: name 'CCompiler' is not defined
```

That pin is stale, not a real constraint — Raven only calls `scipy.io.wavfile.read`. Installing it with `--no-deps` against the pinned scipy 1.11.4 is tested and works on Python 3.10 and 3.11.

Two consequences worth knowing:

- `pip check` will report `rhasspy-wake-raven 0.5.2 requires scipy==1.5.1` forever. This warning is expected and cosmetic.
- Do **not** downgrade `rhasspy-wake-raven` to 0.3.x to dodge the pin. 0.3.x has the same `scipy==1.5.1` pin, additionally requires `rhasspy-silence~=0.3.0`, and lacks the `failed_matches_to_refractory` parameter that `jarvis.py` relies on to cap CPU usage on the Pi 2.

### Two stdlib/tooling shims you cannot drop

- `setuptools==75.8.0` is a **runtime** dependency, not a build tool. `webrtcvad` does `import pkg_resources`, which lives inside setuptools, and setuptools 83.0.0 removed it. Without the pin, `import webrtcvad` fails with `ModuleNotFoundError: No module named 'pkg_resources'`.
- `audioop-lts` is required on **Python 3.13+** (so: on Trixie). Python 3.13 removed the `audioop` stdlib module under PEP 594, and `rhasspy-silence` imports it unconditionally. Without it, `import jarvis` dies with `ModuleNotFoundError: No module named 'audioop'`. The marker in `requirements.txt` installs it only where it is needed.
