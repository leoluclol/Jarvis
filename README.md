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
8. During playback, the assistant supports barge-in if the wake word is spoken again.

## Features

- Always-on microphone listener
- Wake-word activation with Rhasspy Raven (pure MFCC/DTW, no ML runtime)
- Personal wake word recorded by the user (`--record`)
- Voice activity detection with webrtcvad
- Speech-to-text with Whisper
- Conversational memory (short rolling history)
- Text-to-speech response playback
- Barge-in interruption during TTS playback (disable with `ENABLE_BARGE_IN = False`)

## Tuning on a Raspberry Pi 2

All knobs are at the top of `jarvis.py`:

- `RAVEN_AVERAGE_TEMPLATES = True` collapses N templates into one averaged template. This is the single biggest CPU saving — DTW cost scales linearly with template count. Set to `False` only if accuracy is poor and you have headroom.
- `RAVEN_FAILED_MATCHES_TO_REFRACTORY = 10` forces a refractory pause after repeated near-misses, so background chatter cannot pin the CPU.
- `RAVEN_PROBABILITY_THRESHOLD` / `RAVEN_DISTANCE_THRESHOLD` trade false positives against missed detections. Lower distance threshold = stricter.
- `ENABLE_BARGE_IN = False` if the TTS playback stutters — barge-in runs DTW and audio output at the same time.

## Project Structure

- `jarvis.py`: Main assistant loop, audio pipeline, and `--record` template trainer
- `requirements.txt`: Python dependencies configured for 32-bit ARM
- `models/raven/hey_jarvis/`: Your recorded wake-word templates (WAV)

## Requirements

- Linux / Raspberry Pi OS 32-bit (armv7l)
- Python 3.9+
- A working microphone and speakers
- OpenAI API key and a reliable network connection (STT, LLM, and TTS are all remote)
- PortAudio development libraries and OpenBLAS (required for PyAudio and numpy/scipy on Pi 2)

On Raspberry Pi OS / Debian / Ubuntu, install the required system headers first:

```bash
sudo apt update
sudo apt install -y portaudio19-dev libopenblas-dev
```

Then follow the install steps documented at the top of `requirements.txt` — the `--no-deps` step is mandatory, since `rhasspy-wake-raven` pins `scipy==1.5.1` and pip will otherwise try to build scipy from source on the Pi.
