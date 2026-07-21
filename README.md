# Jarvis Voice Assistant (Raspberry Pi 2 / 32-bit ARM Edition)

Jarvis is a local, always-listening voice assistant prototype that combines wake-word detection, speech recognition, LLM responses, and text-to-speech. This version is specifically optimized to run on 32-bit ARM hardware (like the Raspberry Pi 2) using **TensorFlow Lite** instead of ONNXRuntime.

Current language behavior:
- Wake word: English phrase Hey Jarvis
- Conversation: Italian (forced by the system prompt)

## How It Works

Audio pipeline:
1. Microphone audio is captured continuously with PyAudio.
2. WebRTC VAD checks if the input contains human speech.
3. openWakeWord detects the wake word hey_jarvis using TensorFlow Lite (TFLite).
4. User command is transcribed with OpenAI Whisper.
5. Response is generated with OpenAI Chat Completions.
6. Response audio is generated with OpenAI TTS and played back.
7. During playback, the assistant supports barge-in if Hey Jarvis is spoken again.

## Features

- Always-on microphone listener
- Wake-word activation with openWakeWord (TFLite Engine)
- Voice activity detection with webrtcvad
- Speech-to-text with Whisper
- Conversational memory (short rolling history)
- Text-to-speech response playback
- Barge-in interruption during TTS playback

## Project Structure

- jarvis.py: Main assistant loop and audio pipeline
- requirements.txt: Python dependencies configured for 32-bit ARM
- models/: Local wake-word model files
- resources.txt: Resource usage log sample

## Requirements

- Linux / Raspberry Pi OS 32-bit (armv7l)
- Python 3.10+
- A working microphone and speakers
- OpenAI API key
- PortAudio development libraries and OpenBLAS (required for PyAudio and math libraries on Pi 2)
- `webrtcvad` and `tflite-runtime` installed through `requirements.txt`

On Raspberry Pi OS / Debian / Ubuntu, install the required system headers first:

```bash
sudo apt update
sudo apt install -y portaudio19-dev libspeexdsp-dev libopenblas-dev