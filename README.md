# Jarvis Voice Assistant

Jarvis is a local, always-listening voice assistant prototype that combines wake-word detection, speech recognition, LLM responses, and text-to-speech.

Current language behavior:
- Wake word: English phrase Hey Jarvis
- Conversation: Italian (forced by the system prompt)

## How It Works

Audio pipeline:
1. Microphone audio is captured continuously with PyAudio.
2. WebRTC VAD checks if the input contains human speech.
3. openWakeWord detects the wake word hey_jarvis.
4. User command is transcribed with OpenAI Whisper.
5. Response is generated with OpenAI Chat Completions.
6. Response audio is generated with OpenAI TTS and played back.
7. During playback, the assistant supports barge-in if Hey Jarvis is spoken again.

## Features

- Always-on microphone listener
- Wake-word activation with openWakeWord
- Voice activity detection with webrtcvad
- Speech-to-text with Whisper
- Conversational memory (short rolling history)
- Text-to-speech response playback
- Barge-in interruption during TTS playback

## Project Structure

- jarvis.py: Main assistant loop and audio pipeline
- requirements.txt: Python dependencies
- models/: Local wake-word model files
- resources.txt: Resource usage log sample

## Requirements

- Linux (tested in this environment)
- Python 3.10+
- A working microphone and speakers
- OpenAI API key
- PortAudio development libraries (required by PyAudio)
- `webrtcvad` installed through `requirements.txt`

On Debian/Ubuntu, install PortAudio first:

```bash
sudo apt update
sudo apt install -y portaudio19-dev
```

## Installation

1. Clone the repository and enter it.
2. Create and activate a virtual environment.
3. Install dependencies.
4. Create a .env file with your API key.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Run

```bash
python3 jarvis.py
```

When the assistant prints standby mode, say Hey Jarvis to start a conversation.

## Current Runtime Configuration

Important defaults from the code:
- Wake word model id: hey_jarvis
- Audio sample rate: 16000 Hz
- Audio chunk size: 1280 samples
- Silence timeout before cancel: 6.0 seconds
- End-of-speech silence timeout: 1.5 seconds
- VAD mode: 2 (more aggressive than the default)
- Chat model: gpt-5.4-mini-2026-03-17
- TTS model: gpt-4o-mini-tts
- TTS voice: onyx

## Notes and Limitations

- The assistant is configured to answer in Italian.
- Wake-word detection currently expects Hey Jarvis.
- VAD uses 30 ms PCM frames with WebRTC's built-in speech detector.
- Internet connection is required for OpenAI API calls.
- High CPU or RAM usage can occur during active audio + inference loops.

## Raspberry Pi 2 Survival Guide

If you want Jarvis to run reliably on a Raspberry Pi 2 for long periods, apply these optimizations.

1. Pre-allocate buffers in hot loops
	Avoid creating new objects repeatedly inside real-time loops. Frequent allocations increase garbage-collection pauses, which can cause audio glitches on Pi 2. Reuse buffers and temporary arrays whenever possible.

2. Try PyPy (optional)
	If your bottleneck is Python loop performance, PyPy can improve throughput through JIT compilation. In tight loops, this can make the assistant noticeably smoother.

3. Optimize model choice for latency and cost
	Prefer a fast, broadly available chat model for production usage on constrained hardware. For most setups, gpt-4o-mini will reduce latency and cost compared to heavier models.

	Before deployment, verify that your selected model ID is available in your OpenAI account and region.

## Roadmap

- Add real streaming TTS playback
- Improve wake-word behavior to support a single word trigger like Jarvis

## Troubleshooting

- PyAudio install fails:
	Install PortAudio system headers, then reinstall dependencies.
- Wake word not detected:
	Check microphone input level and background noise.
- No responses from OpenAI:
	Verify OPENAI_API_KEY in .env and network access.
- Assistant hears itself too often:
	Lower speaker volume or increase physical separation between mic and speakers.