import os
import wave
import time
import queue
import pyaudio
import numpy as np
import psutil
from dotenv import load_dotenv
from openai import OpenAI

import sys
from unittest.mock import MagicMock
sys.modules["onnxruntime"] = MagicMock()
import openwakeword

from openwakeword.model import Model
import webrtcvad

# ==========================================
# RESOURCE PROFILER (CONTEXT MANAGER)
# ==========================================
import csv
from datetime import datetime

class ResourceProfiler:
    """
    Profiles execution time, CPU, and RAM, printing to the terminal 
    AND appending structured metrics to a CSV file for graphing.
    """
    def __init__(self, phase_name: str, log_file: str = "resources.csv"):
        self.phase_name = phase_name
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        
        # Create CSV header if the file doesn't exist yet
        if not os.path.exists(self.log_file):
            with open(self.log_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "phase", "duration_sec", "cpu_percent", "ram_mb", "ram_diff_mb"])

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_mem = self.process.memory_info().rss / (1024 * 1024)
        self.process.cpu_percent(interval=None)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        end_mem = self.process.memory_info().rss / (1024 * 1024)
        mem_diff = end_mem - self.start_mem
        cpu_usage = self.process.cpu_percent(interval=None)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Print to Terminal
        print(f"\n📊 [RESOURCE LOG: {self.phase_name}]")
        print(f"   ⏱️  Execution Time  : {elapsed_time:.2f} seconds")
        print(f"   ⚙️  CPU Utilization : {cpu_usage:.1f}%")
        print(f"   🧠 RAM Usage       : {end_mem:.2f} MB ({mem_diff:+.2f} MB)")
        print("─" * 55)

        # 2. Append to CSV File
        with open(self.log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.phase_name, f"{elapsed_time:.4f}", f"{cpu_usage:.1f}", f"{end_mem:.2f}", f"{mem_diff:+.2f}"])

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WAKE_WORD = "hey_jarvis"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # 80ms chunks (richiesto da openWakeWord)
COMMAND_AUDIO_PATH = "command.wav"
RESPONSE_AUDIO_PATH = "response.wav"

# VAD (Voice Activity Detection) Settings
SILENCE_TIMEOUT_VAD = 1.5  # Secondi di silenzio prima di chiudere il comando
SILENCE_TIMEOUT = 6.0      # Secondi di silenzio prima di annullare il comando
MIN_SPEECH_CHUNKS = 3      # ~240ms di voce continua richiesti per iniziare a registrare
VAD_MODE = 2               # 0-3, higher is more aggressive
VAD_FRAME_DURATION_MS = 30
VAD_FRAME_SIZE = int(RATE * VAD_FRAME_DURATION_MS / 1000)
VAD_FRAME_BYTES = VAD_FRAME_SIZE * 2

# Conversation Memory (Max 3 coppie = 6 messaggi)
MAX_HISTORY = 6 
conversation_history = [
    {"role": "system", "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."}
]

client = OpenAI(api_key=OPENAI_API_KEY)

# Coda Asincrona Thread-Safe per l'audio del microfono
coda_mic = queue.Queue()

# Inizializzazione VAD WebRTC
print("🧠 Caricamento webrtcvad...")
vad = webrtcvad.Vad(VAD_MODE)
vad_buffer = b""

def reset_vad_buffer():
    global vad_buffer
    vad_buffer = b""

def mic_callback(in_data, frame_count, time_info, status):
    coda_mic.put(in_data)
    return (None, pyaudio.paContinue)

def is_speech(audio_chunk: bytes) -> bool:
    global vad_buffer
    if not audio_chunk:
        return False

    vad_buffer += audio_chunk
    if len(vad_buffer) < VAD_FRAME_BYTES:
        return False

    max_valid_length = len(vad_buffer) - (len(vad_buffer) % VAD_FRAME_BYTES)

    for start in range(0, max_valid_length, VAD_FRAME_BYTES):
        frame = vad_buffer[start:start + VAD_FRAME_BYTES]
        if vad.is_speech(frame, RATE):
            vad_buffer = vad_buffer[start + VAD_FRAME_BYTES:]
            return True

    vad_buffer = vad_buffer[max_valid_length:]
    return False

def record_dynamic_audio(oww_model):
    print("\n🎙️ Ascoltando... (stai zitto per annullare)")
    
    start_time = time.time()
    frames = []
    has_spoken = False
    reset_vad_buffer()
    
    silent_chunks = 0
    max_silent_chunks = int((SILENCE_TIMEOUT_VAD * RATE) / CHUNK)
    
    while True:
        pcm = coda_mic.get()
        
        if not has_spoken:
            if time.time() - start_time > SILENCE_TIMEOUT:
                print("⏳ Timeout: Nessuna parola rilevata, torno in standby.")
                return None
        
        is_current_speech = is_speech(pcm)
        
        if is_current_speech:
            has_spoken = True
            frames.append(pcm)
            silent_chunks = 0
        else:
            if has_spoken:
                frames.append(pcm)
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    print("🛑 Fine del discorso rilevata.")
                    break

    with wave.open(COMMAND_AUDIO_PATH, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return COMMAND_AUDIO_PATH

def play_audio_with_barge_in(pa: pyaudio.PyAudio, file_path: str, oww_model: Model):
    with wave.open(file_path, "rb") as wf:
        wav_chunk_size = int(wf.getframerate() * (CHUNK / RATE))

        out_stream = pa.open(
            format=pa.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            frames_per_buffer=wav_chunk_size,
        )

        interrupted = False
        data = wf.readframes(wav_chunk_size)

        while data:
            out_stream.write(data)

            while not coda_mic.empty():
                pcm = coda_mic.get_nowait()
                audio_data = np.frombuffer(pcm, dtype=np.int16)
                
                prediction = oww_model.predict(audio_data)
                if prediction.get(WAKE_WORD, 0) > 0.5:
                    print("\n⚡ BARGE-IN RILEVATO! ('Hey Jarvis' ascoltato durante la riproduzione)... ⚡")
                    interrupted = True
                    oww_model.reset()
                    break

            if interrupted:
                break

            data = wf.readframes(wav_chunk_size)

        out_stream.stop_stream()
        out_stream.close()
        return interrupted

def run_voice_assistant():
    global conversation_history
    
    print(f"⚙️ Inizializzazione openWakeWord ('{WAKE_WORD}') in modalità TFLite...")
    oww_model = Model(
        wakeword_models=[WAKE_WORD],
        inference_framework="tflite"
    )
    
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=RATE, channels=CHANNELS, format=FORMAT,
        input=True, frames_per_buffer=CHUNK,
        stream_callback=mic_callback
    )
    audio_stream.start_stream()
    
    print(f"\n🤖 Jarvis è in STANDBY. Pronuncia \"Hey Jarvis\" per attivare la conversazione.")
    
    # Initialize telemetry tracking for the standby loop
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)  # Checkpoint starting CPU baseline
    standby_chunks = 0
    standby_infer_time = 0.0
    
    try:
        while True:
            pcm = coda_mic.get()
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            
            # Profile standby inference speed per frame
            t0 = time.perf_counter()
            prediction = oww_model.predict(audio_data)
            standby_infer_time += (time.perf_counter() - t0) * 1000  # Convert to ms
            standby_chunks += 1
            
            # Print Standby Telemetry every 100 chunks (~8 seconds of audio)
            if standby_chunks % 100 == 0:
                avg_ms = standby_infer_time / 100
                cpu_load = process.cpu_percent(interval=None)
                ram_mb = process.memory_info().rss / (1024 * 1024)
                print(f"📡 [STANDBY HEARTBEAT] Infer: {avg_ms:.2f}ms/chunk | CPU: {cpu_load:.1f}% | RAM: {ram_mb:.1f} MB")
                standby_infer_time = 0.0
            
            if prediction.get(WAKE_WORD, 0) > 0.5:
                print("\n✨ Parola d'ordine rilevata! Modalità conversazione ATTIVA. ✨")
                oww_model.reset()
                
                time.sleep(0.1)
                while not coda_mic.empty():
                    coda_mic.get_nowait()
                
                in_active_conversation = True
                
                while in_active_conversation:
                    # 1. Profile Audio Recording & VAD
                    with ResourceProfiler("1. Audio Recording & VAD"):
                        audio_path = record_dynamic_audio(oww_model)
                        
                    if not audio_path:
                        break
                        
                    # 2. Profile OpenAI Whisper STT (Network I/O)
                    print("🧠 Trascrizione in corso...")
                    with ResourceProfiler("2. Whisper Speech-to-Text (Network I/O)"):
                        with open(COMMAND_AUDIO_PATH, "rb") as audio_file:
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1", 
                                file=audio_file,
                                language="it",
                                temperature=0.0,
                                prompt="Comandi vocali per assistente domestico Jarvis in italiano."
                            )
                    user_text = transcription.text.strip()
                    if not user_text:
                        continue
                        
                    print(f"👤 Tu: {user_text}")
                    
                    if "hey jarvis" in user_text.lower() and len(user_text.split()) <= 4:
                        print("🤖 Comando di chiusura vocale riconosciuto. Torno in standby.")
                        break
                    
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    # 3. Profile GPT Reasoning (Network I/O)
                    print("🧠 Elaborazione risposta...")
                    with ResourceProfiler("3. GPT LLM Reasoning (Network I/O)"):
                        response = client.chat.completions.create(
                            model="gpt-5.4-mini-2026-03-17",
                            messages=conversation_history
                        )
                    ai_text = response.choices[0].message.content
                    print(f"🤖 Jarvis: {ai_text}")
                    
                    conversation_history.append({"role": "assistant", "content": ai_text})
                    if len(conversation_history) > MAX_HISTORY + 1:
                        conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]
                    
                    # 4. Profile OpenAI TTS Generation (Network I/O)
                    print("🗣️ Generazione voce...")
                    with ResourceProfiler("4. OpenAI TTS Generation (Network I/O)"):
                        with client.audio.speech.with_streaming_response.create(
                            model="gpt-4o-mini-tts",
                            voice="onyx",
                            response_format="wav",
                            input=ai_text,
                            speed=1.15,
                        ) as tts_response:
                            tts_response.stream_to_file(RESPONSE_AUDIO_PATH)
                        
                    # 5. Profile Audio Playback & Barge-In
                    print("🔊 Riproduzione sulle casse (pronuncia 'Hey Jarvis' per interrompere)...")
                    with ResourceProfiler("5. Audio Playback & Barge-in Monitoring"):
                        interrupted = play_audio_with_barge_in(
                            pa, RESPONSE_AUDIO_PATH, oww_model
                        )
                    
                    if interrupted:
                        time.sleep(0.1)
                        while not coda_mic.empty():
                            coda_mic.get_nowait()
                        print("\n👂 Prontissimo! Dimmi pure il nuovo comando...")
                        continue
                    else:
                        time.sleep(0.2)
                        print("\n👂 In attesa del prossimo turno (o pronuncia 'Hey Jarvis' per uscire)...")
                        
                oww_model.reset()
                print("\n🤖 Torno in STANDBY. In attesa di 'Hey Jarvis'...")
                # Reset baseline checkpoint after exiting active conversation
                process.cpu_percent(interval=None)
                    
    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

if __name__ == "__main__":
    run_voice_assistant()