import os
import wave
import threading
import pyaudio
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import openwakeword
from openwakeword.model import Model
import time

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WAKE_WORD = "hey_jarvis"
PIPER_MODEL = "it_IT-riccardo-x_low.onnx"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  
RECORD_SECONDS = 5
COMMAND_AUDIO_PATH = "command.wav"
RESPONSE_AUDIO_PATH = "response.wav"

# Conversation Memory (Max 3 pairs = 6 messages)
MAX_HISTORY = 6 
conversation_history = [
    {"role": "system", "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."}
]

client = OpenAI(api_key=OPENAI_API_KEY)

def record_audio(pyaudio_instance, duration=RECORD_SECONDS):
    """Records audio from the microphone for a fixed duration."""
    print("\n🎙️ Ascoltando...")
    stream = pyaudio_instance.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, 
        input=True, frames_per_buffer=CHUNK
    )
    
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        
    print("🛑 Registrazione terminata.")
    stream.stop_stream()
    stream.close()
    
    with wave.open(COMMAND_AUDIO_PATH, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pyaudio_instance.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def play_audio(pyaudio_instance, file_path):
    """Plays back a WAV audio file synchronously."""
    with wave.open(file_path, 'rb') as wf:
        stream = pyaudio_instance.open(
            format=pyaudio_instance.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        data = wf.readframes(1024)
        while data:
            stream.write(data)
            data = wf.readframes(1024)
        stream.stop_stream()
        stream.close()

def run_voice_assistant():
    global conversation_history
    
    print(f"⚙️ Inizializzazione modello '{WAKE_WORD}'...")
    oww_model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
    pa = pyaudio.PyAudio()
    
    # Open continuous microphone stream
    audio_stream = pa.open(
        rate=RATE, channels=CHANNELS, format=FORMAT,
        input=True, frames_per_buffer=CHUNK
    )
    
    print(f"\n🤖 Jarvis è pronto. Pronuncia '{WAKE_WORD.replace('_', ' ')}' per iniziare.")
    
    try:
        while True:
            # 1. Idle Loop: Wait for wake word
            pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            prediction = oww_model.predict(audio_data)
            
            if prediction.get(WAKE_WORD, 0) > 0.5:
                print("\n✨ Parola d'ordine rilevata! ✨")
                audio_stream.stop_stream()
                
                # 2. Record Command
                record_audio(pa, duration=5)
                
                # 3. Transcribe
                print("🧠 Trascrizione in corso...")
                with open(COMMAND_AUDIO_PATH, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file
                    )
                user_text = transcription.text
                print(f"👤 Tu: {user_text}")
                
                # Add to memory
                conversation_history.append({"role": "user", "content": user_text})
                
                # 4. LLM Generation (with Memory)
                print("🧠 Elaborazione risposta...")
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=conversation_history
                )
                ai_text = response.choices[0].message.content
                print(f"🤖 Jarvis: {ai_text}")
                
                # Add AI response to memory and trim history (keep system prompt + last 6 messages)
                conversation_history.append({"role": "assistant", "content": ai_text})
                if len(conversation_history) > MAX_HISTORY + 1:
                    conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]
                
                # 5. Online TTS via OpenAI
                print("🗣️ Generazione voce (OpenAI)...")
                tts_response = client.audio.speech.create(
                    model="gpt-4o-mini-tts",
                    voice="onyx", # Change to "alloy", "echo", "fable", "nova", or "shimmer" if you prefer
                    response_format="wav",
                    input=ai_text
                )
                tts_response.stream_to_file(RESPONSE_AUDIO_PATH)
                
                # 6. Playback (Blocking)
                print("🔊 Riproduzione audio...")
                play_audio(pa, RESPONSE_AUDIO_PATH)
                
                # 7. Flush buffer and go back to Idle
                time.sleep(0.3)
                audio_stream.start_stream()
                for _ in range(15):  
                    audio_stream.read(CHUNK, exception_on_overflow=False)
                oww_model.reset()
                print("\n🤖 Torno in standby. In attesa della parola d'ordine...")
                    
    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
    finally:
        audio_stream.close()
        pa.terminate()

if __name__ == "__main__":
    run_voice_assistant()