import os
import wave
import time
import queue
import pyaudio
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import sys
from unittest.mock import MagicMock
sys.modules["onnxruntime"] = MagicMock()
import openwakeword

from openwakeword.model import Model
import webrtcvad

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
MIN_SPEECH_CHUNKS = 3  # ~240ms di voce continua richiesti per iniziare a registrare
VAD_MODE = 2  # 0-3, higher is more aggressive
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
    """
    Callback di PortAudio in background.
    Cattura l'audio del microfono e lo mette nella coda senza bloccare il programma principale.
    """
    coda_mic.put(in_data)
    return (None, pyaudio.paContinue)

def is_speech(audio_chunk: bytes) -> bool:
    """
    Verifica se un frammento audio contiene voce umana usando webrtcvad.
    L'audio viene analizzato in frame PCM mono 16-bit da 30ms.
    """
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
    """
    Registra audio:
    1. Fase di attesa: max 6s. Se non parla mai, return None.
    2. Fase di registrazione: se parla, continua finché non c'è silenzio (1.5s).
    """
    print("\n🎙️ Ascoltando... (stai zitto per annullare)")
    
    start_time = time.time()
    frames = []
    has_spoken = False
    reset_vad_buffer()
    
    # Parametri temporali
    silent_chunks = 0
    max_silent_chunks = int((SILENCE_TIMEOUT_VAD * RATE) / CHUNK) # 1.5s di silenzio
    
    while True:
        # Preleva audio dalla coda (non bloccante)
        pcm = coda_mic.get()
        
        # 1. FASE DI ATTESA (Timeout 6s)
        # Se non ha ancora parlato, controlliamo se sono passati 6 secondi
        if not has_spoken:
            if time.time() - start_time > SILENCE_TIMEOUT:
                print("⏳ Timeout: Nessuna parola rilevata, torno in standby.")
                return None # Ritorna None come richiesto
        
        # 2. CONTROLLO PAROLA (webrtcvad)
        # Verifichiamo se l'audio corrente contiene voce umana
        is_current_speech = is_speech(pcm)
        
        if is_current_speech:
            has_spoken = True
            frames.append(pcm)
            silent_chunks = 0 # Reset del contatore di silenzio
        else:
            if has_spoken:
                frames.append(pcm)
                silent_chunks += 1
                # Se abbiamo già parlato e c'è silenzio per 1.5s, finiamo la registrazione
                if silent_chunks >= max_silent_chunks:
                    print("🛑 Fine del discorso rilevata.")
                    break

    # Scrittura su file se abbiamo parlato
    with wave.open(COMMAND_AUDIO_PATH, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return COMMAND_AUDIO_PATH

def play_audio_with_barge_in(pa: pyaudio.PyAudio, file_path: str, oww_model: Model):
    """
    Riproduce l'audio sulle casse monitorando ESCLUSIVAMENTE la parola d'ordine "Hey Jarvis".
    Ignora la voce normale di Jarvis che esce dalle casse ed evita auto-interruzioni.
    """
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
            # 1. Scrittura audio sulle casse esterne
            out_stream.write(data)

            # 2. Svuotamento coda e controllo ESCLUSIVO del Wake Word (Hey Jarvis)
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
    
    print(f"⚙️ Inizializzazione openWakeWord ('{WAKE_WORD}') in modalità TFLite (32-bit ARM)...")
    # MODIFICATO: Utilizzo di tflite al posto di onnx per compatibilità Raspberry Pi 2
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
    
    try:
        while True: #SENTINELLA
            pcm = coda_mic.get()
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            prediction = oww_model.predict(audio_data)
            
            if prediction.get(WAKE_WORD, 0) > 0.5:
                print("\n✨ Parola d'ordine rilevata! Modalità conversazione ATTIVA. ✨")
                oww_model.reset()
                
                # Pulizia rapida del buffer acustico
                time.sleep(0.1)
                while not coda_mic.empty():
                    coda_mic.get_nowait()
                
                in_active_conversation = True
                initial_barge_frames = []
                
                while in_active_conversation:

                    audio_path = record_dynamic_audio(oww_model) # funzione BLOCCANTE che registra l'utente
                        
                    if not audio_path: #se l'utemte è stato zitto per 6 secondi torniamo in standby
                        break
                    
                        
                    print("🧠 Trascrizione in corso...")
                    with open(COMMAND_AUDIO_PATH, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1", 
                            file=audio_file,
                            language="it",
                            temperature=0.0,
                            prompt="Comandi vocali per assistente domestico Jarvis in italiano. Nessun rumore di fondo."
                        )
                    user_text = transcription.text.strip()
                    if not user_text:
                        continue
                        
                    print(f"👤 Tu: {user_text}")
                    
                    if "hey jarvis" in user_text.lower() and len(user_text.split()) <= 4:
                        print("🤖 Comando di chiusura vocale riconosciuto. Torno in standby.")
                        break
                    
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    print("🧠 Elaborazione risposta...")
                    response = client.chat.completions.create(
                        model="gpt-5.4-mini-2026-03-17",
                        messages=conversation_history
                    )
                    ai_text = response.choices[0].message.content
                    print(f"🤖 Jarvis: {ai_text}")
                    
                    conversation_history.append({"role": "assistant", "content": ai_text})
                    if len(conversation_history) > MAX_HISTORY + 1:
                        conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]
                    
                    print("🗣️ Generazione voce...")
                    with client.audio.speech.with_streaming_response.create(
                        model="gpt-4o-mini-tts",
                        voice="onyx",
                        response_format="wav",
                        input=ai_text,
                        speed=1.15,
                    ) as tts_response:
                        tts_response.stream_to_file(RESPONSE_AUDIO_PATH)
                        
                    print("🔊 Riproduzione sulle casse (pronuncia 'Hey Jarvis' per interrompere)...")
                    interrupted = play_audio_with_barge_in(
                        pa, RESPONSE_AUDIO_PATH, oww_model
                    )
                    
                    if interrupted:
                        # Se interrompi con "Hey Jarvis" mentre le casse suonano, svuotiamo
                        # istantaneamente il buffer per eliminare l'eco della risposta di Jarvis!
                        time.sleep(0.1)
                        while not coda_mic.empty():
                            coda_mic.get_nowait()
                        initial_barge_frames = []
                        print("\n👂 Prontissimo! Dimmi pure il nuovo comando...")
                        continue
                    else:
                        time.sleep(0.2)
                        print("\n👂 In attesa del prossimo turno (o pronuncia 'Hey Jarvis' per uscire)...")
                        
                oww_model.reset()
                print("\n🤖 Torno in STANDBY. In attesa di 'Hey Jarvis'...")
                    
    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

if __name__ == "__main__":
    run_voice_assistant()