import os
import wave
import time
import queue
import pyaudio
import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
import openwakeword
from openwakeword.model import Model
import silero_vad

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
SILENCE_TIMEOUT = 1.5  # Secondi di silenzio prima di chiudere il comando
VAD_THRESHOLD = 0.5    # 50% neural network confidence per la voce umana
MIN_SPEECH_CHUNKS = 3  # ~240ms di voce continua richiesti per iniziare a registrare

# Conversation Memory (Max 3 coppie = 6 messaggi)
MAX_HISTORY = 6 
conversation_history = [
    {"role": "system", "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."}
]

client = OpenAI(api_key=OPENAI_API_KEY)

# Coda Asincrona Thread-Safe per l'audio del microfono
coda_mic = queue.Queue()

# Inizializzazione Modello Neurale Silero VAD
print("🧠 Caricamento modello neurale Silero VAD...")
silero_model = silero_vad.load_silero_vad()

def mic_callback(in_data, frame_count, time_info, status):
    """
    Callback di PortAudio in background.
    Cattura l'audio del microfono e lo mette nella coda senza bloccare il programma principale.
    """
    coda_mic.put(in_data)
    return (None, pyaudio.paContinue)

def is_speech(audio_chunk: bytes) -> bool:
    """
    Verifica se un frammento audio contiene voce umana usando Silero VAD.
    Richiede tassativamente tensori PyTorch da 512 campioni (32ms a 16kHz).
    """
    if not audio_chunk:
        return False
    
    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    window_size = 512
    for i in range(0, len(audio_float32) - window_size + 1, window_size):
        window_numpy = audio_float32[i:i + window_size]
        window_tensor = torch.from_numpy(window_numpy)
        prob = silero_model(window_tensor, RATE).item()
        if prob >= VAD_THRESHOLD:
            return True
            
    return False

def record_dynamic_audio(oww_model: Model, initial_frames: list = None):
    """Registra audio dinamicamente usando Silero VAD ed evitando rumori di fondo brevi."""
    print("\n🎙️ Ascoltando... (parla quanto vuoi, fai una pausa per inviare)")
    frames = initial_frames if initial_frames else []

    has_spoken = len(frames) > 0
    silent_chunks = 0
    consecutive_speech_chunks = 0 

    max_silent_chunks = int((SILENCE_TIMEOUT * RATE) / CHUNK)

    while True:
        pcm = coda_mic.get()
        audio_data = np.frombuffer(pcm, dtype=np.int16)
        frames.append(pcm)

        if is_speech(pcm):
            consecutive_speech_chunks += 1
            if consecutive_speech_chunks >= MIN_SPEECH_CHUNKS:
                has_spoken = True
                silent_chunks = 0
        else:
            consecutive_speech_chunks = 0 
            if has_spoken:
                silent_chunks += 1

        # Controllo Wake Word per uscita anticipata durante la registrazione
        prediction = oww_model.predict(audio_data)
        if prediction.get(WAKE_WORD, 0) > 0.5:
            print("🛑 Parola d'ordine rilevata durante l'ascolto: Ritorno in standby.")
            return None, True

        if has_spoken and silent_chunks >= max_silent_chunks:
            print("🛑 Fine del discorso rilevata.")
            break

        if not has_spoken and len(frames) > int((10.0 * RATE) / CHUNK):
            print("⏳ Nessuna voce rilevata. Annullamento.")
            return None, False

    with wave.open(COMMAND_AUDIO_PATH, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return COMMAND_AUDIO_PATH, False

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
    
    print(f"⚙️ Inizializzazione openWakeWord ('{WAKE_WORD}')...")
    oww_model = Model(
        wakeword_models=[WAKE_WORD],
        inference_framework="onnx"
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
        while True:
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

                last_activity_time = time.time()
                
                while in_active_conversation:
                    # Controlliamo il timeout dei 6 secondi prima di ogni registrazione
                    if time.time() - last_activity_time > 6.0:
                        print("\n⏳ 6 secondi di inattività. Ritorno in standby automaticamente.")
                        break

                    audio_path, go_to_sleep = record_dynamic_audio(
                        oww_model, initial_frames=initial_barge_frames
                    )
                    
                    # Se rileva Hey Jarvis durante l'ascolto, usciamo comunque
                    if go_to_sleep:
                        break
                        
                    if not audio_path:
                        # Qui il timeout è già gestito da record_dynamic_audio, 
                        # ma se non abbiamo audio, aggiorniamo il timer
                        last_activity_time = time.time()
                        continue
                    
                        
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
                        model="gpt-4o-mini",
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

                    last_activity_time = time.time()
                        
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