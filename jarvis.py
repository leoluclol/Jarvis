import collections
import os
import wave
import time
import queue
import pyaudio
import numpy as np
import urllib.request
import onnxruntime as ort
from dotenv import load_dotenv
from openai import OpenAI
import openwakeword
from openwakeword.model import Model

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WAKE_WORD = "hey_jarvis"
SILERO_MODEL_PATH = "silero_vad.onnx"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # 80ms chunks (Tassativo per openWakeWord!)
COMMAND_AUDIO_PATH = "command.wav"
RESPONSE_AUDIO_PATH = "response.wav"

# VAD (Voice Activity Detection) Settings
SILENCE_TIMEOUT_VAD = 1.5  
SILENCE_TIMEOUT = 5.0      
VAD_THRESHOLD = 0.5    

# Conversation Memory 
MAX_HISTORY = 6 
conversation_history = [
    {"role": "system", "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."}
]

client = OpenAI(api_key=OPENAI_API_KEY)
coda_mic = queue.Queue()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def mic_callback(in_data, frame_count, time_info, status):
    coda_mic.put(in_data)
    return (None, pyaudio.paContinue)

def is_speech_onnx(audio_chunk: bytes, vad_session, state: np.ndarray, context: np.ndarray):
    if not audio_chunk:
        return False, state, context
    
    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    
    window_size = 512
    is_speech_detected = False
    
    for i in range(0, len(audio_float32) - window_size + 1, window_size):
        window_numpy = audio_float32[i : i + window_size]
        window_tensor = np.expand_dims(window_numpy, axis=0)
        
        # Aggiunge i 64 campioni di contesto precedenti
        x = np.concatenate([context, window_tensor], axis=1)
        
        inputs = {
            'input': x,
            'sr': np.array(RATE, dtype=np.int64),
            'state': state
        }
        
        out, state = vad_session.run(None, inputs)
        prob = out[0][0]
        
        # Aggiorna il contesto per il prossimo loop
        context = window_tensor[:, -64:]
        
        if prob >= VAD_THRESHOLD:
            is_speech_detected = True
            
    # Ritorna sempre alla fine del blocco per mantenere sincronizzato lo stato RNN
    return is_speech_detected, state, context

def record_dynamic_audio(oww_model, vad_session):
    print("\n🎙️ Ascoltando... (stai zitto per annullare)")
    
    # Inizializza ENTRAMBI: stato interno (RNN) e context (CNN)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    
    start_time = time.time()
    frames = []
    has_spoken = False
    
    silent_chunks = 0
    max_silent_chunks = int((SILENCE_TIMEOUT_VAD * RATE) / CHUNK)
    
    # RING BUFFER: Memorizza gli ultimi 4 chunk (~320 millisecondi)
    # Quando l'utente inizia a parlare, recuperiamo questo audio per non tagliare la prima sillaba.
    pre_speech_buffer = collections.deque(maxlen=4)  # 4 chunk da 80ms = 320ms
    
    while True:
        pcm = coda_mic.get()
        
        if not has_spoken:
            if time.time() - start_time > SILENCE_TIMEOUT:
                print("⏳ Timeout: Nessuna parola rilevata, torno in standby.")
                return None
        
        # Passa e ricevi sia lo state che il context ad ogni iterazione
        is_current_speech, state, context = is_speech_onnx(pcm, vad_session, state, context)
        
        if is_current_speech:
            if not has_spoken:
                # IL VAD SI È APPENA ATTIVATO!
                has_spoken = True
                # Riversiamo il ring buffer dentro frames per recuperare l'inizio della parola
                frames.extend(pre_speech_buffer)
                
            frames.append(pcm)
            silent_chunks = 0
        else:
            if has_spoken:
                frames.append(pcm)
                silent_chunks += 1
                if silent_chunks >= max_silent_chunks:
                    print("🛑 Fine del discorso rilevata.")
                    break
            else:
                # Se non ha ancora parlato, continuiamo a salvare il silenzio/rumore 
                # di fondo nel buffer temporaneo, scartando i chunk più vecchi di 400ms.
                pre_speech_buffer.append(pcm)

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
    
    print(f"⚙️ Inizializzazione openWakeWord ('{WAKE_WORD}')...")
    oww_model = Model(
        wakeword_models=[WAKE_WORD],
        inference_framework="onnx"
    )
    
    print("🧠 Caricamento modello neurale Silero VAD (ONNX runtime)...")
    vad_session = ort.InferenceSession(SILERO_MODEL_PATH)
    
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
                
                time.sleep(0.1)
                while not coda_mic.empty():
                    coda_mic.get_nowait()
                
                in_active_conversation = True
                
                while in_active_conversation:
                    audio_path = record_dynamic_audio(oww_model, vad_session) 
                        
                    if not audio_path: 
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
                        model="gpt-4o-mini", # Aggiornato ai modelli openai attuali
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
                        speed=1.3,
                    ) as tts_response:
                        tts_response.stream_to_file(RESPONSE_AUDIO_PATH)
                        
                    print("🔊 Riproduzione sulle casse (pronuncia 'Hey Jarvis' per interrompere)...")
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
                    
    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

if __name__ == "__main__":
    run_voice_assistant()