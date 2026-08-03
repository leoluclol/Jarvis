import collections
import os
import wave
import time
import queue
import pyaudio
import numpy as np
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
pa = pyaudio.PyAudio()
time.sleep(3)  # Attendi che il microfono sia pronto

WAKE_WORD = "hey_jarvis"
SILERO_MODEL_PATH = "silero_vad.onnx"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280  # 80ms chunks (Tassativo per openWakeWord!)
COMMAND_AUDIO_PATH = "command.wav"

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

def play_stream_with_barge_in(pa: pyaudio.PyAudio, audio_stream_iterator, oww_model: Model):
    """
    Riproduce uno stream di frammenti PCM crudi mantenendo in ascolto il microfono per il Barge-in.
    Implementa un pre-buffer per evitare i "grattamenti" causati dai rallentamenti di rete (Buffer Underflow).
    """
    OPENAI_TTS_RATE = 24000 
    
    # L'aggiunta di frames_per_buffer aiuta il driver audio del sistema operativo 
    # a gestire il flusso in modo più fluido
    out_stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=OPENAI_TTS_RATE,
        output=True,
        frames_per_buffer=2048 
    )

    interrupted = False
    audio_buffer = bytearray()
    
    SAFE_CHUNK_SIZE = 4096 
    
    # 16384 byte equivalgono a circa 340 millisecondi di audio a 24000Hz 16-bit.
    # Questo è il nostro "cuscinetto". Se senti ancora gracchiare, alzalo a 32768.
    PREBUFFER_SIZE = 16384 
    is_playing = False

    for chunk in audio_stream_iterator:
        if chunk:
            audio_buffer.extend(chunk)

        # Iniziamo a suonare solo quando il cuscinetto è pieno per attutire i cali di rete
        if not is_playing and len(audio_buffer) >= PREBUFFER_SIZE:
            is_playing = True

        if is_playing:
            while len(audio_buffer) >= SAFE_CHUNK_SIZE:
                data_to_write = bytes(audio_buffer[:SAFE_CHUNK_SIZE])
                del audio_buffer[:SAFE_CHUNK_SIZE]
                out_stream.write(data_to_write)

        # Controllo Barge-in ("Hey Jarvis") in tempo reale
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

    # A fine stream, svuota tutto l'audio rimanente nel buffer
    if not interrupted and audio_buffer:
        # Taglia l'eventuale byte dispari per evitare disallineamenti a 16-bit
        if len(audio_buffer) % 2 != 0:
            audio_buffer = audio_buffer[:-1]
        
        if len(audio_buffer) > 0:
            out_stream.write(bytes(audio_buffer))

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
                    
                    # Filtro allucinazioni a livello di Whisper (se Whisper stesso allucina il prompt dell'utente)
                    if "amara.org" in user_text.lower() or "qtss" in user_text.lower():
                        print("⚠️ Rilevata allucinazione nella trascrizione (Amara/QTSS). Ignoro l'input.")
                        continue

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

                    testo_lower = ai_text.lower()
                    if "amara.org" in testo_lower or "qtss" in testo_lower:
                        print("⚠️ Rilevata allucinazione del modello LLM (Amara.org/QTSS). Risposta ignorata.")
                        # Rimuoviamo l'ultimo comando dell'utente dalla cronologia così non rimane appeso senza risposta
                        conversation_history.pop() 
                        print("\n👂 In attesa del prossimo turno...")
                        continue

                    print(f"🤖 Jarvis: {ai_text}")
                    
                    conversation_history.append({"role": "assistant", "content": ai_text})
                    if len(conversation_history) > MAX_HISTORY + 1:
                        conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]
                    
                    print("🗣️ Generazione voce e riproduzione streaming (pronuncia 'Hey Jarvis' per interrompere)...")
                    
                    # Usa il context manager e ottieni la risposta streaming da OpenAI TTS
                    with client.audio.speech.with_streaming_response.create(
                        model="gpt-4o-mini-tts", # NON modificare, il modello esiste davvero
                        voice="onyx",
                        response_format="pcm", # Bit grezzi
                        input=ai_text,
                        speed=1.3,
                    ) as tts_response:
                        
                        interrupted = play_stream_with_barge_in(
                            pa, 
                            tts_response.iter_bytes(chunk_size=4096), 
                            oww_model
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