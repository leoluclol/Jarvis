import collections
import os
import wave
import time
import queue
import pyaudio
import numpy as np
import onnxruntime as ort
import re  # Aggiunto per pulire la punteggiatura
from dotenv import load_dotenv
from openai import OpenAI
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model
import socket
import httpx

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Forza IPv4 per tutte le connessioni httpx per evitare black-hole IPv6
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

WAKE_WORD_MODEL_NAME = "hey_jarvis"
SILERO_MODEL_PATH = "silero_vad.onnx"

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 160  # 10ms chunks per microWakeWord
COMMAND_AUDIO_PATH = "/dev/shm/command.wav"

# VAD Settings
SILENCE_TIMEOUT_VAD = 1.5  
SILENCE_TIMEOUT = 5.0      
VAD_THRESHOLD = 0.5    

# Conversation Memory 
MAX_HISTORY = 6 
conversation_history = [
    {"role": "system", "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."}
]

http_client = httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0))
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
coda_mic = queue.Queue()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def init_wakeword():
    """Ricrea da zero il modello e le feature per cancellare lo stato della memoria RNN/CNN."""
    try:
        m = MicroWakeWord.from_builtin(Model.HEY_JARVIS)
    except AttributeError:
        m = MicroWakeWord.from_builtin(WAKE_WORD_MODEL_NAME)
    return m, MicroWakeWordFeatures()

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
        
        x = np.concatenate([context, window_tensor], axis=1)
        inputs = {'input': x, 'sr': np.array(RATE, dtype=np.int64), 'state': state}
        out, state = vad_session.run(None, inputs)
        prob = out[0][0]
        
        context = window_tensor[:, -64:]
        if prob >= VAD_THRESHOLD:
            is_speech_detected = True
            
    return is_speech_detected, state, context

def record_dynamic_audio(vad_session):
    print("\n🎙️ Ascoltando... (stai zitto per annullare)")
    
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    
    start_time = time.time()
    frames = []
    has_spoken = False
    
    vad_buffer = bytearray()
    silent_windows = 0
    max_silent_windows = int(SILENCE_TIMEOUT_VAD / 0.032)
    pre_speech_buffer = collections.deque(maxlen=32) 
    
    while True:
        pcm = coda_mic.get()
        vad_buffer.extend(pcm)
        
        if not has_spoken and time.time() - start_time > SILENCE_TIMEOUT:
            print("⏳ Timeout: Nessuna parola rilevata, torno in standby.")
            return None
                
        if has_spoken:
            frames.append(pcm)
        else:
            pre_speech_buffer.append(pcm)
        
        while len(vad_buffer) >= 1024:
            vad_chunk = bytes(vad_buffer[:1024])
            del vad_buffer[:1024]
            
            is_speech, state, context = is_speech_onnx(vad_chunk, vad_session, state, context)
            
            if is_speech:
                if not has_spoken:
                    has_spoken = True
                    frames = list(pre_speech_buffer) + frames
                silent_windows = 0
            else:
                if has_spoken:
                    silent_windows += 1

        if has_spoken and silent_windows >= max_silent_windows:
            print("🛑 Fine del discorso rilevata.")
            break

    with wave.open(COMMAND_AUDIO_PATH, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return COMMAND_AUDIO_PATH

def play_stream_with_barge_in(pa: pyaudio.PyAudio, audio_stream_iterator, mww: MicroWakeWord, mww_features: MicroWakeWordFeatures):
    OPENAI_TTS_RATE = 24000 
    
    while not coda_mic.empty():
        try: coda_mic.get_nowait()
        except queue.Empty: break

    out_stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=OPENAI_TTS_RATE, output=True,
        frames_per_buffer=2048 
    )

    interrupted = False
    audio_buffer = bytearray()
    
    SAFE_CHUNK_SIZE = 4096 
    PREBUFFER_SIZE = 16384 
    is_playing = False
    playback_start_time = 0
    GRACE_PERIOD = 1.0  

    for chunk in audio_stream_iterator:
        if chunk:
            audio_buffer.extend(chunk)

        if not is_playing and len(audio_buffer) >= PREBUFFER_SIZE:
            is_playing = True
            playback_start_time = time.time()

        if is_playing:
            while len(audio_buffer) >= SAFE_CHUNK_SIZE:
                out_stream.write(bytes(audio_buffer[:SAFE_CHUNK_SIZE]))
                del audio_buffer[:SAFE_CHUNK_SIZE]

        while not coda_mic.empty():
            try:
                pcm = coda_mic.get_nowait()
            except queue.Empty:
                break
                
            for features in mww_features.process_streaming(pcm):
                prob = mww.process_streaming_prob(features)
                
                if is_playing and (time.time() - playback_start_time > GRACE_PERIOD):
                    if prob > 0.75:
                        print(f"\n⚡ BARGE-IN RILEVATO! (Prob: {prob:.2f})... ⚡")
                        interrupted = True
                        break
            
            if interrupted:
                break
                
        if interrupted:
            break

    if not interrupted and audio_buffer:
        if len(audio_buffer) % 2 != 0:
            audio_buffer = audio_buffer[:-1]
        if len(audio_buffer) > 0:
            out_stream.write(bytes(audio_buffer))

    out_stream.stop_stream()
    out_stream.close()
    
    while not coda_mic.empty():
        try: coda_mic.get_nowait()
        except queue.Empty: break
            
    return interrupted

def run_voice_assistant():
    global conversation_history
    
    print(f"⚙️ Inizializzazione microWakeWord ('{WAKE_WORD_MODEL_NAME}')...")
    mww, mww_features = init_wakeword()
    
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
            while not coda_mic.empty():
                try: coda_mic.get_nowait()
                except queue.Empty: break

            pcm = coda_mic.get()
            
            wakeword_detected = False
            for features in mww_features.process_streaming(pcm):
                 if mww.process_streaming_prob(features) > 0.75: 
                     wakeword_detected = True
                     break
            
            if wakeword_detected:
                print("\n✨ Parola d'ordine rilevata! Modalità conversazione ATTIVA. ✨")
                
                mww, mww_features = init_wakeword()
                
                while not coda_mic.empty():
                    try: coda_mic.get_nowait()
                    except queue.Empty: break
                
                in_active_conversation = True
                
                while in_active_conversation:
                    audio_path = record_dynamic_audio(vad_session) 
                        
                    if not audio_path: 
                        break
                    
                    print("🧠 Trascrizione in corso...")
                    
                    user_text = ""
                    try:
                        with open(COMMAND_AUDIO_PATH, "rb") as audio_file:
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1", 
                                file=audio_file, 
                                language="it", 
                                temperature=0.0
                            )
                        user_text = transcription.text.strip()
                    except httpx.TimeoutException:
                        print("❌ ERRORE: La connessione a Whisper è andata in timeout! (Rete lenta)")
                        continue # Torna ad ascoltare
                    except Exception as e:
                        print(f"❌ ERRORE IMPREVISTO WHISPER: {e}")
                        continue
                        
                    print(f"👤 Tu: {user_text}")
                    
                    # --- INIZIO MODIFICA: Controllo comandi di spegnimento ---
                    # Rimuoviamo la punteggiatura e mettiamo tutto in minuscolo
                    text_clean = re.sub(r'[^\w\s]', '', user_text.lower()).strip()
                    
                    stop_commands = ["stop", "basta", "stai zitto", "zitto", "fermati", "spegniti", "hey jarvis"]
                    
                    # Controlliamo se la frase contiene una di queste parole e se è composta da 4 parole o meno
                    is_stop_command = any(cmd in text_clean for cmd in stop_commands) and len(text_clean.split()) <= 4
                    
                    if is_stop_command:
                        print("🤖 Comando di spegnimento riconosciuto.")
                        break # Rompe il ciclo in_active_conversation e torna allo standby
                    # --- FINE MODIFICA ---
                    
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    print("🧠 Elaborazione risposta...")
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", messages=conversation_history
                    )
                    ai_text = response.choices[0].message.content

                    if "amara.org" in ai_text.lower() or "qtss" in ai_text.lower():
                        conversation_history.pop() 
                        continue

                    print(f"🤖 Jarvis: {ai_text}")
                    conversation_history.append({"role": "assistant", "content": ai_text})
                    if len(conversation_history) > MAX_HISTORY + 1:
                        conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]
                    
                    print("🗣️ Riproduzione in corso (pronuncia 'Hey Jarvis' per interrompere)...")
                    
                    with client.audio.speech.with_streaming_response.create(
                        model="gpt-4o-mini-tts", voice="onyx", response_format="pcm", 
                        input=ai_text, speed=1.3,
                    ) as tts_response:
                        
                        interrupted = play_stream_with_barge_in(
                            pa, tts_response.iter_bytes(chunk_size=4096), mww, mww_features
                        )
                    
                    if interrupted:
                        mww, mww_features = init_wakeword()
                        print("\n👂 Prontissimo! Dimmi pure il nuovo comando...")
                        continue
                    else:
                        print("\n👂 In attesa del prossimo turno...")
                
                mww, mww_features = init_wakeword()
                print("\n🤖 Torno in STANDBY. In attesa di 'Hey Jarvis'...")

    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

if __name__ == "__main__":
    run_voice_assistant()