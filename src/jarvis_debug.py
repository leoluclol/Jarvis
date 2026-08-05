import collections
import os
import wave
import time
import queue
import threading
import pyaudio
import numpy as np
import onnxruntime as ort
import re
from dotenv import load_dotenv
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model
import socket
import io
import requests

# --- MODIFICA [DEBUG]: Importazioni per il monitoraggio ---
import psutil
import tracemalloc

# ==========================================
# PROJECT PATHS
# ==========================================
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT_DIR, "models")
AUDIO_DIR = os.path.join(ROOT_DIR, "audio")

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WAKE_WORD_MODEL_NAME = "hey_jarvis"
SILERO_MODEL_PATH = os.path.join(MODEL_DIR, "silero_vad.onnx")

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 160  # 10ms chunks per microWakeWord
COMMAND_AUDIO_PATH = os.path.join(AUDIO_DIR, "command.wav")

# VAD Settings
SILENCE_TIMEOUT_VAD = 1.5  
SILENCE_TIMEOUT = 5.0      
VAD_THRESHOLD = 0.5 
ABSOLUTE_MAX_RECORD_TIME = 30.0 # [SICUREZZA] Limite massimo per evitare RAM leak

# Conversation Memory 
MAX_HISTORY = 6 
conversation_history = [
    {"role": "system", "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."}
]

openai_session = requests.Session()
openai_session.headers.update({
    "Authorization": f"Bearer {OPENAI_API_KEY}"
})

coda_mic = queue.Queue()

# ==========================================
# HELPER FUNCTIONS
# ==========================================

# --- MODIFICA [DEBUG]: Thread di monitoraggio risorse ---
def monitor_resources():
    """Stampa CPU e RAM usata dal processo ogni 10 secondi."""
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None) # Inizializza
    
    while True:
        time.sleep(10)
        mem_mb = process.memory_info().rss / (1024 * 1024)
        cpu_usage = process.cpu_percent(interval=None)
        print(f"\n\033[93m[DEBUG] Utilizzo Sistema -> CPU: {cpu_usage:.1f}% | RAM: {mem_mb:.2f} MB\033[0m")

def init_wakeword():
    """Ricrea da zero il modello e le feature per cancellare lo stato della memoria RNN/CNN."""
    try:
        m = MicroWakeWord.from_builtin(Model.HEY_JARVIS)
    except AttributeError:
        m = MicroWakeWord.from_builtin(WAKE_WORD_MODEL_NAME)
    return m, MicroWakeWordFeatures()

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

def record_dynamic_audio(vad_session, audio_stream):
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
        try:
            pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
        except OSError:
            continue
            
        vad_buffer.extend(pcm)
        
        elapsed = time.time() - start_time
        
        # Aggiunto \n per andare a capo e non rompere la "forma d'onda visiva"
        if not has_spoken and elapsed > SILENCE_TIMEOUT:
            print("\n⏳ Timeout: Nessuna parola rilevata, torno in standby.")
            return None
        
        if elapsed > ABSOLUTE_MAX_RECORD_TIME:
            print(f"\n🛑 [SICUREZZA] Limite max ({ABSOLUTE_MAX_RECORD_TIME}s) raggiunto. Interruzione.")
            break
                
        if has_spoken:
            frames.append(pcm)
        else:
            pre_speech_buffer.append(pcm)
        
        while len(vad_buffer) >= 1024:
            vad_chunk = bytes(vad_buffer[:1024])
            del vad_buffer[:1024]
            
            is_speech, state, context = is_speech_onnx(vad_chunk, vad_session, state, context)
            
            # --- MODIFICA [DEBUG]: Flag visivo del VAD ---
            if is_speech:
                print("█", end="", flush=True) # Voce
            else:
                print(".", end="", flush=True) # Silenzio
            # ---------------------------------------------
            
            if is_speech:
                if not has_spoken:
                    has_spoken = True
                    frames = list(pre_speech_buffer) + frames
                silent_windows = 0
            else:
                if has_spoken:
                    silent_windows += 1

        if has_spoken and silent_windows >= max_silent_windows:
            print("\n🛑 Fine del discorso rilevata.") # Aggiunto \n per la pulizia del terminale
            break

    audio_buffer_io = io.BytesIO()
    
    with wave.open(audio_buffer_io, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))
        
    audio_buffer_io.seek(0)
    return audio_buffer_io

def play_stream_with_barge_in(pa: pyaudio.PyAudio, audio_stream_iterator, mww: MicroWakeWord, mww_features: MicroWakeWordFeatures, audio_stream):
    OPENAI_TTS_RATE = 24000

    while not coda_mic.empty():
        try: coda_mic.get_nowait()
        except queue.Empty: break

    out_stream = pa.open(
        format=pyaudio.paInt16, channels=1,
        rate=OPENAI_TTS_RATE, output=True,
        frames_per_buffer=2048
    )

    stop_reader = threading.Event()

    def mic_reader():
        while not stop_reader.is_set():
            try:
                pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                continue
            coda_mic.put(pcm)

    reader_thread = threading.Thread(target=mic_reader, daemon=True)
    reader_thread.start()

    interrupted = False
    audio_buffer = bytearray()

    SAFE_CHUNK_SIZE = 4096
    PREBUFFER_SIZE = 16384
    is_playing = False
    playback_start_time = 0
    GRACE_PERIOD = 1.0

    def detect_bargein():
        while not coda_mic.empty():
            try:
                pcm = coda_mic.get_nowait()
            except queue.Empty:
                break
            for features in mww_features.process_streaming(pcm):
                prob = mww.process_streaming_prob(features)
                if is_playing and (time.time() - playback_start_time > GRACE_PERIOD) and prob > 0.6:
                    print(f"\n⚡ BARGE-IN RILEVATO! (Prob: {prob:.2f})... ⚡")
                    return True
        return False

    try:
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
                    if detect_bargein():
                        interrupted = True
                        break

            if not interrupted and detect_bargein():
                interrupted = True

            if interrupted:
                break

        if not interrupted:
            if not is_playing:
                is_playing = True
                playback_start_time = time.time()
            while len(audio_buffer) >= 2 and not interrupted:
                n = min(SAFE_CHUNK_SIZE, len(audio_buffer))
                if n % 2 != 0:
                    n -= 1
                if n <= 0:
                    break
                out_stream.write(bytes(audio_buffer[:n]))
                del audio_buffer[:n]
                if detect_bargein():
                    interrupted = True
    finally:
        stop_reader.set()
        reader_thread.join(timeout=1.0)
        out_stream.stop_stream()
        out_stream.close()
        while not coda_mic.empty():
            try: coda_mic.get_nowait()
            except queue.Empty: break

    return interrupted

def run_voice_assistant():
    global conversation_history
    
    # --- MODIFICA [DEBUG]: Inizio tracciamento e monitoraggio ---
    tracemalloc.start()
    threading.Thread(target=monitor_resources, daemon=True).start()
    
    print(f"⚙️ Inizializzazione microWakeWord ('{WAKE_WORD_MODEL_NAME}')...")
    mww, mww_features = init_wakeword()
    
    print("🧠 Caricamento modello neurale Silero VAD (ONNX runtime)...")
    vad_session = ort.InferenceSession(SILERO_MODEL_PATH)
    
    pa = pyaudio.PyAudio()
    
    audio_stream = pa.open(
        rate=RATE, channels=CHANNELS, format=FORMAT,
        input=True, frames_per_buffer=CHUNK
    )
    audio_stream.start_stream()
    
    print(f"\n🤖 Jarvis è in STANDBY. Pronuncia \"Hey Jarvis\" per attivare la conversazione.")
    
    try:
        while True: 
            try:
                pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                continue 
            
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
                    audio_io = record_dynamic_audio(vad_session, audio_stream) 
                    
                    if not audio_io: 
                        break

                    audio_stream.stop_stream()
                    
                    print("🧠 Trascrizione in corso")
                    t0 = time.time()
                    
                    try:
                        files = {
                            "file": ("command.wav", audio_io, "audio/wav")
                        }
                        data = {
                            "model": "whisper-1",
                            "language": "it",
                            "temperature": "0.0"
                        }
                        
                        response = openai_session.post(
                            "https://api.openai.com/v1/audio/transcriptions",
                            files=files,
                            data=data,
                            timeout=20.0
                        )
                        
                        response.raise_for_status() 
                        user_text = response.json().get("text", "").strip()
                        
                        print(f"⏱️ Trascrizione completata in {time.time() - t0:.2f} secondi.")
                        
                    except requests.HTTPError:
                        print(response.status_code)
                        print(response.text)
                        break # Se fallisce, meglio uscire dal loop invece di crashare
                    except Exception as e:
                        print(e)
                        audio_stream.start_stream()
                        continue
                        
                    print(f"👤 Tu: {user_text}")
                    
                    text_clean = re.sub(r'[^\w\s]', '', user_text.lower()).strip()
                    stop_commands = ["stop", "basta", "stai zitto", "zitto", "fermati", "spegniti"]
                    
                    is_stop_command = any(cmd in text_clean for cmd in stop_commands) and len(text_clean.split()) <= 4
                    
                    if is_stop_command:
                        print("🤖 Comando di spegnimento riconosciuto.")
                        break 
                    
                    conversation_history.append({"role": "user", "content": user_text})
                    
                    print("🧠 Elaborazione risposta...")
                    response = openai_session.post(
                        "https://api.openai.com/v1/chat/completions",
                        json={
                            "model": "gpt-4o-mini",
                            "messages": conversation_history
                        },
                        timeout=60
                    )

                    response.raise_for_status()

                    ai_text = response.json()["choices"][0]["message"]["content"]

                    if "amara.org" in ai_text.lower() or "qtss" in ai_text.lower():
                        conversation_history.pop() 
                        continue

                    print(f"🤖 Jarvis: {ai_text}")
                    conversation_history.append({"role": "assistant", "content": ai_text})
                    if len(conversation_history) > MAX_HISTORY + 1:
                        conversation_history = [conversation_history[0]] + conversation_history[-MAX_HISTORY:]

                    audio_stream.start_stream()
                    
                    print("🗣️ Riproduzione in corso (pronuncia 'Hey Jarvis' per interrompere)...")
                    
                    tts_response = openai_session.post(
                        "https://api.openai.com/v1/audio/speech",
                        json={
                            "model": "gpt-4o-mini-tts",
                            "voice": "onyx",
                            "input": ai_text,
                            "response_format": "pcm",
                            "speed": 1.3
                        },
                        stream=True,
                        timeout=120
                    )

                    tts_response.raise_for_status()

                    interrupted = play_stream_with_barge_in(
                        pa,
                        tts_response.iter_content(chunk_size=4096),
                        mww,
                        mww_features,
                        audio_stream
                    )

                    tts_response.close()
                    
                    if interrupted:
                        mww, mww_features = init_wakeword()
                        print("\n👂 Prontissimo! Dimmi pure il nuovo comando...")
                        continue
                    else:
                        print("\n👂 In attesa del prossimo turno...")
                
                # --- MODIFICA [DEBUG]: Stampa analisi della memoria a fine interazione ---
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("\n\033[96m[DEBUG] Analisi Memoria - Top 3 line colpevoli dell'utilizzo RAM:\033[0m")
                for stat in top_stats[:3]:
                    print(f"\033[96m{stat}\033[0m")
                
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