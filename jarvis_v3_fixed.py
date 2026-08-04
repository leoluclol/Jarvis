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
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model
import socket
import io
import requests

# --- FIX #1: forza le connessioni su IPv4 ---
# Questa macchina non ha IPv6 funzionante: ad ogni prima connessione (cache DNS
# fredda) getaddrinfo chiede anche il record AAAA, che puo' bloccarsi per secondi
# in attesa del timeout DNS prima di ripiegare su IPv4. Saltiamo del tutto AAAA.
import urllib3.util.connection as urllib3_conn
urllib3_conn.allowed_gai_family = lambda: socket.AF_INET

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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

openai_session = requests.Session()
openai_session.headers.update({
    "Authorization": f"Bearer {OPENAI_API_KEY}"
})

# --- FIX #3: retry automatico su connessioni keep-alive morte ---
# Tra un turno e l'altro il socket riusato puo' venire chiuso di nascosto dal
# peer/NAT (connessione "half-open"): l'upload successivo scrive nel vuoto e va in
# "write operation timed out". Con un Retry che copre anche il POST, urllib3 se ne
# accorge e ripete la richiesta su una connessione FRESCA, senza stallo visibile.
# Ri-caricare lo stesso audio e' sicuro: la trascrizione non ha effetti collaterali.
try:
    _retry = Retry(
        total=3,
        connect=3,
        read=2,
        status=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),  # includi POST nei retry
    )
except TypeError:
    # urllib3 < 1.26 usa il vecchio nome del parametro
    _retry = Retry(
        total=3,
        connect=3,
        read=2,
        status=2,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        method_whitelist=frozenset(["GET", "POST"]),
    )
_adapter = HTTPAdapter(max_retries=_retry, pool_maxsize=4)
openai_session.mount("https://", _adapter)
openai_session.mount("http://", _adapter)

coda_mic = queue.Queue()

# --- FIX #2: "riscalda" la connessione all'avvio ---
# Facciamo una richiesta cheap subito, mentre i modelli si stanno caricando, cosi'
# DNS + TCP + TLS avvengono ora e la connessione resta nel pool. La prima
# trascrizione reale riusa il socket gia' caldo invece di pagare il cold start.
def warmup_connection():
    t0 = time.time()
    try:
        openai_session.get("https://api.openai.com/v1/models", timeout=5)
        print(f"🔥 Connessione OpenAI riscaldata in {time.time() - t0:.2f}s.")
    except Exception as e:
        # La risposta non ci interessa: serve solo a primare DNS/TCP/TLS.
        print(f"🔥 Warm-up connessione fallito ({e}) dopo {time.time() - t0:.2f}s (proseguo comunque).")

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
            # Leggiamo dal dispositivo
            pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
        except OSError:
            continue

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

    audio_buffer_io = io.BytesIO()

    with wave.open(audio_buffer_io, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    audio_buffer_io.seek(0) # Riporta il cursore all'inizio del file virtuale
    return audio_buffer_io

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

    # Riscaldiamo la connessione mentre l'hardware audio si inizializza.
    warmup_connection()

    pa = pyaudio.PyAudio()

    audio_stream = pa.open(
        rate=RATE, channels=CHANNELS, format=FORMAT,
        input=True, frames_per_buffer=CHUNK
    )
    audio_stream.start_stream()

    print(f"\n🤖 Jarvis è in STANDBY. Pronuncia \"Hey Jarvis\" per attivare la conversazione.")

    try:
        while True:
            # Non usiamo più coda_mic.get(), ma leggiamo direttamente dal device
            try:
                # exception_on_overflow=False previene crash se il Pi perde un frame
                pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
            except OSError as e:
                # In caso di errori hardware temporanei, ignora e prosegui
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
                        # Usiamo la sessione leggera per mandare l'audio in RAM direttamente a OpenAI
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
                            timeout=(3.05, 15)  # (connect, read): un connect lento fallisce presto invece di bloccare
                        )

                        response.raise_for_status() # Controlla se ci sono stati errori HTTP
                        user_text = response.json().get("text", "").strip()

                        # Timer diagnostico: la dimensione dell'audio aiuta a distinguere
                        # uno stallo di rete (audio piccolo, tempo alto) da un audio lungo.
                        audio_kb = len(audio_io.getbuffer()) // 1024
                        print(f"⏱️ Trascrizione completata in {time.time() - t0:.2f}s (audio {audio_kb}KB).")

                    except requests.HTTPError:
                        print(response.status_code)
                        print(response.text)
                    except Exception as e:
                        print(e)
                        audio_stream.start_stream()
                        continue

                    print(f"👤 Tu: {user_text}")

                    # --- INIZIO MODIFICA: Controllo comandi di spegnimento ---
                    # Rimuoviamo la punteggiatura e mettiamo tutto in minuscolo
                    text_clean = re.sub(r'[^\w\s]', '', user_text.lower()).strip()

                    stop_commands = ["stop", "basta", "stai zitto", "zitto", "fermati", "spegniti"]

                    # Controlliamo se la frase contiene una di queste parole e se è composta da 4 parole o meno
                    is_stop_command = any(cmd in text_clean for cmd in stop_commands) and len(text_clean.split()) <= 4

                    if is_stop_command:
                        print("🤖 Comando di spegnimento riconosciuto.")
                        break # Rompe il ciclo in_active_conversation e torna allo standby
                    # --- FINE MODIFICA ---

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
                        mww_features
                    )

                    tts_response.close()

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
