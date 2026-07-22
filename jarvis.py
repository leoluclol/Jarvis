import os
import sys
import wave
import time
import queue
import pyaudio
from collections import deque
import numpy as np
from scipy.signal import lfilter, lfilter_zi
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from rhasspywake_raven import Raven, Template
from rhasspywake_raven.utils import trim_silence, buffer_to_wav
from rhasspysilence import WebRtcVadRecorder
import webrtcvad

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WAKE_WORD = "hey_jarvis"
# Raven non ha modelli pre-addestrati: la parola d'ordine è definita da 3+
# registrazioni WAV dell'utente (vedi "python jarvis.py --record").
TEMPLATE_DIR = Path(__file__).parent / "models" / "raven" / WAKE_WORD
TEMPLATE_FORMAT = "example-{n:02d}.wav"
MIN_TEMPLATES = 3

# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 960  # 60ms = 2 frame VAD da 30ms (Raven lavora su chunk da 480 campioni)
COMMAND_AUDIO_PATH = "command.wav"
RESPONSE_AUDIO_PATH = "response.wav"

# Correzioni per le stranezze di questo microfono
# 1. All'attivazione il microfono spara picchi di RMS per qualche secondo:
#    quell'audio va buttato, altrimenti finisce dentro ai template.
MIC_WARMUP_SEC = 3.0
# 2. Il microfono ha un offset (~+3000) sul segnale. Va rimosso PRIMA del VAD e
#    di Raven: un DC offset gonfia l'energia del segnale, quindi il VAD sente
#    "voce" ovunque e gli MFCC dei template risultano falsati.
#    Misurato sulle registrazioni reali, il bias NON è costante: oscilla fra
#    +1200 e +4600. Per questo la correzione insegue la media del segnale invece
#    di sottrarre un numero fisso; MIC_DC_OFFSET è solo la stima iniziale.
#    ATTENZIONE: NON sottrarre una costante ricalcolata a ogni chunk. Sembra
#    innocuo ma introduce un gradino a ogni confine di chunk (60ms), cioè un
#    click a 16 Hz che il VAD scambia per voce. Serve un filtro continuo, con
#    stato che attraversa i chunk: il classico "DC blocker"
#        y[n] = x[n] - x[n-1] + R*y[n-1]
#    R=0.999 mette il taglio a ~2.5 Hz, due decadi sotto la banda vocale.
DC_BLOCK_R = 0.999
# Sotto questo RMS (dopo la correzione) un template è praticamente silenzio.
MIN_TEMPLATE_RMS = 300.0
# "Hey Jarvis" dura ~1s: oltre questa soglia è rumore, non la parola d'ordine.
MAX_TEMPLATE_SEC = 3.0

# Raven (wake word) Settings
RAVEN_PROBABILITY_THRESHOLD = 0.5
# Lasciare 0.22 (default di Raven). Alzarlo sembra sensato, perché con lo skip
# attivo i template arrivano a 0.21 contro se stessi, ma su microfono vero è
# controproducente: misurato su 20s di stanza silenziosa,
#   0.30 -> 3 attivazioni spurie
#   0.22 -> 0
#   0.15 -> 0
# Se dal vivo Jarvis non ti sente, NON alzare questa soglia: abbassa invece
# RAVEN_SKIP_PROBABILITY_THRESHOLD (costa CPU ma migliora il match).
RAVEN_DISTANCE_THRESHOLD = 0.22
RAVEN_MINIMUM_MATCHES = 1   # quanti template devono combaciare (0 = tutti)
RAVEN_REFRACTORY_SEC = 2.0  # blocco anti-doppia-attivazione dopo un match
# Su hardware lento (Pi 2) questo evita che il DTW giri all'infinito su rumore
RAVEN_FAILED_MATCHES_TO_REFRACTORY = 10
# NON attivare la media dei template: sembra un risparmio di CPU, ma
# Template.average_templates() allinea con DTW registrazioni di durata diversa e
# il risultato non somiglia più a nessuna di esse. Misurato con test_raven.py su
# 7 template reali: media attiva 1/7 rilevati (distanza 0.68), media spenta 7/7
# (distanza 0.04). Nessuna soglia recupera la differenza.
RAVEN_AVERAGE_TEMPLATES = False
# Questo è il vero risparmio di CPU, e non costa accuratezza: se il primo
# template è chiaramente lontano, Raven salta il DTW su tutti gli altri.
# Caso peggiore (rumore che non combacia) su Pi 2, 7 template:
#   senza skip -> ~190ms per finestra da 30ms, cioè 6 core inesistenti
#   con skip   -> ~12ms, come se ci fosse un template solo
RAVEN_SKIP_PROBABILITY_THRESHOLD = 0.2

# Barge-in.
# Spento di default, e non è una resa: mentre le casse suonano, il microfono
# riprende la voce di Jarvis sovrapposta alla tua. openWakeWord reggeva perché è
# una rete addestrata con rumore ed eco; Raven confronta MFCC con DTW, quindi
# l'eco sposta le feature e la distanza esplode. Non esiste soglia che lo salvi:
# servirebbe la cancellazione d'eco, troppo per un Pi 2.
# Se lo riaccendi, l'interruzione NON usa più la parola d'ordine: si misura
# quanto è più forte il microfono rispetto all'eco delle casse.
ENABLE_BARGE_IN = False
# Il livello dell'eco non è costante (Jarvis alza e abbassa la voce), quindi si
# stima in continuo come percentile basso degli ultimi secondi invece di
# misurarlo una volta sola all'inizio.
# LIMITE NOTO: per interrompere devi iniziare a parlare DOPO che la riproduzione
# è partita. Se parli fin dal primo istante, la tua voce entra nella stima
# dell'eco e diventa il riferimento: nessuna soglia può più distinguerla.
BARGE_IN_CALIBRATION_CHUNKS = 8   # chunk minimi prima di poter decidere
BARGE_IN_WINDOW_CHUNKS = 50       # ~3s di storico per stimare l'eco
BARGE_IN_PERCENTILE = 25          # percentile "basso" = livello dell'eco
BARGE_IN_RATIO = 3.0              # quante volte sopra l'eco per contare come voce
BARGE_IN_MIN_CHUNKS = 5           # ~300ms continui, per non fermarsi a ogni tonfo

# VAD (Voice Activity Detection) Settings
SILENCE_TIMEOUT_VAD = 1.5  # Secondi di silenzio prima di chiudere il comando
SILENCE_TIMEOUT = 6.0      # Secondi di silenzio prima di annullare il comando
# Rete di sicurezza: se il rumore è tale che il VAD non vede mai la fine del
# discorso, la registrazione va chiusa lo stesso invece di crescere all'infinito.
MAX_COMMAND_SEC = 15.0
VAD_MODE = 3  # 0-3, higher is more aggressive
# webrtcvad da solo non basta: su rumore di fondo continua a votare "voce" e la
# registrazione non si chiude più. Due filtri in più:
# 1. il chunk deve stare sopra il rumore ambientale misurato all'avvio
# 2.0 (~6dB) è il punto giusto: misurato, taglia la coda di rumore esattamente
# come 2.5 o 3.0, ma in una stanza rumorosa quelli scartano anche la tua voce.
SPEECH_SNR_RATIO = 2.0     # quante volte sopra il rumore di fondo
SPEECH_MIN_RMS = 400.0     # pavimento assoluto, se la stanza è silenziosissima
# 2. non basta UN frame da 30ms: serve la maggioranza dei frame del chunk
VAD_SPEECH_FRAME_RATIO = 0.5
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

# DC blocker: coefficienti e stato che attraversa i chunk (vedi remove_dc_offset)
DC_BLOCK_B = np.array([1.0, -1.0])
DC_BLOCK_A = np.array([1.0, -DC_BLOCK_R])
dc_state = None

# Livello del rumore ambientale, misurato durante il riscaldamento del microfono
noise_floor = 0.0


def reset_vad_buffer():
    global vad_buffer
    vad_buffer = b""

def reset_dc_filter():
    """Azzera lo stato del DC blocker (si riaggancia al primo chunk successivo)."""
    global dc_state
    dc_state = None

def remove_dc_offset(pcm: bytes) -> bytes:
    """
    Toglie il bias del microfono con un filtro passa-alto del primo ordine.
    Lo stato viene mantenuto fra una chiamata e l'altra: è ciò che rende il
    segnale continuo ai confini dei chunk. Sottrarre la media del singolo chunk
    farebbe la stessa cosa "in media", ma lascerebbe un gradino ogni 60ms.
    """
    global dc_state

    if not pcm:
        return pcm

    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if dc_state is None:
        # Parte già a regime sul livello attuale, così il primo chunk non
        # contiene il transitorio di aggancio del filtro.
        dc_state = lfilter_zi(DC_BLOCK_B, DC_BLOCK_A) * x[0]

    y, dc_state = lfilter(DC_BLOCK_B, DC_BLOCK_A, x, zi=dc_state)
    return np.clip(y, -32768, 32767).astype(np.int16).tobytes()

def rms(pcm: bytes) -> float:
    """RMS di un blocco PCM 16-bit, usato per diagnosticare le registrazioni."""
    if not pcm:
        return 0.0

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    return float(np.sqrt(np.mean(np.square(samples))))

def mic_callback(in_data, frame_count, time_info, status):
    """
    Callback di PortAudio in background.
    Cattura l'audio del microfono e lo mette nella coda senza bloccare il programma principale.
    La correzione del DC offset viene applicata QUI, così tutto ciò che sta a
    valle (VAD, Raven, registrazione comandi, template) vede audio già pulito.
    """
    coda_mic.put(remove_dc_offset(in_data))
    return (None, pyaudio.paContinue)

def discard_mic_warmup():
    """
    Butta via i primi secondi di audio dopo l'apertura dello stream.
    Questo microfono genera picchi di RMS all'attivazione: se finiscono in un
    template, Raven impara il picco invece della parola d'ordine.
    """
    global noise_floor

    reset_dc_filter()

    if MIC_WARMUP_SEC <= 0:
        return

    print(f"🔥 Riscaldamento microfono ({MIC_WARMUP_SEC:.0f}s, audio scartato)...")
    deadline = time.time() + MIC_WARMUP_SEC
    levels = []
    while time.time() < deadline:
        try:
            levels.append(rms(coda_mic.get(timeout=0.1)))
        except queue.Empty:
            pass

    # Svuota anche ciò che si è accumulato mentre aspettavamo
    while not coda_mic.empty():
        coda_mic.get_nowait()

    # Il rumore di fondo si misura solo sulla coda del riscaldamento: l'inizio
    # contiene i picchi di accensione del microfono e falserebbe la mediana.
    tail = levels[len(levels) // 2:] or levels
    noise_floor = float(np.median(tail)) if tail else 0.0
    print(f"   Rumore di fondo {noise_floor:.0f} → soglia voce {speech_rms_threshold():.0f}")

def speech_rms_threshold() -> float:
    """Quanto deve essere forte un chunk per essere preso in considerazione."""
    return max(SPEECH_MIN_RMS, noise_floor * SPEECH_SNR_RATIO)

def is_speech(audio_chunk: bytes) -> bool:
    """
    Dice se un frammento audio contiene voce.

    Non si affida al solo webrtcvad: su rumore di fondo continuo vota "voce"
    abbastanza spesso da tenere aperta la registrazione all'infinito. Servono
    due condizioni insieme:
      1. il chunk deve superare il rumore ambientale misurato all'avvio;
      2. la maggioranza dei frame da 30ms deve essere voce (non uno qualsiasi).
    """
    global vad_buffer

    if not audio_chunk:
        return False

    # 1. Cancello di energia: sotto il rumore di fondo non si discute nemmeno.
    if rms(audio_chunk) < speech_rms_threshold():
        return False

    vad_buffer += audio_chunk

    if len(vad_buffer) < VAD_FRAME_BYTES:
        return False

    max_valid_length = len(vad_buffer) - (len(vad_buffer) % VAD_FRAME_BYTES)
    frames = [
        vad_buffer[start:start + VAD_FRAME_BYTES]
        for start in range(0, max_valid_length, VAD_FRAME_BYTES)
    ]
    vad_buffer = vad_buffer[max_valid_length:]

    if not frames:
        return False

    # 2. Voto di maggioranza sui frame, non "basta uno".
    votes = sum(1 for frame in frames if vad.is_speech(frame, RATE))
    return votes >= max(1, round(len(frames) * VAD_SPEECH_FRAME_RATIO))

# ==========================================
# WAKE WORD (Rhasspy Raven)
# ==========================================

def make_recorder() -> WebRtcVadRecorder:
    """
    Rilevatore di silenzio condiviso da Raven. Raven calcola MFCC/DTW SOLO sui
    chunk non silenziosi: è ciò che rende praticabile il DTW su CPU lente.
    """
    return WebRtcVadRecorder(
        vad_mode=VAD_MODE,
        sample_rate=RATE,
        chunk_size=VAD_FRAME_BYTES,
        min_seconds=0.5,
        before_seconds=1,
    )

def load_raven() -> Raven:
    """
    Costruisce il detector Raven a partire dai WAV registrati dall'utente.
    """
    wav_paths = sorted(TEMPLATE_DIR.glob("*.wav"))
    if len(wav_paths) < MIN_TEMPLATES:
        print(
            f"❌ Servono almeno {MIN_TEMPLATES} registrazioni della parola d'ordine in "
            f"{TEMPLATE_DIR} (trovate: {len(wav_paths)}).\n"
            f"   Registrale con:  python jarvis.py --record"
        )
        sys.exit(1)

    templates = [
        Raven.wav_to_template(str(p), name=p.name) for p in wav_paths
    ]

    if RAVEN_AVERAGE_TEMPLATES:
        print(f"🧬 Media di {len(templates)} template in uno solo (risparmio CPU)...")
        templates = [Template.average_templates(templates, name=WAKE_WORD)]

    return Raven(
        templates=templates,
        keyword_name=WAKE_WORD,
        recorder=make_recorder(),
        probability_threshold=RAVEN_PROBABILITY_THRESHOLD,
        distance_threshold=RAVEN_DISTANCE_THRESHOLD,
        minimum_matches=RAVEN_MINIMUM_MATCHES,
        refractory_sec=RAVEN_REFRACTORY_SEC,
        failed_matches_to_refractory=RAVEN_FAILED_MATCHES_TO_REFRACTORY,
        skip_probability_threshold=RAVEN_SKIP_PROBABILITY_THRESHOLD,
    )

def raven_detected(raven: Raven, pcm: bytes) -> bool:
    """
    True se il chunk audio contiene la parola d'ordine.
    process_chunk() ritorna la lista dei template che hanno fatto match.
    """
    return bool(raven.process_chunk(pcm))

def raven_reset(raven: Raven):
    """
    Azzera lo stato interno di Raven ed entra nel periodo refrattario, così
    l'eco della propria voce non riattiva subito il detector.
    """
    raven._reset_state()
    raven._begin_refractory()

def record_templates():
    """
    Modalità "--record": registra le WAV che DEFINISCONO la parola d'ordine.
    Ogni frase viene ritagliata automaticamente tramite VAD.
    Ctrl+C per terminare.
    """
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(TEMPLATE_DIR.glob("*.wav"))
    num_templates = len(existing)

    recorder = make_recorder()
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=RATE, channels=CHANNELS, format=FORMAT,
        input=True, frames_per_buffer=CHUNK,
        stream_callback=mic_callback
    )
    audio_stream.start_stream()
    discard_mic_warmup()

    print(
        f"\n🎤 Registrazione template in {TEMPLATE_DIR}\n"
        f"   Pronuncia la parola d'ordine, aspetta la conferma, ripeti.\n"
        f"   Servono almeno {MIN_TEMPLATES} registrazioni. Ctrl+C per finire.\n"
    )

    buffer = b""
    try:
        print(f"▶️  Template {num_templates}: parla ora...")
        recorder.start()
        while True:
            buffer += coda_mic.get()
            while len(buffer) >= VAD_FRAME_BYTES:
                frame, buffer = buffer[:VAD_FRAME_BYTES], buffer[VAD_FRAME_BYTES:]
                if not recorder.process_chunk(frame):
                    continue

                audio_bytes = trim_silence(recorder.stop())
                level = rms(audio_bytes)
                duration = len(audio_bytes) / (RATE * 2)

                # Un template troppo silenzioso o troppo corto rende Raven
                # inaffidabile: meglio dirlo subito che scoprirlo dopo.
                if level < MIN_TEMPLATE_RMS or duration < 0.3:
                    print(
                        f"⚠️  Scartato: troppo debole o troppo corto "
                        f"(RMS {level:.0f}, {duration:.2f}s). Riprova più vicino al microfono."
                    )
                    recorder.start()
                    continue

                if duration > MAX_TEMPLATE_SEC:
                    print(
                        f"⚠️  Scartato: troppo lungo ({duration:.2f}s, max {MAX_TEMPLATE_SEC:.0f}s). "
                        f"Di' solo la parola d'ordine, senza pause."
                    )
                    recorder.start()
                    continue

                path = TEMPLATE_DIR / TEMPLATE_FORMAT.format(n=num_templates)
                path.write_bytes(buffer_to_wav(audio_bytes))
                num_templates += 1
                print(f"✅ Salvato {path.name} (RMS {level:.0f}, {duration:.2f}s)")
                print(f"▶️  Template {num_templates}: parla ora...")
                recorder.start()
    except KeyboardInterrupt:
        print(f"\n🏁 Fatto: {num_templates} template in {TEMPLATE_DIR}")
        if num_templates < MIN_TEMPLATES:
            print(f"⚠️  Ne servono almeno {MIN_TEMPLATES} per far partire Jarvis.")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

# ==========================================
# PIPELINE VOCALE
# ==========================================

def record_dynamic_audio():
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
        # Preleva audio dalla coda (bloccante)
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

        # Rete di sicurezza: con rumore continuo il VAD può non vedere mai la
        # fine del discorso. Meglio troncare che registrare per sempre.
        if has_spoken and (time.time() - start_time) > MAX_COMMAND_SEC:
            print(f"✂️  Limite di {MAX_COMMAND_SEC:.0f}s raggiunto, chiudo la registrazione.")
            break

    # Scrittura su file se abbiamo parlato
    with wave.open(COMMAND_AUDIO_PATH, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(b"".join(frames))

    return COMMAND_AUDIO_PATH

def play_audio_with_barge_in(pa: pyaudio.PyAudio, file_path: str):
    """
    Riproduce la risposta sulle casse e, se ENABLE_BARGE_IN è attivo, si ferma
    quando qualcuno parla sopra.

    NON si cerca la parola d'ordine: durante la riproduzione il microfono sente
    soprattutto Jarvis, e con l'eco addosso il DTW di Raven non aggancia più
    nulla. Si misura invece il livello dell'eco nei primi chunk e si considera
    "qualcuno sta parlando" solo un audio molto più forte di quello, per un
    tempo continuato.
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
        echo_levels = deque(maxlen=BARGE_IN_WINDOW_CHUNKS)
        loud_chunks = 0
        data = wf.readframes(wav_chunk_size)

        while data:
            # 1. Scrittura audio sulle casse esterne
            out_stream.write(data)

            # 2. Confronto del microfono con il livello dell'eco
            while not coda_mic.empty():
                pcm = coda_mic.get_nowait()

                if not ENABLE_BARGE_IN:
                    continue

                level = rms(pcm)
                echo_levels.append(level)

                # Servono un po' di chunk prima di poter dire cos'è "l'eco".
                if len(echo_levels) < BARGE_IN_CALIBRATION_CHUNKS:
                    continue

                echo_floor = max(
                    float(np.percentile(echo_levels, BARGE_IN_PERCENTILE)), 1.0
                )
                if level > echo_floor * BARGE_IN_RATIO:
                    loud_chunks += 1
                    if loud_chunks >= BARGE_IN_MIN_CHUNKS:
                        print("\n⚡ BARGE-IN: qualcuno sta parlando sopra Jarvis. ⚡")
                        interrupted = True
                        break
                else:
                    loud_chunks = 0

            if interrupted:
                break

            data = wf.readframes(wav_chunk_size)

        out_stream.stop_stream()
        out_stream.close()
        return interrupted

def run_voice_assistant():
    global conversation_history

    print(f"⚙️ Inizializzazione Rhasspy Raven ('{WAKE_WORD}', DTW su MFCC, no ONNX/TFLite)...")
    raven = load_raven()

    pa = pyaudio.PyAudio()

    audio_stream = pa.open(
        rate=RATE, channels=CHANNELS, format=FORMAT,
        input=True, frames_per_buffer=CHUNK,
        stream_callback=mic_callback
    )
    audio_stream.start_stream()
    discard_mic_warmup()

    print(f"\n🤖 Jarvis è in STANDBY. Pronuncia \"Hey Jarvis\" per attivare la conversazione.")

    try:
        while True: #SENTINELLA
            pcm = coda_mic.get()

            if raven_detected(raven, pcm):
                print("\n✨ Parola d'ordine rilevata! Modalità conversazione ATTIVA. ✨")
                raven_reset(raven)

                # Pulizia rapida del buffer acustico
                time.sleep(0.1)
                while not coda_mic.empty():
                    coda_mic.get_nowait()

                in_active_conversation = True

                while in_active_conversation:

                    audio_path = record_dynamic_audio() # funzione BLOCCANTE che registra l'utente

                    if not audio_path: #se l'utente è stato zitto per 6 secondi torniamo in standby
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

                    hint = "parla sopra per interrompere" if ENABLE_BARGE_IN else "barge-in disattivato"
                    print(f"🔊 Riproduzione sulle casse ({hint})...")
                    interrupted = play_audio_with_barge_in(pa, RESPONSE_AUDIO_PATH)

                    if interrupted:
                        # Se interrompi con "Hey Jarvis" mentre le casse suonano, svuotiamo
                        # istantaneamente il buffer per eliminare l'eco della risposta di Jarvis!
                        time.sleep(0.1)
                        while not coda_mic.empty():
                            coda_mic.get_nowait()
                        print("\n👂 Prontissimo! Dimmi pure il nuovo comando...")
                        continue
                    else:
                        time.sleep(0.2)
                        print("\n👂 In attesa del prossimo turno (o pronuncia 'Hey Jarvis' per uscire)...")

                raven_reset(raven)
                print("\n🤖 Torno in STANDBY. In attesa di 'Hey Jarvis'...")

    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
    finally:
        audio_stream.stop_stream()
        audio_stream.close()
        pa.terminate()

if __name__ == "__main__":
    if "--record" in sys.argv:
        record_templates()
    else:
        run_voice_assistant()
