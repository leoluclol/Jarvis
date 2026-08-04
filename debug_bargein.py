#!/usr/bin/env python3
"""
Debug del BARGE-IN.

Suona del parlato dall'altoparlante e, CONTEMPORANEAMENTE, legge il microfono e
fa girare il detector della wakeword "Hey Jarvis", stampando la probabilita' in
tempo reale. Serve a capire se la wakeword viene rilevata MENTRE Jarvis parla.

Perche' esiste: in jarvis_v3_fixed.py il barge-in legge l'audio del mic dalla
coda `coda_mic`, ma NESSUNO riempie mai quella coda (non esiste alcun
`coda_mic.put`). Quindi durante la riproduzione il rilevatore legge una coda
sempre vuota e il barge-in non scatta mai. Qui invece leggiamo il mic
DIRETTAMENTE, cosi' vediamo se il detector funziona durante il playback.

Uso (sul Raspberry Pi, dove ci sono mic e casse):
    python3 debug_bargein.py
    python3 debug_bargein.py --wav frase.wav          # suona un WAV invece della TTS
    python3 debug_bargein.py --text "Ciao sono Jarvis" # testo TTS personalizzato
    python3 debug_bargein.py --threshold 0.6 --loops 5
    python3 debug_bargein.py --stop-on-detect          # ferma il playback al rilevamento
    python3 debug_bargein.py --no-baseline             # salta la fase di test in silenzio

Ctrl+C per uscire.
"""

import argparse
import os
import sys
import time
import wave
import io
import math
import struct
import threading

import pyaudio
from dotenv import load_dotenv
import requests
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model

# ------------------------------------------------------------------
# Parametri audio (identici a Jarvis per il ramo microfono/wakeword)
# ------------------------------------------------------------------
MIC_RATE = 16000            # la wakeword vuole 16 kHz mono
MIC_CHUNK = 160             # 10 ms per microWakeWord
DETECT_HEARTBEAT = 0.5      # ogni quanti secondi stampare il picco di probabilita'
WAKE_WORD_MODEL_NAME = "hey_jarvis"

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def init_wakeword():
    """Stesso init di Jarvis: azzera lo stato interno del modello."""
    try:
        m = MicroWakeWord.from_builtin(Model.HEY_JARVIS)
    except AttributeError:
        m = MicroWakeWord.from_builtin(WAKE_WORD_MODEL_NAME)
    return m, MicroWakeWordFeatures()


def get_tts_pcm(text):
    """Sintetizza `text` con la stessa TTS di Jarvis. Ritorna (pcm_bytes, rate)."""
    if not OPENAI_API_KEY:
        return None, None
    print(f"🔊 Genero la voce TTS per: “{text}” ...")
    try:
        r = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4o-mini-tts",
                "voice": "onyx",
                "input": text,
                "response_format": "pcm",   # PCM grezzo, 24 kHz mono int16
                "speed": 1.0,
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.content, 24000
    except Exception as e:
        print(f"⚠️  TTS fallita ({e}).")
        return None, None


def load_wav_pcm(path):
    """Carica un WAV mono 16-bit. Ritorna (pcm_bytes, rate)."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2 or wf.getnchannels() != 1:
            print("⚠️  Il WAV dovrebbe essere mono 16-bit; provo comunque.")
        return wf.readframes(wf.getnframes()), wf.getframerate()


def make_fallback_tone(seconds=3, rate=24000, freq=180):
    """Se non c'e' ne' TTS ne' WAV, genera un tono modulato (parlato finto)."""
    print("🔊 Uso un tono di fallback (niente TTS/WAV).")
    frames = bytearray()
    for t in range(int(seconds * rate)):
        # ampiezza modulata per simulare grossolanamente il parlato
        env = 0.5 + 0.5 * math.sin(2 * math.pi * 3 * t / rate)
        val = int(8000 * env * math.sin(2 * math.pi * freq * t / rate))
        frames += struct.pack("<h", val)
    return bytes(frames), rate


class Playback:
    """Riproduce audio in un thread separato, in loop, finche' non finisce o si ferma."""

    def __init__(self, pa, pcm, rate, loops):
        self.pa = pa
        self.pcm = pcm
        self.rate = rate
        self.loops = loops
        self.stop_flag = threading.Event()
        self.done = threading.Event()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        out = self.pa.open(
            format=pyaudio.paInt16, channels=1, rate=self.rate,
            output=True, frames_per_buffer=1024,
        )
        self.started.set()
        CH = 2048  # byte per write
        for _ in range(self.loops):
            if self.stop_flag.is_set():
                break
            for i in range(0, len(self.pcm), CH):
                if self.stop_flag.is_set():
                    break
                out.write(self.pcm[i:i + CH])
        out.stop_stream()
        out.close()
        self.done.set()

    def start(self):
        self.thread.start()
        self.started.wait(timeout=5)

    def stop(self):
        self.stop_flag.set()


def listen_loop(in_stream, mww, mww_features, threshold, until_event,
                stop_on_detect=False, on_detect=None, label=""):
    """
    Legge il mic e fa girare la wakeword finche' `until_event` non e' set.
    Stampa un battito con il picco di probabilita' e segnala i rilevamenti.
    Ritorna il numero di rilevamenti.
    """
    detections = 0
    window_peak = 0.0
    last_beat = time.time()
    start = time.time()

    while not until_event.is_set():
        try:
            pcm = in_stream.read(MIC_CHUNK, exception_on_overflow=False)
        except OSError:
            continue

        for features in mww_features.process_streaming(pcm):
            prob = mww.process_streaming_prob(features)
            window_peak = max(window_peak, prob)

            if prob >= threshold:
                detections += 1
                dt = time.time() - start
                print(f"   ⚡ [{label}] RILEVATA! prob={prob:.3f}  (t={dt:5.1f}s)")
                if on_detect:
                    on_detect()
                if stop_on_detect:
                    until_event.set()
                    break

        now = time.time()
        if now - last_beat >= DETECT_HEARTBEAT:
            bar = "█" * int(window_peak * 20)
            print(f"   [{label}] picco={window_peak:.3f} |{bar:<20}|", flush=True)
            window_peak = 0.0
            last_beat = now

    return detections


def main():
    ap = argparse.ArgumentParser(description="Debug del barge-in (wakeword durante il playback)")
    ap.add_argument("--wav", help="riproduci questo WAV invece della TTS")
    ap.add_argument("--text", default="Certo, posso aiutarti con questo. Ti spiego subito come funziona, "
                                       "cosi' puoi provare a interrompermi mentre parlo dicendo Hey Jarvis.",
                    help="testo da sintetizzare con la TTS")
    ap.add_argument("--threshold", type=float, default=0.75, help="soglia di rilevamento (default 0.75, come Jarvis)")
    ap.add_argument("--loops", type=int, default=3, help="quante volte ripetere l'audio (default 3)")
    ap.add_argument("--stop-on-detect", action="store_true", help="ferma il playback al primo rilevamento")
    ap.add_argument("--no-baseline", dest="baseline", action="store_false",
                    help="salta la fase di test in silenzio")
    ap.add_argument("--baseline-seconds", type=float, default=6.0, help="durata fase in silenzio")
    args = ap.parse_args()

    print("⚙️  Inizializzazione wakeword 'Hey Jarvis'...")
    mww, mww_features = init_wakeword()

    # Prepara l'audio da suonare
    if args.wav:
        pcm, rate = load_wav_pcm(args.wav)
    else:
        pcm, rate = get_tts_pcm(args.text)
        if pcm is None:
            pcm, rate = make_fallback_tone()
    dur = len(pcm) / 2 / rate
    print(f"🎧 Audio pronto: {dur:.1f}s @ {rate} Hz, ripetuto {args.loops}x "
          f"(~{dur * args.loops:.0f}s totali).")

    pa = pyaudio.PyAudio()
    in_stream = pa.open(
        rate=MIC_RATE, channels=1, format=pyaudio.paInt16,
        input=True, frames_per_buffer=MIC_CHUNK,
    )

    try:
        # ---- FASE 1: baseline in SILENZIO ----
        # Verifica che il detector funzioni SENZA playback. Se qui non rileva,
        # il problema e' il detector/mic, non il barge-in.
        if args.baseline:
            print("\n===== FASE 1: SILENZIO =====")
            print(f"👉 Di' \"Hey Jarvis\" ADESSO (niente audio in riproduzione). "
                  f"{args.baseline_seconds:.0f}s...")
            stop = threading.Event()
            t = threading.Timer(args.baseline_seconds, stop.set)
            t.start()
            n = listen_loop(in_stream, mww, mww_features, args.threshold, stop, label="SILENZIO")
            t.cancel()
            print(f"➡️  Rilevamenti in silenzio: {n}  "
                  f"({'OK, il detector funziona' if n else 'NESSUNO — controlla mic/soglia'})")
            # reset stato del modello prima della fase successiva
            mww, mww_features = init_wakeword()

        # ---- FASE 2: BARGE-IN durante il PLAYBACK ----
        print("\n===== FASE 2: BARGE-IN (audio in riproduzione) =====")
        print("👉 Mentre Jarvis parla, di' \"Hey Jarvis\" per interromperlo.")
        playback = Playback(pa, pcm, rate, args.loops)

        def on_detect():
            if args.stop_on_detect:
                print("   🛑 (stop-on-detect) fermo il playback.")
                playback.stop()

        playback.start()
        time.sleep(0.2)  # lascia partire il suono
        n2 = listen_loop(
            in_stream, mww, mww_features, args.threshold,
            until_event=playback.done, stop_on_detect=args.stop_on_detect,
            on_detect=on_detect, label="BARGE-IN",
        )
        playback.stop()
        playback.done.wait(timeout=2)

        print(f"\n➡️  Rilevamenti durante il playback: {n2}")
        if args.baseline:
            print("\n📊 DIAGNOSI:")
            if n and not n2:
                print("   Il detector funziona in silenzio ma NON durante il playback.")
                print("   => Probabile eco acustica (il mic sente l'altoparlante) o mic")
                print("      attenuato durante l'output. Vedi note sotto.")
            elif not n and not n2:
                print("   Non rileva mai: problema di mic/soglia/modello, non di barge-in.")
            elif n2:
                print("   Il barge-in FUNZIONA a livello di detector. In Jarvis il bug e'")
                print("   che 'coda_mic' non viene mai riempita durante play_stream_with_barge_in.")

    except KeyboardInterrupt:
        print("\nInterrotto.")
    finally:
        in_stream.stop_stream()
        in_stream.close()
        pa.terminate()


if __name__ == "__main__":
    main()
