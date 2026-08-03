import os
import asyncio
import websockets
import json
import base64
import pyaudio
import numpy as np
import time
from dotenv import load_dotenv
from openwakeword.model import Model

# ==========================================
# OTTIMIZZAZIONI CPU RASPBERRY PI 3
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ==========================================
# CONFIGURAZIONE
# ==========================================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WAKE_WORD = "hey_jarvis"
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1-mini"

# SOGLIA DEL RUMORE (Noise Gate Hardware)
# Taglia i rumori di fondo leggeri prima di inviarli a OpenAI.
# Alzalo a 1500 se hai molto rumore in stanza.
NOISE_THRESHOLD = 800 

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE_WAKEWORD = 16000
RATE_OPENAI = 24000
CHUNK = 1024

pa = pyaudio.PyAudio()

# ==========================================
# 1. RILEVAMENTO WAKE WORD
# ==========================================
def listen_for_wakeword():
    print(f"\n⚙️ Inizializzazione openWakeWord ('{WAKE_WORD}')...")
    oww_model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
    
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE_WAKEWORD, input=True, frames_per_buffer=CHUNK)
    print("\n🤖 Jarvis è in STANDBY. Pronuncia 'Hey Jarvis' per attivare...")
    
    try:
        while True:
            pcm = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(pcm, dtype=np.int16)
            prediction = oww_model.predict(audio_data)
            
            if prediction.get(WAKE_WORD, 0) > 0.5:
                print("\n✨ Parola d'ordine rilevata!")
                stream.stop_stream()
                stream.close()
                del oww_model
                return
    except KeyboardInterrupt:
        stream.close()
        exit(0)

# ==========================================
# 2. CONVERSAZIONE REALTIME VIA WEBSOCKET
# ==========================================
async def realtime_session():
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    mic_queue = asyncio.Queue()
    speaker_queue = asyncio.Queue()

    # Callback con Noise Gate hardware
    # Callback con Noise Gate hardware basato su Numpy
    def mic_callback(in_data, frame_count, time_info, status):
        # Convertiamo i byte in numeri per calcolare il volume
        audio_array = np.frombuffer(in_data, dtype=np.int16)
        
        # Calcolo matematico del volume RMS (Radice della Media dei Quadrati)
        # Usiamo float32 per evitare che i numeri si saturino durante il calcolo
        volume = np.sqrt(np.mean(np.square(audio_array.astype(np.float32))))
        
        if volume < NOISE_THRESHOLD:
            # Ammutolisce il fruscio di fondo rimpiazzandolo con byte vuoti
            in_data = b'\x00' * len(in_data)
            
        loop.call_soon_threadsafe(mic_queue.put_nowait, in_data)
        return (None, pyaudio.paContinue)

    in_stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE_OPENAI, input=True, frames_per_buffer=CHUNK, stream_callback=mic_callback)
    out_stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE_OPENAI, output=True, frames_per_buffer=2048)

    loop = asyncio.get_running_loop()
    print("🌐 Connessione a OpenAI Realtime API in corso...")

    try:
        async with websockets.connect(WS_URL, additional_headers=headers) as ws:
            print("✅ Connesso! Parla pure con Jarvis.")

            # IL NUOVO PAYLOAD UFFICIALE (Adattato per PyAudio)
            session_update = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime-2.1",
                    "output_modalities": ["audio", "text"], # Vogliamo anche il testo per vederlo a schermo
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": 24000,
                            },
                            "turn_detection": {"type": "semantic_vad"}, # Il nuovo nome del VAD!
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm", # Trasformato in PCM standard per le tue casse
                                "rate": 24000,
                            },
                            "voice": "marin",
                        },
                    },
                    "instructions": "Sei Jarvis, un assistente vocale domestico. Rispondi in italiano in modo conciso e diretto.",
                },
            }
            await ws.send(json.dumps(session_update))

            async def send_audio():
                while True:
                    data = await mic_queue.get()
                    # Inviamo il silenzio solo se necessario, OpenAI lo gestirà col semantic_vad
                    base64_audio = base64.b64encode(data).decode('utf-8')
                    msg = {"type": "input_audio_buffer.append", "audio": base64_audio}
                    await ws.send(json.dumps(msg))

            async def play_audio():
                while True:
                    data = await speaker_queue.get()
                    await asyncio.to_thread(out_stream.write, data)

            async def receive_events():
                async for message in ws:
                    event = json.loads(message)
                    event_type = event.get("type")

                    # I nomi degli eventi scoperti con il radar
                    if event_type == "response.output_audio.delta":
                        speaker_queue.put_nowait(base64.b64decode(event["delta"]))
                        
                    elif event_type in ["response.output_audio_transcript.delta", "response.text.delta"]:
                        print(event.get("delta", ""), end="", flush=True)

                    elif event_type == "input_audio_buffer.speech_started":
                        print("\n🎙️ [Server] Hai iniziato a parlare...")
                        if not speaker_queue.empty():
                            print("\n⚡ Interruzione IA in corso! Svuoto audio...")
                            while not speaker_queue.empty():
                                speaker_queue.get_nowait()
                                
                    elif event_type == "input_audio_buffer.speech_stopped":
                        print("\n⏳ [Server] Hai finito di parlare. Attendo risposta...")

                    elif event_type == "response.done":
                        print("\n\n🤖 [Server] Risposta completata.")
                        
                    elif event_type == "error":
                        print(f"\n❌ ERRORE DA OPENAI: {event.get('error', {}).get('message', event)}")

            tasks = [
                asyncio.create_task(send_audio()),
                asyncio.create_task(play_audio()),
                asyncio.create_task(receive_events())
            ]

            await asyncio.sleep(25)
            print("\n⏳ Fine sessione conversazione. Torno in standby...")

            for t in tasks:
                t.cancel()

    except Exception as e:
        print(f"❌ Errore durante la sessione WebSocket: {e}")
    finally:
        in_stream.stop_stream()
        in_stream.close()
        out_stream.stop_stream()
        out_stream.close()

# ==========================================
# MAIN LOOP
# ==========================================
def main():
    while True:
        listen_for_wakeword()
        asyncio.run(realtime_session())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSpegnimento Jarvis...")
        pa.terminate()