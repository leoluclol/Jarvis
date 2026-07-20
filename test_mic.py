import time
import numpy as np
import openwakeword
from openwakeword.model import Model
import pyaudio

# 1. PARAMETRI TASSATIVI PER OPENWAKEWORD
RATE = 16000  # Deve essere esattamente 16000 Hz
CHUNK = 1280  # 1280 campioni = 80ms di audio (ottimale per OWW)
CHANNELS = 1
FORMAT = pyaudio.paInt16

print("⚙️ Caricamento modello per il test...")
# Trova il percorso del modello come abbiamo fatto per la v0.5.0
all_onnx_models = openwakeword.get_pretrained_model_paths()
jarvis_model_path = [p for p in all_onnx_models if "hey_jarvis" in p][0]

oww_model = Model(wakeword_model_paths=[jarvis_model_path])

# Stampa le chiavi esatte del modello (FONDAMENTALE PER IL DEBUG!)
print(
    f"🔑 CHIAVI ATTIVE NEL MODELLO: {list(oww_model.models.keys())}\n"
)

pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=RATE,
    channels=CHANNELS,
    format=FORMAT,
    input=True,
    frames_per_buffer=CHUNK,
)

print(
    "🎙️ Microfono aperto! Parla per vedere il livello del volume e le predizioni."
)
print("Premi Ctrl+C per uscire.\n")
print("VOLUME AUDIO          | PREDIZIONE DEL MODELLO")
print("-" * 55)

try:
    while True:
        # Leggi i dati dal microfono
        pcm = audio_stream.read(CHUNK, exception_on_overflow=False)
        audio_data = np.frombuffer(pcm, dtype=np.int16)

        # 1. Calcola il volume (Root Mean Square) per la barra visiva
        volume = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        bar_length = int(
            min(volume / 50, 20)
        )  # Adatta la barra alla larghezza
        volume_bar = "█" * bar_length + "-" * (20 - bar_length)

        # 2. Fai la predizione con OpenWakeWord
        prediction = oww_model.predict(audio_data)

        # Trova il punteggio più alto nel dizionario
        max_score = max(prediction.values()) if prediction else 0.0

        # Stampa a schermo: se il punteggio supera 0.05, evidenzialo!
        if max_score > 0.05:
            # Pulisce la riga e stampa in tempo reale
            print(f"\r[{volume_bar}] | ⚠️ RILEVAMENTO: {prediction}", end="")
            if max_score > 0.5:
                print("\n🚀 PAROLA D'ORDINE SUPERATA (>0.5)!")
                oww_model.reset()  # Resetta il buffer dopo un successo
        else:
            print(f"\r[{volume_bar}] | Silenzio... (max: {max_score:.3f})", end="")

        time.sleep(0.01)

except KeyboardInterrupt:
    print("\n\n🛑 Test terminato.")
finally:
    audio_stream.close()
    pa.terminate()