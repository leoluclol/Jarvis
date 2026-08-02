import os
import sys
import pyaudio
import numpy as np
import onnxruntime as ort
import math

SILERO_MODEL_PATH = "silero_vad.onnx"
RATE = 16000
CHUNK = 512

# ==========================================
# 🎚️ SOFTWARE VOLUME BOOST
# Se il volume è troppo basso, aumenta questo numero (es. 2.0, 5.0, 10.0)
# ==========================================
SOFTWARE_GAIN = 5.0  

def run_debug():
    ort.set_default_logger_severity(3)
    vad_session = ort.InferenceSession(SILERO_MODEL_PATH)
    pa = pyaudio.PyAudio()
    
    stream = pa.open(
        rate=RATE, channels=1, format=pyaudio.paInt16,
        input=True, frames_per_buffer=CHUNK
    )
    
    print("\n✅ Microphone LIVE. Speak loudly!")
    print(f"Software Gain applied: {SOFTWARE_GAIN}x\n")
    
    state = np.zeros((2, 1, 128), dtype=np.float32)
    
    try:
        while True:
            pcm = stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(pcm, dtype=np.int16)
            
            # Convert to float32
            audio_float32 = audio_np.astype(np.float32) / 32768.0
            
            # 1. Calculate actual volume (Root Mean Square) BEFORE gain
            rms_volume = math.sqrt(np.mean(audio_float32**2))
            
            # 2. Apply Software Gain and clip to prevent audio distortion
            audio_float32 = audio_float32 * SOFTWARE_GAIN
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)
            
            audio_tensor = np.expand_dims(audio_float32, axis=0)
            
            # Run ONNX inference
            inputs = {
                'input': audio_tensor,
                'sr': np.array(RATE, dtype=np.int64),
                'state': state
            }
            
            out, state = vad_session.run(None, inputs)
            prob = out[0][0]
            
            # Visual representation
            bar_length = 30
            filled_length = int(prob * bar_length)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # Print both the VAD probability AND the raw audio volume
            sys.stdout.write(f"\rVAD: [{bar}] {prob:.3f}  |  Vol: {rms_volume:.4f}")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping debug script.")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

if __name__ == "__main__":
    run_debug()