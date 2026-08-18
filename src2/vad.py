from main import Config
import numpy as np
import onnxruntime as ort
from audio_controller import AudioController
import time
import collections
import io
import wave
from typing import Optional


# ==========================================
# 5. VOICE ACTIVITY DETECTOR (VAD)
# ==========================================
class VoiceActivityDetector:
    def __init__(self, config: Config):
        self.config = config
        self.session = ort.InferenceSession(config.SILERO_MODEL_PATH)
        
        # Dichiariamo esplicitamente il tipo per evitare l'errore del linter su np.concatenate
        self.state: np.ndarray
        self.context: np.ndarray
        self._reset_neural_state()

    def _reset_neural_state(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, 64), dtype=np.float32)

    def _is_speech_detected(self, audio_chunk: bytes) -> bool:
        if not audio_chunk:
            return False
        
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        window_size = 512
        is_speech = False
        
        for i in range(0, len(audio_float32) - window_size + 1, window_size):
            window_numpy = audio_float32[i : i + window_size]
            window_tensor = np.expand_dims(window_numpy, axis=0)
            
            # Ora il linter sa per certo che self.context è un np.ndarray
            x = np.concatenate([self.context, window_tensor], axis=1)
            inputs = {'input': x, 'sr': np.array(self.config.RATE, dtype=np.int64), 'state': self.state}
            out, self.state = self.session.run(None, inputs)
            prob = out[0][0]
            
            self.context = window_tensor[:, -64:]
            if prob >= self.config.VAD_THRESHOLD:
                is_speech = True
                
        return is_speech

    def record_until_silence(self, audio_controller: AudioController) -> Optional[io.BytesIO]:
        print("\n🎙️ Ascoltando... (stai zitto per annullare)")
        self._reset_neural_state()
        
        start_time = time.time()
        frames = []
        has_spoken = False
        
        vad_buffer = bytearray()
        silent_windows = 0
        max_silent_windows = int(self.config.SILENCE_TIMEOUT_VAD / 0.032)
        pre_speech_buffer = collections.deque(maxlen=32) 
        
        while True:
            pcm = audio_controller.read_mic_chunk()
            if not pcm:
                continue
                
            vad_buffer.extend(pcm)
            
            if not has_spoken and time.time() - start_time > self.config.SILENCE_TIMEOUT:
                print("⏳ Timeout: Nessuna parola rilevata, torno in standby.")
                return None
                    
            if has_spoken:
                frames.append(pcm)
            else:
                pre_speech_buffer.append(pcm)
            
            while len(vad_buffer) >= 1024:
                vad_chunk = bytes(vad_buffer[:1024])
                del vad_buffer[:1024]
                
                is_speech = self._is_speech_detected(vad_chunk)
                
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
            wf.setnchannels(self.config.CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(self.config.RATE)
            wf.writeframes(b"".join(frames))
            
        audio_buffer_io.seek(0)
        return audio_buffer_io