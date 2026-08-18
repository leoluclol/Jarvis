import os
import pyaudio
from dotenv import load_dotenv
from assistant_orchestrator import AssistantOrchestrator


# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    def __init__(self):
        load_dotenv()
        self.ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "models")
        self.AUDIO_DIR = os.path.join(self.ROOT_DIR, "audio")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY non trovata")
        self.OPENAI_API_KEY: str = api_key

        self.WAKE_WORD_MODEL_NAME = "hey_jarvis"
        self.SILERO_MODEL_PATH = os.path.join(self.MODEL_DIR, "silero_vad.onnx")

        # Audio Settings
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 160
        self.OPENAI_TTS_RATE = 24000

        # VAD & Logic Settings
        self.SILENCE_TIMEOUT_VAD = 1.5
        self.SILENCE_TIMEOUT = 5.0
        self.VAD_THRESHOLD = 0.5
        self.MAX_HISTORY = 6
        
        # Hardware Settings
        self.LED_THINK = 5
        self.LED_LISTEN = 13


if __name__ == "__main__":
    app = AssistantOrchestrator()
    app.run()