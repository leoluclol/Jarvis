import requests
import io

class OpenAIClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def transcribe_audio(self, audio_io: io.BytesIO) -> str:
        files = {"file": ("command.wav", audio_io, "audio/wav")}
        data = {"model": "whisper-1", "language": "it", "temperature": "0.0"}
        
        response = self.session.post(
            "https://api.openai.com/v1/audio/transcriptions",
            files=files, data=data, timeout=20.0
        )
        response.raise_for_status()
        return response.json().get("text", "").strip()

    def generate_chat_response(self, conversation_history: list) -> str:
        response = self.session.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4o-mini", "messages": conversation_history},
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def stream_tts_audio(self, text: str):
        response = self.session.post(
            "https://api.openai.com/v1/audio/speech",
            json={
                "model": "gpt-4o-mini-tts",
                "voice": "onyx",
                "input": text,
                "response_format": "pcm",
                "speed": 1.3
            },
            stream=True, timeout=120
        )
        response.raise_for_status()
        return response