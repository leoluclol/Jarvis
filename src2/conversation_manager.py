import re

class ConversationManager:
    def __init__(self, max_history: int):
        self.max_history = max_history
        self.system_prompt = {
            "role": "system", 
            "content": "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. Rispondi in italiano."
        }
        self.clear_history()

    def add_user_message(self, text: str):
        self.history.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str):
        self.history.append({"role": "assistant", "content": text})
        self._trim_history()

    def get_history(self) -> list:
        return self.history

    def clear_history(self):
        self.history = [self.system_prompt]

    def revert_last_user_message(self):
        self.history.pop()

    def is_stop_command(self, text: str) -> bool:
        text_clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
        stop_commands = ["stop", "basta", "stai zitto", "zitto", "fermati", "spegniti"]
        return any(cmd in text_clean for cmd in stop_commands) and len(text_clean.split()) <= 4

    def is_hallucination(self, text: str) -> bool:
        return "amara.org" in text.lower() or "qtss" in text.lower()

    def _trim_history(self):
        if len(self.history) > self.max_history + 1:
            self.history = [self.history[0]] + self.history[-self.max_history:]