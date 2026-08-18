from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model

class WakeWordEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.mww = None
        self.mww_features = None
        self.reset_state()

    def reset_state(self):
        try:
            self.mww = MicroWakeWord.from_builtin(Model.HEY_JARVIS)
        except AttributeError:
            self.mww = MicroWakeWord.from_builtin(self.model_name)
        self.mww_features = MicroWakeWordFeatures()

    def detect_in_stream(self, audio_chunk: bytes, threshold=0.75) -> bool:
        for features in self.mww_features.process_streaming(audio_chunk):
            if self.mww.process_streaming_prob(features) > threshold:
                return True
        return False

    def detect_barge_in(self, audio_chunk: bytes) -> float:
        max_prob = 0.0
        for features in self.mww_features.process_streaming(audio_chunk):
            prob = self.mww.process_streaming_prob(features)
            if prob > max_prob: max_prob = prob
        return max_prob