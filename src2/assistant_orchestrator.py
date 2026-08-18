from conversation_manager import ConversationManager
from openai_client import OpenAIClient
from main import Config
from wakeword_engine import WakeWordEngine
from vad import VoiceActivityDetector
from hardware_controller import HardwareController
from audio_controller import AudioController

import time

class AssistantOrchestrator:
    def __init__(self):
        self.config = Config()
        self.hw = HardwareController(self.config)
        self.audio = AudioController(self.config)
        self.wakeword = WakeWordEngine(self.config.WAKE_WORD_MODEL_NAME)
        self.vad = VoiceActivityDetector(self.config)
        self.llm = OpenAIClient(self.config.OPENAI_API_KEY)
        self.memory = ConversationManager(self.config.MAX_HISTORY)

        self.hw.setup()
        self.hw.turn_off_all()

    def _handle_barge_in(self, pcm_chunk: bytes) -> float:
        return self.wakeword.detect_barge_in(pcm_chunk)

    def run(self):
        print(f"⚙️ Inizializzazione microWakeWord ('{self.config.WAKE_WORD_MODEL_NAME}')...")
        print("🧠 Caricamento modello neurale Silero VAD (ONNX runtime)...")
        self.audio.start_mic()
        
        try:
            while True:
                self._standby_loop()
                self._active_conversation_loop()
        except KeyboardInterrupt:
            print("\nSpegnimento Jarvis...")
        finally:
            self._shutdown()

    def _standby_loop(self):
        print(f"\n🤖 Jarvis è in STANDBY. Pronuncia \"Hey Jarvis\" per attivare la conversazione.")
        self.wakeword.reset_state()
        
        while True:
            pcm = self.audio.read_mic_chunk()
            if not pcm:
                continue

            if self.wakeword.detect_in_stream(pcm, threshold=0.75):
                print("\n✨ Parola d'ordine rilevata! Modalità conversazione ATTIVA. ✨")
                self.wakeword.reset_state()
                self.audio.clear_mic_queue()
                return

    def _active_conversation_loop(self):
        in_active_conversation = True

        while in_active_conversation:
            self.hw.set_listen_led(True)
            audio_io = self.vad.record_until_silence(self.audio)
            self.hw.set_listen_led(False)

            if not audio_io:
                break

            self.audio.stop_mic()
            self.hw.set_think_led(True)
            
            print("🧠 Trascrizione in corso")
            t0 = time.time()
            try:
                user_text = self.llm.transcribe_audio(audio_io)
                print(f"⏱️ Trascrizione completata in {time.time() - t0:.2f} secondi.")
            except Exception as e:
                print(f"Errore di trascrizione: {e}")
                self.hw.set_think_led(False)
                self.audio.start_mic()
                continue
            finally:
                self.hw.set_think_led(False)

            print(f"👤 Tu: {user_text}")
            
            if self.memory.is_stop_command(user_text):
                print("🤖 Comando di spegnimento riconosciuto.")
                break 
                
            self.memory.add_user_message(user_text)
            
            print("🧠 Elaborazione risposta...")
            try:
                ai_text = self.llm.generate_chat_response(self.memory.get_history())
            except Exception as e:
                print(f"Errore LLM: {e}")
                self.audio.start_mic()
                continue

            if self.memory.is_hallucination(ai_text):
                self.memory.revert_last_user_message()
                continue

            print(f"🤖 Jarvis: {ai_text}")
            self.memory.add_assistant_message(ai_text)

            self.audio.start_mic()
            print("🗣️ Riproduzione in corso (pronuncia 'Hey Jarvis' per interrompere)...")
            
            try:
                tts_response = self.llm.stream_tts_audio(ai_text)
                interrupted = self.audio.play_tts_stream_with_barge_in(
                    tts_response.iter_content(chunk_size=4096),
                    self._handle_barge_in
                )
                tts_response.close()
            except Exception as e:
                print(f"Errore TTS: {e}")
                continue
            
            if interrupted:
                self.wakeword.reset_state()
                self.hw.set_listen_led(True)
                print("\n👂 Prontissimo! Dimmi pure il nuovo comando...")
                continue
            else:
                print("\n👂 In attesa del prossimo turno...")
        
        self.audio.start_mic()
        print("\n🤖 Torno in STANDBY. In attesa di 'Hey Jarvis'...")

    def _shutdown(self):
        self.audio.stop()
        self.hw.cleanup()