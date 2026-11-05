from config import Config
import queue
import pyaudio
import threading
import time

class AudioController:
    def __init__(self, config: Config):
        self.config = config
        self.pa = pyaudio.PyAudio()
        self.mic_stream = self.pa.open(
            rate=self.config.RATE,
            channels=self.config.CHANNELS,
            format=self.config.FORMAT,
            input=True,
            input_device_index=self._pick_device(True),
            frames_per_buffer=self.config.CHUNK
        )
        self.coda_mic = queue.Queue()

    def _pick_device(self, for_input: bool) -> int:
        """Pick a device: explicit override > system default > name match
        (USB mic in, bcm2835 jack out) > any usable device."""
        override = (self.config.AUDIO_INPUT_INDEX if for_input
                    else self.config.AUDIO_OUTPUT_INDEX)
        if override not in (None, ""):
            print(f"🔌 Dispositivo forzato: index {override}")
            return int(override)

        def usable(info):
            ch = info["maxInputChannels"] if for_input else info["maxOutputChannels"]
            return ch > 0, ch

        # 1) system default
        try:
            info = (self.pa.get_default_input_device_info()
                    if for_input else self.pa.get_default_output_device_info())
            print(f"🔌 Dispositivo default: {info['name']}")
            return info["index"]
        except OSError:
            pass

        patterns = ("camera", "usb", "mic") if for_input else ("bcm2835", "headphone", "hdmi")

        # 2) match by name
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
            except OSError:
                continue
            ok, _ = usable(info)
            if ok and any(p in info.get("name", "").lower() for p in patterns):
                print(f"🔌 Dispositivo selezionato: {info['name']}")
                return i

        # 3) any usable device
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
            except OSError:
                continue
            ok, _ = usable(info)
            if ok:
                print(f"🔌 Dispositivo selezionato: {info['name']}")
                return i

        # 4) diagnostics: show exactly what PortAudio sees
        print(f"⚠️  PortAudio vede {self.pa.get_device_count()} dispositivi:")
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
                print(f"    [{i}] {info['name']} "
                      f"(in={info['maxInputChannels']}, out={info['maxOutputChannels']})")
            except OSError as e:
                print(f"    [{i}] errore: {e}")
        raise RuntimeError(
            "Nessun dispositivo audio trovato. Vedi la lista qui sopra: se è vuota, "
            "PortAudio non vede la scheda audio (configura /etc/asound.conf da "
            "asound.conf.example). Se invece il dispositivo c'è, forzalo con "
            "JARVIS_AUDIO_INPUT_INDEX / JARVIS_AUDIO_OUTPUT_INDEX."
        )

    def read_mic_chunk(self):
        try:
            return self.mic_stream.read(self.config.CHUNK, exception_on_overflow=False)
        except OSError:
            return None

    def start_mic(self):
        if self.mic_stream.is_stopped():
            self.mic_stream.start_stream()

    def stop_mic(self):
        if not self.mic_stream.is_stopped():
            self.mic_stream.stop_stream()

    def clear_mic_queue(self):
        while not self.coda_mic.empty():
            try:
                self.coda_mic.get_nowait()
            except queue.Empty:
                break

    def play_tts_stream_with_barge_in(self, tts_iterator, barge_in_callback) -> bool:
        self.clear_mic_queue()
        
        out_stream = self.pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=self.config.OPENAI_TTS_RATE, output=True,
            output_device_index=self._pick_device(False),
            frames_per_buffer=2048
        )

        stop_reader = threading.Event()

        def mic_reader():
            while not stop_reader.is_set():
                pcm = self.read_mic_chunk()
                if pcm:
                    self.coda_mic.put(pcm)

        reader_thread = threading.Thread(target=mic_reader, daemon=True)
        reader_thread.start()

        interrupted = False
        audio_buffer = bytearray()

        SAFE_CHUNK_SIZE = 4096
        PREBUFFER_SIZE = 16384
        is_playing = False
        playback_start_time = 0
        GRACE_PERIOD = 1.0

        def detect_bargein():
            while not self.coda_mic.empty():
                try:
                    pcm = self.coda_mic.get_nowait()
                except queue.Empty:
                    break
                
                prob = barge_in_callback(pcm)
                if is_playing and (time.time() - playback_start_time > GRACE_PERIOD) and prob > 0.6:
                    print(f"\n⚡ BARGE-IN RILEVATO! (Prob: {prob:.2f})... ⚡")
                    return True
            return False

        try:
            # Phase 1: Streaming
            for chunk in tts_iterator:
                if chunk:
                    audio_buffer.extend(chunk)

                if not is_playing and len(audio_buffer) >= PREBUFFER_SIZE:
                    is_playing = True
                    playback_start_time = time.time()

                if is_playing:
                    while len(audio_buffer) >= SAFE_CHUNK_SIZE:
                        out_stream.write(bytes(audio_buffer[:SAFE_CHUNK_SIZE]))
                        del audio_buffer[:SAFE_CHUNK_SIZE]
                        if detect_bargein():
                            interrupted = True
                            break

                if not interrupted and detect_bargein():
                    interrupted = True

                if interrupted:
                    break

            # Phase 2: Tail processing
            if not interrupted:
                if not is_playing:
                    is_playing = True
                    playback_start_time = time.time()
                while len(audio_buffer) >= 2 and not interrupted:
                    n = min(SAFE_CHUNK_SIZE, len(audio_buffer))
                    if n % 2 != 0: n -= 1
                    if n <= 0: break
                    out_stream.write(bytes(audio_buffer[:n]))
                    del audio_buffer[:n]
                    if detect_bargein():
                        interrupted = True

        finally:
            stop_reader.set()
            reader_thread.join(timeout=1.0)
            out_stream.stop_stream()
            out_stream.close()
            self.clear_mic_queue()

        return interrupted

    def stop(self):
        self.stop_mic()
        self.mic_stream.close()
        self.pa.terminate()