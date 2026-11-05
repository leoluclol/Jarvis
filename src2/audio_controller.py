from config import Config
import queue
import pyaudio
import re
import select
import subprocess
import threading
import time

INPUT_PATTERNS = ("camera", "usb", "mic")
OUTPUT_PATTERNS = ("bcm2835", "headphone", "hdmi")


class AudioController:
    def __init__(self, config: Config):
        self.config = config
        self.pa = pyaudio.PyAudio()
        self.coda_mic = queue.Queue()
        self._mic = None          # pyaudio input stream (when PortAudio works)
        self._mic_proc = None     # arecord fallback (when PortAudio sees no input)
        self._arecord_warned = False

        in_idx = self._pick_input_device()
        if in_idx is None:
            print("🎙️ Nessun ingresso via PortAudio: cattura con ALSA direttamente (arecord).")
            self._start_arecord()
        else:
            self._mic = self.pa.open(
                rate=self.config.RATE,
                channels=self.config.CHANNELS,
                format=self.config.FORMAT,
                input=True,
                input_device_index=in_idx,
                frames_per_buffer=self.config.CHUNK,
            )

    # ---------- device selection ----------

    def _scan_devices(self, for_input: bool, patterns=()):
        for i in range(self.pa.get_device_count()):
            try:
                info = self.pa.get_device_info_by_index(i)
            except OSError:
                continue
            ch = info["maxInputChannels"] if for_input else info["maxOutputChannels"]
            if ch > 0 and (not patterns or any(p in info.get("name", "").lower() for p in patterns)):
                print(f"🔌 Dispositivo: {info['name']}")
                return i
        return None

    def _pick_input_device(self):
        try:
            return self.pa.get_default_input_device_info()["index"]
        except OSError:
            pass
        idx = self._scan_devices(True, INPUT_PATTERNS)
        if idx is not None:
            return idx
        return self._scan_devices(True)

    def _pick_output_device(self):
        override = self.config.AUDIO_OUTPUT_INDEX
        if override not in (None, ""):
            print(f"🔌 Uscita forzata: index {override}")
            return int(override)
        try:
            return self.pa.get_default_output_device_info()["index"]
        except OSError:
            pass
        idx = self._scan_devices(False, OUTPUT_PATTERNS)
        if idx is not None:
            return idx
        idx = self._scan_devices(False)
        if idx is not None:
            return idx
        raise RuntimeError(
            "Nessun dispositivo di uscita audio trovato (serve il jack bcm2835)."
        )

    # ---------- ALSA (arecord) fallback for the microphone ----------

    def _detect_alsa_device(self):
        dev = self.config.ALSA_INPUT_DEVICE
        if dev:
            return dev
        try:
            out = subprocess.run(["arecord", "-l"], capture_output=True,
                                 text=True, timeout=5).stdout
            for line in out.splitlines():
                m = re.match(r"card (\d+):.*?device (\d+):", line)
                if m:
                    return f"plughw:{m.group(1)},{m.group(2)}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "default"

    def _start_arecord(self):
        dev = self._detect_alsa_device()
        print(f"🎙️ ALSA capture su: {dev}")
        self._mic_proc = subprocess.Popen(
            ["arecord", "-D", dev, "-r", str(self.config.RATE),
             "-c", str(self.config.CHANNELS), "-f", "S16_LE", "-q", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def _stop_arecord(self):
        if self._mic_proc is not None:
            try:
                self._mic_proc.terminate()
                self._mic_proc.wait(timeout=2)
            except Exception:
                self._mic_proc.kill()
            self._mic_proc = None

    def _drain_mic(self):
        """Scarta l'audio vecchio accumulato nella pipe mentre il mic era fermo."""
        while True:
            r, _, _ = select.select([self._mic_proc.stdout], [], [], 0)
            if not r:
                break
            if not self._mic_proc.stdout.read(4096):
                break

    # ---------- mic interface ----------

    def read_mic_chunk(self):
        if self._mic is not None:
            try:
                return self._mic.read(self.config.CHUNK, exception_on_overflow=False)
            except OSError:
                return None
        if self._mic_proc is None or self._mic_proc.poll() is not None:
            if not self._arecord_warned:
                print("⚠️  arecord non sta più catturando: controlla il microfono USB.")
                self._arecord_warned = True
            return None
        try:
            data = self._mic_proc.stdout.read(self.config.CHUNK)
        except (OSError, ValueError):
            return None
        return data or None

    def start_mic(self):
        if self._mic is not None:
            if self._mic.is_stopped():
                self._mic.start_stream()
        elif self._mic_proc is not None:
            self._drain_mic()

    def stop_mic(self):
        if self._mic is not None and not self._mic.is_stopped():
            self._mic.stop_stream()

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
            output_device_index=self._pick_output_device(),
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
        if self._mic is not None:
            try:
                self._mic.stop_stream()
                self._mic.close()
            except OSError:
                pass
        self._stop_arecord()
        self.pa.terminate()
