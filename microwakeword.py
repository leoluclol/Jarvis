import time
import os
import psutil
import pyaudio
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model

def main():
    # Audio capture settings required by microWakeWord
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    # 10ms of audio @ 16kHz = 160 samples (320 bytes)
    CHUNK_SAMPLES = 160

    print("Initializing microWakeWord ('Hey Jarvis')...")
    
    # Load 'HEY_JARVIS' model (downloads automatically if not cached)
    try:
        mww = MicroWakeWord.from_builtin(Model.HEY_JARVIS)
    except AttributeError:
        # Fallback to model string/path if enum naming differs in version
        mww = MicroWakeWord.from_builtin("hey_jarvis")
        
    mww_features = MicroWakeWordFeatures()

    # PyAudio setup
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK_SAMPLES
    )

    process = psutil.Process(os.getpid())
    print("\n[READY] Listening for 'Hey Jarvis'... (Press Ctrl+C to stop)")
    print("-" * 55)

    detection_count = 0
    inference_times = []

    try:
        while True:
            # Read 10ms of raw PCM audio (320 bytes)
            audio_data = stream.read(CHUNK_SAMPLES, exception_on_overflow=False)

            # Process 10ms frame into features and pass to streaming inference
            for features in mww_features.process_streaming(audio_data):
                t0 = time.perf_counter()
                
                # Returns float probability (0.0 - 1.0)
                prob = mww.process_streaming_prob(features)
                
                t_elapsed_ms = (time.perf_counter() - t0) * 1000
                inference_times.append(t_elapsed_ms)

                # Keep a rolling window of stats
                if len(inference_times) > 100:
                    inference_times.pop(0)

                # Check for detection (cutoff usually 0.5 - 0.7)
                if prob > 0.5:
                    detection_count += 1
                    cpu_usage = process.cpu_percent()
                    avg_latency = sum(inference_times) / len(inference_times)
                    
                    print(
                        f"🔥 WAKE WORD DETECTED! #{detection_count} "
                        f"| Prob: {prob:.2f} "
                        f"| Inference Latency: {avg_latency:.2f} ms "
                        f"| Process CPU: {cpu_usage:.1f}%"
                    )

    except KeyboardInterrupt:
        print("\nStopping stream...")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        if inference_times:
            avg_lat = sum(inference_times) / len(inference_times)
            print("-" * 55)
            print(f"Average Inference Latency per 30ms window: {avg_lat:.2f} ms")
            print(f"Total Detections: {detection_count}")

if __name__ == "__main__":
    main()