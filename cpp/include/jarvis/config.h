#pragma once

#include <string>

namespace jarvis {

// Mirrors the Python `Config` class from src2/main.py.
struct Config {
    // Paths (resolved at load time; the compile-time JARVIS_* macros give the
    // repo layout regardless of the process CWD).
    std::string root_dir;
    std::string model_dir;   // <root>/models  (silero_vad.onnx)
    std::string audio_dir;   // <root>/audio
    std::string cpp_dir;     // <root>/cpp     (wake-word model lives here)

    std::string api_key;

    std::string wake_word_model_name = "hey_jarvis";
    std::string wake_word_model_json;   // cpp/models/hey_jarvis.json
    std::string wake_word_model_tflite; // cpp/models/hey_jarvis.tflite
    std::string silero_model_path;      // <root>/models/silero_vad.onnx

    // Audio settings
    int channels = 1;
    int rate = 16000;              // mic sample rate
    int chunk = 160;               // mic frames per read (10 ms)
    int openai_tts_rate = 24000;   // TTS output sample rate

    // VAD & conversation logic
    double silence_timeout_vad = 1.5;  // s of silence to end a turn
    double silence_timeout = 5.0;      // s before a recording is cancelled
    double vad_threshold = 0.5;        // Silero speech probability threshold
    int max_history = 6;               // rolling conversation turns

    // Hardware (Raspberry Pi GPIO, BCM numbering)
    int led_think = 5;
    int led_listen = 13;

    // Throws std::runtime_error if OPENAI_API_KEY is missing.
    static Config load();
};

}  // namespace jarvis
