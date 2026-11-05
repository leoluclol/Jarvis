#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "jarvis/config.h"

namespace jarvis {

class AudioController;

// Silero VAD running through ONNX Runtime. Mirrors src2/vad.py, including the
// neural state (state + context) that must be threaded through inference.
class VoiceActivityDetector {
public:
    explicit VoiceActivityDetector(const Config& config);
    ~VoiceActivityDetector();

    // Records from `audio` until end-of-speech is detected. Returns the
    // recorded audio as WAV bytes, or an empty vector on timeout/cancel.
    std::vector<uint8_t> record_until_silence(AudioController& audio);

private:
    friend class VadTest;  // tests drive is_speech_detected() directly

    bool is_speech_detected(const std::vector<uint8_t>& chunk);
    void reset_neural_state();

    Config config_;

    struct Impl;                 // ONNX Runtime session/values
    std::unique_ptr<Impl> impl_;

    // Neural state: (2,1,128) floats and (1,64) context window.
    std::vector<float> state_;
    std::vector<float> context_;
};

}  // namespace jarvis
