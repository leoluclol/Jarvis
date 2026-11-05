#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "jarvis/vad.h"

#ifndef JARVIS_REPO_ROOT
#define JARVIS_REPO_ROOT "."
#endif

namespace jarvis {

// Friend of VoiceActivityDetector: drives the ONNX model directly, without
// needing a microphone. Validates the model loads and that our input tensor
// shapes (input [1,576], state [2,1,128], scalar sr) are correct.
class VadTest {
public:
    static int run() {
        Config cfg;
        cfg.rate = 16000;
        cfg.silero_model_path =
            std::string(JARVIS_REPO_ROOT) + "/models/silero_vad.onnx";
        cfg.vad_threshold = 0.5;

        std::cout << "loading Silero VAD from " << cfg.silero_model_path
                  << std::endl;
        VoiceActivityDetector vad(cfg);

        // Silence: 1024 bytes of zeros -> should be < threshold.
        std::vector<uint8_t> silence(1024, 0);
        bool s1 = vad.is_speech_detected(silence);
        std::cout << "silence -> speech=" << s1 << std::endl;

        // Speech-like: strong low-frequency tone -> should be > threshold.
        std::vector<int16_t> tone(512);
        for (size_t i = 0; i < tone.size(); ++i) {
            tone[i] = static_cast<int16_t>(
                30000.0 * std::sin(2.0 * M_PI * 200.0 * i / 16000.0));
        }
        std::vector<uint8_t> tone_bytes;
        tone_bytes.reserve(tone.size() * 2);
        for (int16_t s : tone) {
            tone_bytes.push_back(static_cast<uint8_t>(s & 0xFF));
            tone_bytes.push_back(static_cast<uint8_t>((s >> 8) & 0xFF));
        }
        bool s2 = vad.is_speech_detected(tone_bytes);
        std::cout << "tone    -> speech=" << s2 << std::endl;

        // A second call after the first (state continuity) must not crash.
        bool s3 = vad.is_speech_detected(silence);
        std::cout << "silence2-> speech=" << s3 << std::endl;

        if (s1 || !s2) {
            std::cerr << "VAD behavior unexpected (silence=" << s1
                      << ", tone=" << s2 << ")" << std::endl;
            return 1;
        }
        std::cout << "test_vad: OK" << std::endl;
        return 0;
    }
};

}  // namespace jarvis

int main() { return jarvis::VadTest::run(); }
