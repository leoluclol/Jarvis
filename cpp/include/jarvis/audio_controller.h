#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "jarvis/config.h"
#include "jarvis/thread_queue.h"
#include "jarvis/types.h"

namespace jarvis {

// PortAudio-backed microphone capture and TTS playback with barge-in.
// Mirrors src2/audio_controller.py.
class AudioController {
public:
    explicit AudioController(const Config& config);
    ~AudioController();

    // Blocks until a mic chunk (chunk * channels * 2 bytes) is available.
    // Returns an empty vector on a transient hardware error.
    std::vector<uint8_t> read_mic_chunk();

    void start_mic();
    void stop_mic();
    void clear_mic_queue();

    // Streams TTS PCM from `tts` to the speakers while listening for a
    // barge-in. `barge_in_prob` computes a wake-word probability from one raw
    // mic chunk; returns true if playback was interrupted.
    bool play_tts_stream_with_barge_in(
        TtsChunkSource& tts,
        const std::function<float(const std::vector<uint8_t>&)>& barge_in_prob);

    // Closes streams and terminates PortAudio.
    void stop();

private:
    Config config_;
    bool mic_running_ = false;
    void* mic_stream_ = nullptr;  // PaStream*

    ThreadQueue<std::vector<uint8_t>> mic_queue_;
};

}  // namespace jarvis
