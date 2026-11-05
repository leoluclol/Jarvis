#pragma once

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace jarvis {

class TfLiteLibrary;  // dlopen'd TFLite C API (pimpl)

// Fixed-point TFLite Micro audio frontend wrapper. Produces 40 mel-filterbank
// features every 10 ms from 16 kHz mono int16 audio, using the exact same
// configuration and scaling as upstream pymicro_features.
class MicroFrontend {
public:
    MicroFrontend();
    ~MicroFrontend();

    // Appends raw int16 samples and extracts any complete feature frames.
    // Each frame is `feature_size()` floats. Returns the number of frames
    // produced (appended to `out`).
    std::size_t process_samples(
        const int16_t* samples,
        std::size_t count,
        std::vector<std::vector<float>>& out);

    void reset();
    std::size_t feature_size() const { return feature_size_; }
    std::size_t samples_per_chunk() const { return samples_per_chunk_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::size_t feature_size_ = 40;
    std::size_t samples_per_chunk_ = 160;
};

// Wake-word detection engine. Mirrors src2/wakeword_engine.py: the audio
// frontend feeds a sliding window of feature frames into a quantized TFLite
// model (hey_jarvis), and the reported probability is the mean over the
// model's sliding_window_size.
class WakeWordEngine {
public:
    // model_json: path to hey_jarvis.json (defines model + config).
    explicit WakeWordEngine(const std::string& model_json);
    ~WakeWordEngine();

    // Recreates the frontend + model state so RNN/CNN memory is cleared.
    void reset_state();

    // Feeds one mic chunk through the frontend; returns true as soon as a
    // per-chunk probability exceeds `threshold`.
    bool detect_in_stream(const std::vector<uint8_t>& audio_chunk,
                          float threshold = 0.75f);

    // Feeds one mic chunk and returns the highest per-chunk probability.
    float detect_barge_in(const std::vector<uint8_t>& audio_chunk);

private:
    friend class WakeWordTest;  // tests exercise process_streaming_prob()

    float process_streaming_prob(const std::vector<float>& features);

    std::string model_json_path_;
    std::string model_tflite_path_;
    std::string lib_path_;

    std::unique_ptr<TfLiteLibrary> tflite_;
    void* model_ = nullptr;
    void* interpreter_ = nullptr;
    void* input_tensor_ = nullptr;
    void* output_tensor_ = nullptr;

    int stride_ = 1;                 // feature frames per inference
    int sliding_window_size_ = 5;    // probabilities averaged
    std::size_t feature_size_ = 40;

    float input_scale_ = 1.0f;
    int input_zero_point_ = 0;
    int input_type_ = 0;
    std::size_t input_bytes_ = 0;

    float output_scale_ = 1.0f;
    int output_zero_point_ = 0;
    int output_type_ = 0;
    std::size_t output_bytes_ = 0;

    std::unique_ptr<MicroFrontend> frontend_;
    std::vector<std::vector<float>> pending_features_;
    std::deque<float> probabilities_;
};

}  // namespace jarvis
