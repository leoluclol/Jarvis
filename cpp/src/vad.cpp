#include "jarvis/vad.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>

#include "jarvis/audio_controller.h"
#include "jarvis/logger.h"
#include "jarvis/wav.h"

namespace jarvis {

namespace {

constexpr size_t kContextSize = 64;    // last samples of the previous window
constexpr size_t kWindowSize = 512;    // samples per VAD inference
constexpr size_t kStateSize = 2 * 1 * 128;

std::vector<const char*> kInputNames = {"input", "sr", "state"};
std::vector<const char*> kOutputNames = {"output", "stateN"};

}  // namespace

struct VoiceActivityDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "jarvis-vad"};
    Ort::SessionOptions options;
    Ort::MemoryInfo memory{
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::Session session{nullptr};

    explicit Impl(const std::string& model_path) {
        options.SetIntraOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model_path.c_str(), options);
    }
};

VoiceActivityDetector::VoiceActivityDetector(const Config& config)
    : config_(config), impl_(new Impl(config.silero_model_path)) {
    reset_neural_state();
}

VoiceActivityDetector::~VoiceActivityDetector() = default;

void VoiceActivityDetector::reset_neural_state() {
    state_.assign(kStateSize, 0.0f);
    context_.assign(kContextSize, 0.0f);
}

bool VoiceActivityDetector::is_speech_detected(
    const std::vector<uint8_t>& chunk) {
    if (chunk.empty()) return false;

    const int16_t* samples =
        reinterpret_cast<const int16_t*>(chunk.data());

    // x = [context (64), window (512)] -> 1x576 tensor.
    std::vector<float> x(kContextSize + kWindowSize);
    for (size_t i = 0; i < kContextSize; ++i) x[i] = context_[i];
    for (size_t i = 0; i < kWindowSize; ++i) {
        x[kContextSize + i] = static_cast<float>(samples[i]) / 32768.0f;
    }

    const std::array<int64_t, 1> sr = {static_cast<int64_t>(config_.rate)};
    const std::array<int64_t, 2> input_shape = {
        static_cast<int64_t>(1),
        static_cast<int64_t>(x.size())};
    const std::array<int64_t, 0> sr_shape = {};
    const std::array<int64_t, 3> state_shape = {2, 1, 128};

    std::vector<Ort::Value> inputs;
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        impl_->memory, x.data(), x.size(), input_shape.data(),
        input_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
        impl_->memory, const_cast<int64_t*>(sr.data()), sr.size(),
        sr_shape.data(), sr_shape.size()));
    inputs.emplace_back(Ort::Value::CreateTensor<float>(
        impl_->memory, state_.data(), state_.size(), state_shape.data(),
        state_shape.size()));

    auto outputs = impl_->session.Run(
        Ort::RunOptions{}, kInputNames.data(), inputs.data(), inputs.size(),
        kOutputNames.data(), kOutputNames.size());

    const float prob = outputs[0].GetTensorData<float>()[0];

    const float* stateN = outputs[1].GetTensorData<float>();
    std::copy(stateN, stateN + state_.size(), state_.begin());

    // Context becomes the tail of the current window.
    std::copy(x.end() - kContextSize, x.end(), context_.begin());

    return prob >= static_cast<float>(config_.vad_threshold);
}

std::vector<uint8_t> VoiceActivityDetector::record_until_silence(
    AudioController& audio) {
    log_info("\n🎙️ Ascoltando... (stai zitto per annullare)");
    reset_neural_state();

    const auto start = std::chrono::steady_clock::now();
    const auto elapsed_seconds = [&start]() {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - start)
            .count();
    };

    std::vector<uint8_t> frames;
    bool has_spoken = false;

    std::vector<uint8_t> vad_buffer;
    int silent_windows = 0;
    const int max_silent_windows =
        static_cast<int>(config_.silence_timeout_vad / 0.032);

    // Short pre-roll so the first syllable is not clipped.
    std::deque<std::vector<uint8_t>> pre_speech;
    constexpr size_t kPreRollChunks = 32;

    while (true) {
        std::vector<uint8_t> pcm = audio.read_mic_chunk();
        if (pcm.empty()) continue;

        vad_buffer.insert(vad_buffer.end(), pcm.begin(), pcm.end());

        if (!has_spoken && elapsed_seconds() > config_.silence_timeout) {
            log_info("⏳ Timeout: Nessuna parola rilevata, torno in standby.");
            return {};
        }

        if (has_spoken) {
            frames.insert(frames.end(), pcm.begin(), pcm.end());
        } else {
            pre_speech.push_back(pcm);
            if (pre_speech.size() > kPreRollChunks) pre_speech.pop_front();
        }

        while (vad_buffer.size() >= 1024) {
            std::vector<uint8_t> vad_chunk(vad_buffer.begin(),
                                           vad_buffer.begin() + 1024);
            vad_buffer.erase(vad_buffer.begin(), vad_buffer.begin() + 1024);

            const bool speech = is_speech_detected(vad_chunk);
            if (speech) {
                if (!has_spoken) {
                    has_spoken = true;
                    for (auto& chunk : pre_speech) {
                        frames.insert(frames.end(), chunk.begin(), chunk.end());
                    }
                }
                silent_windows = 0;
            } else if (has_spoken) {
                ++silent_windows;
            }
        }

        if (has_spoken && silent_windows >= max_silent_windows) {
            log_info("🛑 Fine del discorso rilevata.");
            break;
        }
    }

    return build_wav(frames, config_.rate, config_.channels, 2);
}

}  // namespace jarvis
