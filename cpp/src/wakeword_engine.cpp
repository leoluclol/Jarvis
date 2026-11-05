#include "jarvis/wakeword_engine.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "jarvis/logger.h"

// The vendored TFLite Micro audio frontend (C API, extern "C").
extern "C" {
#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
}

namespace jarvis {

#ifndef JARVIS_CPP_DIR
#define JARVIS_CPP_DIR "."
#endif

namespace {

constexpr float kFloat32Scale = 0.0390625f;  // 1/256, same as pymicro_features

bool file_exists(const std::string& path) {
    std::ifstream in(path);
    return in.good();
}

// numpy's np.round: round-half-to-even.
int32_t round_half_even(float x) {
    float floored = std::floor(x);
    float diff = x - floored;
    if (diff > 0.5f) return static_cast<int32_t>(floored) + 1;
    if (diff < 0.5f) return static_cast<int32_t>(floored);
    int32_t r = static_cast<int32_t>(floored);
    return (r % 2 == 0) ? r : r + 1;
}

}  // namespace

// ---------------------------------------------------------------------------
// dlopen() wrapper for the TFLite C API, mirroring pymicro_wakeword's ctypes
// binding. The shared library is resolved at runtime so the same binary works
// with the x86_64 and Raspberry Pi ARM builds.
// ---------------------------------------------------------------------------
using TfLiteStatus = int;

struct TfLiteQuantizationParams {
    float scale;
    int32_t zero_point;
};

class TfLiteLibrary {
public:
    static std::unique_ptr<TfLiteLibrary> load(const std::string& path) {
        auto lib = std::unique_ptr<TfLiteLibrary>(new TfLiteLibrary());
        if (path.empty()) {
            throw std::runtime_error(
                "Libreria TFLite (libtensorflowlite_c.so) non trovata. "
                "Imposta TFLITE_LIB_PATH o esegui cpp/scripts/setup_deps.sh");
        }
        lib->handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!lib->handle_) {
            throw std::runtime_error(
                std::string("Impossibile caricare libtensorflowlite_c.so: ") +
                dlerror());
        }
#define RESOLVE(member, symbol)                                              \
    lib->member = reinterpret_cast<decltype(lib->member)>(                   \
        dlsym(lib->handle_, symbol))
        RESOLVE(model_create_from_file, "TfLiteModelCreateFromFile");
        RESOLVE(model_delete, "TfLiteModelDelete");
        RESOLVE(interpreter_create, "TfLiteInterpreterCreate");
        RESOLVE(interpreter_delete, "TfLiteInterpreterDelete");
        RESOLVE(interpreter_allocate_tensors, "TfLiteInterpreterAllocateTensors");
        RESOLVE(interpreter_invoke, "TfLiteInterpreterInvoke");
        RESOLVE(interpreter_get_input_tensor, "TfLiteInterpreterGetInputTensor");
        RESOLVE(interpreter_get_output_tensor, "TfLiteInterpreterGetOutputTensor");
        RESOLVE(tensor_byte_size, "TfLiteTensorByteSize");
        RESOLVE(tensor_num_dims, "TfLiteTensorNumDims");
        RESOLVE(tensor_dim, "TfLiteTensorDim");
        RESOLVE(tensor_type, "TfLiteTensorType");
        RESOLVE(tensor_quantization_params, "TfLiteTensorQuantizationParams");
        RESOLVE(tensor_copy_from_buffer, "TfLiteTensorCopyFromBuffer");
        RESOLVE(tensor_copy_to_buffer, "TfLiteTensorCopyToBuffer");
#undef RESOLVE
        if (!lib->model_create_from_file || !lib->interpreter_create ||
            !lib->interpreter_invoke || !lib->tensor_copy_from_buffer ||
            !lib->tensor_copy_to_buffer) {
            throw std::runtime_error(
                "Simboli TFLite mancanti nella libreria caricata.");
        }
        return lib;
    }

    ~TfLiteLibrary() {
        if (handle_) dlclose(handle_);
    }

    void* (*model_create_from_file)(const char*) = nullptr;
    void (*model_delete)(void*) = nullptr;
    void* (*interpreter_create)(void*, void*) = nullptr;
    void (*interpreter_delete)(void*) = nullptr;
    TfLiteStatus (*interpreter_allocate_tensors)(void*) = nullptr;
    TfLiteStatus (*interpreter_invoke)(void*) = nullptr;
    void* (*interpreter_get_input_tensor)(void*, int32_t) = nullptr;
    void* (*interpreter_get_output_tensor)(void*, int32_t) = nullptr;
    size_t (*tensor_byte_size)(void*) = nullptr;
    int32_t (*tensor_num_dims)(void*) = nullptr;
    int32_t (*tensor_dim)(void*, int32_t) = nullptr;
    int32_t (*tensor_type)(void*) = nullptr;
    TfLiteQuantizationParams (*tensor_quantization_params)(void*) = nullptr;
    TfLiteStatus (*tensor_copy_from_buffer)(void*, const void*, size_t) = nullptr;
    TfLiteStatus (*tensor_copy_to_buffer)(void*, void*, size_t) = nullptr;

private:
    void* handle_ = nullptr;
};

std::string find_tflite_lib() {
    if (const char* env = std::getenv("TFLITE_LIB_PATH")) {
        if (file_exists(env)) return env;
    }
    const std::vector<std::string> candidates = {
        std::string(JARVIS_CPP_DIR) +
            "/third_party/libtensorflowlite_c/libtensorflowlite_c.so",
        "/usr/local/lib/libtensorflowlite_c.so",
        "/usr/lib/libtensorflowlite_c.so",
        "/usr/lib/x86_64-linux-gnu/libtensorflowlite_c.so",
    };
    for (const auto& c : candidates) {
        if (file_exists(c)) return c;
    }
    return "";
}

// ---------------------------------------------------------------------------
// Fixed-point frontend state. Same configuration as pymicro_features
// (src/micro_features.cpp -> init_cfg).
// ---------------------------------------------------------------------------
struct MicroFrontend::Impl {
    FrontendConfig cfg{};
    FrontendState state{};
    std::vector<int16_t> audio_buffer;

    Impl() {
        cfg.window.size_ms = 30;
        cfg.window.step_size_ms = 10;

        cfg.filterbank.num_channels = 40;
        cfg.filterbank.lower_band_limit = 125.0f;
        cfg.filterbank.upper_band_limit = 7500.0f;

        cfg.noise_reduction.smoothing_bits = 10;
        cfg.noise_reduction.even_smoothing = 0.025f;
        cfg.noise_reduction.odd_smoothing = 0.06f;
        cfg.noise_reduction.min_signal_remaining = 0.05f;

        cfg.pcan_gain_control.enable_pcan = 1;
        cfg.pcan_gain_control.strength = 0.95f;
        cfg.pcan_gain_control.offset = 80.0f;
        cfg.pcan_gain_control.gain_bits = 21;

        cfg.log_scale.enable_log = 1;
        cfg.log_scale.scale_shift = 6;

        FrontendPopulateState(&cfg, &state, 16000);
    }

    ~Impl() { FrontendFreeStateContents(&state); }

    void reset() {
        FrontendFreeStateContents(&state);
        FrontendPopulateState(&cfg, &state, 16000);
        audio_buffer.clear();
    }
};

// ---------------------------------------------------------------------------
// MicroFrontend
// ---------------------------------------------------------------------------
MicroFrontend::MicroFrontend() : impl_(new Impl()) {}
MicroFrontend::~MicroFrontend() = default;

std::size_t MicroFrontend::process_samples(
    const int16_t* samples, std::size_t count,
    std::vector<std::vector<float>>& out) {
    impl_->audio_buffer.insert(impl_->audio_buffer.end(), samples,
                               samples + count);

    std::size_t frames = 0;
    std::size_t consumed = 0;
    while (impl_->audio_buffer.size() - consumed >= samples_per_chunk_) {
        size_t samples_read = 0;
        FrontendOutput fo = FrontendProcessSamples(
            &impl_->state, impl_->audio_buffer.data() + consumed,
            samples_per_chunk_, &samples_read);
        consumed += samples_read;

        if (fo.size > 0) {
            std::vector<float> feats(fo.size);
            for (size_t i = 0; i < fo.size; ++i) {
                feats[i] = static_cast<float>(fo.values[i]) * kFloat32Scale;
            }
            out.push_back(std::move(feats));
            ++frames;
        }
    }

    if (consumed > 0) {
        impl_->audio_buffer.erase(impl_->audio_buffer.begin(),
                                  impl_->audio_buffer.begin() + consumed);
    }
    return frames;
}

void MicroFrontend::reset() { impl_->reset(); }

// ---------------------------------------------------------------------------
// WakeWordEngine
// ---------------------------------------------------------------------------
WakeWordEngine::WakeWordEngine(const std::string& model_json)
    : model_json_path_(model_json) {
    std::ifstream in(model_json);
    if (!in.is_open()) {
        throw std::runtime_error("Modello wake word non trovato: " +
                                 model_json);
    }
    auto config = nlohmann::json::parse(in);

    // The model path in the JSON is relative to the JSON file.
    const std::string model_file = config.value("model", "hey_jarvis.tflite");
    const auto slash = model_json.find_last_of('/');
    model_tflite_path_ =
        (slash == std::string::npos ? "" : model_json.substr(0, slash + 1)) +
        model_file;

    if (config.contains("micro") &&
        config["micro"].contains("sliding_window_size")) {
        sliding_window_size_ = config["micro"]["sliding_window_size"].get<int>();
    }

    lib_path_ = find_tflite_lib();
    reset_state();
}

WakeWordEngine::~WakeWordEngine() {
    if (tflite_ && interpreter_) tflite_->interpreter_delete(interpreter_);
    if (tflite_ && model_) tflite_->model_delete(model_);
}

void WakeWordEngine::reset_state() {
    if (tflite_) {
        if (interpreter_) tflite_->interpreter_delete(interpreter_);
        if (model_) tflite_->model_delete(model_);
    }
    interpreter_ = nullptr;
    model_ = nullptr;
    pending_features_.clear();
    probabilities_.clear();

    frontend_ = std::make_unique<MicroFrontend>();

    if (!tflite_) tflite_ = TfLiteLibrary::load(lib_path_);

    model_ = tflite_->model_create_from_file(model_tflite_path_.c_str());
    if (!model_) {
        throw std::runtime_error("Impossibile caricare il modello TFLite: " +
                                 model_tflite_path_);
    }
    interpreter_ = tflite_->interpreter_create(model_, nullptr);
    if (!interpreter_) {
        throw std::runtime_error("Impossibile creare l'interprete TFLite.");
    }
    if (tflite_->interpreter_allocate_tensors(interpreter_) != 0) {
        throw std::runtime_error("Errore nell'allocazione dei tensori TFLite.");
    }

    input_tensor_ = tflite_->interpreter_get_input_tensor(interpreter_, 0);
    output_tensor_ = tflite_->interpreter_get_output_tensor(interpreter_, 0);

    // Input shape is [1, stride, feature_size].
    stride_ = tflite_->tensor_dim(input_tensor_, 1);
    feature_size_ =
        static_cast<std::size_t>(tflite_->tensor_dim(input_tensor_, 2));

    const auto in_q = tflite_->tensor_quantization_params(input_tensor_);
    const auto out_q = tflite_->tensor_quantization_params(output_tensor_);
    input_scale_ = in_q.scale;
    input_zero_point_ = in_q.zero_point;
    input_type_ = tflite_->tensor_type(input_tensor_);
    output_scale_ = out_q.scale;
    output_zero_point_ = out_q.zero_point;
    output_type_ = tflite_->tensor_type(output_tensor_);
    input_bytes_ = tflite_->tensor_byte_size(input_tensor_);
    output_bytes_ = tflite_->tensor_byte_size(output_tensor_);
}

bool WakeWordEngine::detect_in_stream(const std::vector<uint8_t>& audio_chunk,
                                      float threshold) {
    const int16_t* samples =
        reinterpret_cast<const int16_t*>(audio_chunk.data());
    const std::size_t count = audio_chunk.size() / 2;

    std::vector<std::vector<float>> frames;
    frontend_->process_samples(samples, count, frames);

    bool detected = false;
    for (auto& f : frames) {
        if (process_streaming_prob(f) > threshold) detected = true;
    }
    return detected;
}

float WakeWordEngine::detect_barge_in(const std::vector<uint8_t>& audio_chunk) {
    const int16_t* samples =
        reinterpret_cast<const int16_t*>(audio_chunk.data());
    const std::size_t count = audio_chunk.size() / 2;

    std::vector<std::vector<float>> frames;
    frontend_->process_samples(samples, count, frames);

    float max_prob = 0.0f;
    for (auto& f : frames) {
        float p = process_streaming_prob(f);
        if (p > max_prob) max_prob = p;
    }
    return max_prob;
}

float WakeWordEngine::process_streaming_prob(
    const std::vector<float>& features) {
    if (!interpreter_ || !model_) return 0.0f;

    pending_features_.push_back(features);
    if (static_cast<int>(pending_features_.size()) < stride_) return 0.0f;

    // Concatenate the `stride` frames along the feature axis (like numpy's
    // np.concatenate(..., axis=1)) and quantize to int8.
    std::vector<int8_t> quant(input_bytes_);
    for (int i = 0; i < stride_; ++i) {
        for (std::size_t j = 0; j < feature_size_; ++j) {
            const float v = pending_features_[static_cast<size_t>(i)][j];
            const float qf =
                v / input_scale_ + static_cast<float>(input_zero_point_);
            const int32_t rounded = round_half_even(qf);
            quant[static_cast<size_t>(i) * feature_size_ + j] =
                static_cast<int8_t>(std::clamp<int32_t>(rounded, -128, 127));
        }
    }
    pending_features_.clear();

    if (tflite_->tensor_copy_from_buffer(input_tensor_, quant.data(),
                                         input_bytes_) != 0) {
        return 0.0f;
    }
    if (tflite_->interpreter_invoke(interpreter_) != 0) return 0.0f;

    std::vector<uint8_t> output(output_bytes_);
    if (tflite_->tensor_copy_to_buffer(output_tensor_, output.data(),
                                       output_bytes_) != 0) {
        return 0.0f;
    }

    float result = 0.0f;
    if (output_type_ == 3) {  // kTfLiteUInt8
        result = (static_cast<int32_t>(output[0]) - output_zero_point_) *
                 output_scale_;
    } else if (output_type_ == 9) {  // kTfLiteInt8
        result = (static_cast<int32_t>(static_cast<int8_t>(output[0])) -
                  output_zero_point_) *
                 output_scale_;
    } else {
        return 0.0f;
    }

    probabilities_.push_back(result);
    if (static_cast<int>(probabilities_.size()) < sliding_window_size_) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (float p : probabilities_) sum += p;
    return sum / static_cast<float>(probabilities_.size());
}

}  // namespace jarvis
