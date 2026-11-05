#include "jarvis/audio_controller.h"

#include <portaudio.h>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

#include "jarvis/logger.h"

namespace jarvis {

namespace {

double now_seconds() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

}  // namespace

AudioController::AudioController(const Config& config) : config_(config) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error(
            "Errore di inizializzazione PortAudio: " +
            std::string(Pa_GetErrorText(err)));
    }

    PaStreamParameters in;
    in.device = Pa_GetDefaultInputDevice();
    if (in.device == paNoDevice) {
        Pa_Terminate();
        throw std::runtime_error("Nessun dispositivo di input audio trovato.");
    }
    in.channelCount = config_.channels;
    in.sampleFormat = paInt16;
    in.suggestedLatency =
        Pa_GetDeviceInfo(in.device)->defaultLowInputLatency;
    in.hostApiSpecificStreamInfo = nullptr;

    err = Pa_OpenStream(&mic_stream_, &in, nullptr, config_.rate,
                        config_.chunk, paClipOff, nullptr, nullptr);
    if (err != paNoError) {
        Pa_Terminate();
        throw std::runtime_error(
            "Errore di apertura microfono: " +
            std::string(Pa_GetErrorText(err)));
    }
}

AudioController::~AudioController() { stop(); }

std::vector<uint8_t> AudioController::read_mic_chunk() {
    if (!mic_stream_) return {};
    std::vector<uint8_t> buf(
        static_cast<size_t>(config_.chunk) * config_.channels * 2);
    PaError err = Pa_ReadStream(mic_stream_, buf.data(), config_.chunk);
    if (err != paNoError && err != paInputOverflowed) return {};
    return buf;
}

void AudioController::start_mic() {
    if (!mic_stream_ || mic_running_) return;
    if (Pa_StartStream(mic_stream_) == paNoError) mic_running_ = true;
}

void AudioController::stop_mic() {
    if (mic_stream_ && mic_running_) {
        Pa_StopStream(mic_stream_);
        mic_running_ = false;
    }
}

void AudioController::clear_mic_queue() { mic_queue_.clear(); }

bool AudioController::play_tts_stream_with_barge_in(
    TtsChunkSource& tts,
    const std::function<float(const std::vector<uint8_t>&)>& barge_in_prob) {
    clear_mic_queue();

    PaStream* out_stream = nullptr;
    PaStreamParameters out;
    out.device = Pa_GetDefaultOutputDevice();
    if (out.device == paNoDevice) {
        log_error("Nessun dispositivo di output audio trovato.");
        return false;
    }
    out.channelCount = 1;
    out.sampleFormat = paInt16;
    out.suggestedLatency =
        Pa_GetDeviceInfo(out.device)->defaultLowOutputLatency;
    out.hostApiSpecificStreamInfo = nullptr;

    PaError err = Pa_OpenStream(&out_stream, nullptr, &out,
                                config_.openai_tts_rate, 2048, paClipOff,
                                nullptr, nullptr);
    if (err != paNoError) {
        log_error("Errore di apertura altoparlante: " +
                  std::string(Pa_GetErrorText(err)));
        return false;
    }
    Pa_StartStream(out_stream);

    // Reader thread keeps feeding the mic queue while we play.
    std::atomic<bool> stop_reader{false};
    std::thread reader([this, &stop_reader]() {
        while (!stop_reader.load()) {
            std::vector<uint8_t> pcm = read_mic_chunk();
            if (!pcm.empty()) mic_queue_.push(std::move(pcm));
        }
    });

    bool interrupted = false;
    std::vector<uint8_t> audio_buffer;
    constexpr size_t SAFE_CHUNK_SIZE = 4096;
    constexpr size_t PREBUFFER_SIZE = 16384;
    bool is_playing = false;
    double playback_start_time = 0.0;
    constexpr double GRACE_PERIOD = 1.0;
    constexpr float BARGE_IN_THRESHOLD = 0.6f;

    auto detect_bargein = [&]() -> bool {
        std::vector<uint8_t> pcm;
        while (mic_queue_.popNowait(pcm)) {
            float prob = barge_in_prob(pcm);
            if (is_playing &&
                (now_seconds() - playback_start_time > GRACE_PERIOD) &&
                prob > BARGE_IN_THRESHOLD) {
                log_emphasize("\n⚡ BARGE-IN RILEVATO! (Prob: " +
                              std::to_string(prob) + ")... ⚡");
                return true;
            }
        }
        return false;
    };

    // Phase 1: stream TTS chunks and play them as they arrive.
    std::vector<uint8_t> chunk;
    while (tts.nextChunk(chunk)) {
        if (!chunk.empty()) {
            audio_buffer.insert(audio_buffer.end(), chunk.begin(), chunk.end());
        }
        if (!is_playing && audio_buffer.size() >= PREBUFFER_SIZE) {
            is_playing = true;
            playback_start_time = now_seconds();
        }
        if (is_playing) {
            while (audio_buffer.size() >= SAFE_CHUNK_SIZE && !interrupted) {
                Pa_WriteStream(out_stream, audio_buffer.data(),
                               static_cast<unsigned long>(SAFE_CHUNK_SIZE / 2));
                audio_buffer.erase(audio_buffer.begin(),
                                   audio_buffer.begin() + SAFE_CHUNK_SIZE);
                if (detect_bargein()) interrupted = true;
            }
        }
        if (!interrupted && detect_bargein()) interrupted = true;
        if (interrupted) break;
    }

    // Phase 2: flush whatever was buffered but not yet played.
    if (!interrupted) {
        if (!is_playing) {
            is_playing = true;
            playback_start_time = now_seconds();
        }
        while (audio_buffer.size() >= 2 && !interrupted) {
            size_t n = std::min(SAFE_CHUNK_SIZE, audio_buffer.size());
            if (n % 2 != 0) --n;
            if (n == 0) break;
            Pa_WriteStream(out_stream, audio_buffer.data(),
                           static_cast<unsigned long>(n / 2));
            audio_buffer.erase(audio_buffer.begin(),
                               audio_buffer.begin() + n);
            if (detect_bargein()) interrupted = true;
        }
    }

    stop_reader.store(true);
    if (reader.joinable()) reader.join();
    Pa_StopStream(out_stream);
    Pa_CloseStream(out_stream);
    clear_mic_queue();
    return interrupted;
}

void AudioController::stop() {
    stop_mic();
    if (mic_stream_) {
        Pa_CloseStream(mic_stream_);
        mic_stream_ = nullptr;
    }
    Pa_Terminate();
}

}  // namespace jarvis
