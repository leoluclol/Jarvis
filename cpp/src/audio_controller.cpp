#include "jarvis/audio_controller.h"

#include <portaudio.h>

#ifdef JARVIS_HAS_ALSA
#include <alsa/asoundlib.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstdlib>
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

bool name_matches(const char* name, const std::vector<const char*>& patterns) {
    if (!name) return false;
    std::string n(name);
    std::transform(n.begin(), n.end(), n.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    for (const char* p : patterns) {
        if (n.find(p) != std::string::npos) return true;
    }
    return false;
}

// Scans PortAudio devices for a usable one (input or output), preferring a
// name match. Returns paNoDevice when nothing is usable.
int find_portaudio_device(bool for_input,
                          const std::vector<const char*>& patterns) {
    int count = Pa_GetDeviceCount();
    if (count < 0) return paNoDevice;
    for (int i = 0; i < count; ++i) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(i);
        if (!info) continue;
        int ch = for_input ? info->maxInputChannels : info->maxOutputChannels;
        if (ch <= 0) continue;
        if (patterns.empty() || name_matches(info->name, patterns)) return i;
    }
    return paNoDevice;
}

}  // namespace

AudioController::AudioController(const Config& config) : config_(config) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        throw std::runtime_error(
            "Errore di inizializzazione PortAudio: " +
            std::string(Pa_GetErrorText(err)));
    }

    // Input: system default, else name match (USB mic), else any usable.
    int in_dev = Pa_GetDefaultInputDevice();
    if (in_dev == paNoDevice) {
        in_dev = find_portaudio_device(true, {"camera", "usb", "mic"});
    }
    if (in_dev == paNoDevice) {
        in_dev = find_portaudio_device(true, {});
    }

    if (in_dev != paNoDevice) {
        const PaDeviceInfo* info = Pa_GetDeviceInfo(in_dev);
        PaStreamParameters in{};
        in.device = in_dev;
        in.channelCount = config_.channels;
        in.sampleFormat = paInt16;
        in.suggestedLatency =
            info ? info->defaultLowInputLatency : 0.1;
        in.hostApiSpecificStreamInfo = nullptr;

        err = Pa_OpenStream(&mic_stream_, &in, nullptr, config_.rate,
                            config_.chunk, paClipOff, nullptr, nullptr);
        if (err != paNoError) {
            Pa_Terminate();
            throw std::runtime_error(
                "Errore di apertura microfono: " +
                std::string(Pa_GetErrorText(err)));
        }
        log_info("🎙️ Microfono PortAudio: " +
                 std::string(info ? info->name : "?"));
    } else {
#ifdef JARVIS_HAS_ALSA
        try {
            open_alsa_mic();
        } catch (...) {
            Pa_Terminate();
            throw;
        }
#else
        Pa_Terminate();
        throw std::runtime_error(
            "Nessun dispositivo di input audio trovato via PortAudio. "
            "Installa libasound2-dev e ricompila per il fallback ALSA.");
#endif
    }
}

AudioController::~AudioController() { stop(); }

#ifdef JARVIS_HAS_ALSA
std::vector<std::string> AudioController::alsa_device_candidates() {
    std::vector<std::string> candidates;
    const char* env = std::getenv("JARVIS_ALSA_INPUT_DEVICE");
    if (env) candidates.push_back(env);

    // Enumerate ALSA PCMs: prefer the first capture-capable hw device
    // (tried as plughw so ALSA converts rate/format when needed).
    void** hints = nullptr;
    if (snd_device_name_hint(-1, "pcm", &hints) >= 0) {
        for (void** h = hints; h && *h; ++h) {
            char* name = snd_device_name_get_hint(*h, "NAME");
            if (!name) continue;
            std::string n(name);
            free(name);
            if (n.rfind("hw:", 0) == 0) {
                candidates.push_back("plughw:" + n.substr(3));
                candidates.push_back(n);
            }
        }
        snd_device_name_free_hint(hints);
    }
    candidates.push_back("default");
    return candidates;
}

void AudioController::open_alsa_mic() {
    for (const std::string& dev : alsa_device_candidates()) {
        snd_pcm_t* pcm = nullptr;
        if (snd_pcm_open(&pcm, dev.c_str(), SND_PCM_STREAM_CAPTURE, 0) < 0) {
            continue;
        }
        if (snd_pcm_set_params(pcm, SND_PCM_FORMAT_S16_LE,
                               SND_PCM_ACCESS_RW_INTERLEAVED,
                               config_.channels, config_.rate, 1, 500000) < 0) {
            snd_pcm_close(pcm);
            continue;
        }
        log_info("🎙️ Microfono ALSA: " + dev);
        alsa_pcm_ = pcm;
        return;
    }
    throw std::runtime_error(
        "Impossibile aprire un microfono ALSA (verifica `arecord -l`).");
}
#endif  // JARVIS_HAS_ALSA

std::vector<uint8_t> AudioController::read_mic_chunk() {
    if (mic_stream_) {
        std::vector<uint8_t> buf(
            static_cast<size_t>(config_.chunk) * config_.channels * 2);
        PaError err = Pa_ReadStream(mic_stream_, buf.data(), config_.chunk);
        if (err != paNoError && err != paInputOverflowed) return {};
        return buf;
    }

#ifdef JARVIS_HAS_ALSA
    if (alsa_pcm_) {
        snd_pcm_t* pcm = static_cast<snd_pcm_t*>(alsa_pcm_);
        snd_pcm_state_t st = snd_pcm_state(pcm);
        if (st != SND_PCM_STATE_RUNNING && st != SND_PCM_STATE_PREPARED) {
            snd_pcm_prepare(pcm);
        }
        std::vector<uint8_t> buf(
            static_cast<size_t>(config_.chunk) * config_.channels * 2);
        size_t got = 0;
        while (got < buf.size()) {
            snd_pcm_sframes_t frames = snd_pcm_readi(
                pcm, buf.data() + got,
                static_cast<snd_pcm_uframes_t>(
                    (buf.size() - got) / (config_.channels * 2)));
            if (frames < 0) {
                if (frames == -EPIPE) snd_pcm_prepare(pcm);  // xrun
                return {};
            }
            got += static_cast<size_t>(frames) * config_.channels * 2;
        }
        return buf;
    }
#endif

    return {};
}

void AudioController::start_mic() {
    if (mic_stream_) {
        if (!mic_running_ && Pa_StartStream(mic_stream_) == paNoError) {
            mic_running_ = true;
        }
        return;
    }
#ifdef JARVIS_HAS_ALSA
    if (alsa_pcm_) {
        snd_pcm_t* pcm = static_cast<snd_pcm_t*>(alsa_pcm_);
        if (snd_pcm_state(pcm) != SND_PCM_STATE_RUNNING) snd_pcm_prepare(pcm);
    }
#endif
}

void AudioController::stop_mic() {
    if (mic_stream_) {
        if (mic_running_) {
            Pa_StopStream(mic_stream_);
            mic_running_ = false;
        }
        return;
    }
#ifdef JARVIS_HAS_ALSA
    if (alsa_pcm_) snd_pcm_drop(static_cast<snd_pcm_t*>(alsa_pcm_));
#endif
}

void AudioController::clear_mic_queue() { mic_queue_.clear(); }

bool AudioController::play_tts_stream_with_barge_in(
    TtsChunkSource& tts,
    const std::function<float(const std::vector<uint8_t>&)>& barge_in_prob) {
    clear_mic_queue();

    // Output: system default, else name match (bcm2835 jack / HDMI), else any.
    int out_dev = Pa_GetDefaultOutputDevice();
    if (out_dev == paNoDevice) {
        out_dev = find_portaudio_device(false, {"bcm2835", "headphone", "hdmi"});
    }
    if (out_dev == paNoDevice) {
        out_dev = find_portaudio_device(false, {});
    }
    if (out_dev == paNoDevice) {
        log_error("Nessun dispositivo di output audio trovato.");
        return false;
    }

    PaStream* out_stream = nullptr;
    const PaDeviceInfo* out_info = Pa_GetDeviceInfo(out_dev);
    PaStreamParameters out{};
    out.device = out_dev;
    out.channelCount = 1;
    out.sampleFormat = paInt16;
    out.suggestedLatency =
        out_info ? out_info->defaultLowOutputLatency : 0.1;
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
#ifdef JARVIS_HAS_ALSA
    if (alsa_pcm_) {
        snd_pcm_drop(static_cast<snd_pcm_t*>(alsa_pcm_));
        snd_pcm_close(static_cast<snd_pcm_t*>(alsa_pcm_));
        alsa_pcm_ = nullptr;
    }
#endif
    Pa_Terminate();
}

}  // namespace jarvis
