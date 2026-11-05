#include "jarvis/assistant_orchestrator.h"

#include <chrono>
#include <thread>

#include "jarvis/logger.h"
#include "jarvis/signals.h"

namespace jarvis {

AssistantOrchestrator::AssistantOrchestrator()
    : config_(Config::load()),
      hw_(config_),
      audio_(config_),
      wakeword_(config_.wake_word_model_json),
      vad_(config_),
      llm_(config_.api_key),
      memory_(config_.max_history) {
    hw_.setup();
    hw_.turn_off_all();
}

AssistantOrchestrator::~AssistantOrchestrator() { shutdown(); }

void AssistantOrchestrator::run() {
    log_info("⚙️ Inizializzazione microWakeWord ('" +
             config_.wake_word_model_name + "')...");
    log_info("🧠 Caricamento modello neurale Silero VAD (ONNX runtime)...");
    audio_.start_mic();

    try {
        while (!interrupted()) {
            standby_loop();
            if (interrupted()) break;
            active_conversation_loop();
        }
    } catch (const std::exception& e) {
        log_error("Errore: " + std::string(e.what()));
    } catch (...) {
        log_error("Errore sconosciuto.");
    }

    shutdown();
    log_info("\nSpegnimento Jarvis...");
}

float AssistantOrchestrator::handle_barge_in(
    const std::vector<uint8_t>& pcm_chunk) {
    return wakeword_.detect_barge_in(pcm_chunk);
}

void AssistantOrchestrator::standby_loop() {
    log_emphasize(
        "🤖 Jarvis è in STANDBY. Pronuncia \"Hey Jarvis\" per attivare la "
        "conversazione.");
    wakeword_.reset_state();

    while (!interrupted()) {
        std::vector<uint8_t> pcm = audio_.read_mic_chunk();
        if (pcm.empty()) continue;

        if (wakeword_.detect_in_stream(pcm, 0.75f)) {
            log_emphasize(
                "\n✨ Parola d'ordine rilevata! Modalità conversazione ATTIVA. "
                "✨");
            wakeword_.reset_state();
            audio_.clear_mic_queue();
            return;
        }
    }
}

void AssistantOrchestrator::active_conversation_loop() {
    bool in_active_conversation = true;

    while (in_active_conversation && !interrupted()) {
        hw_.set_listen_led(true);
        std::vector<uint8_t> audio_io = vad_.record_until_silence(audio_);
        hw_.set_listen_led(false);

        if (audio_io.empty()) break;

        audio_.stop_mic();
        hw_.set_think_led(true);
        log_info("🧠 Trascrizione in corso");

        std::string user_text;
        try {
            user_text = llm_.transcribe_audio(audio_io);
        } catch (const std::exception& e) {
            log_error("Errore di trascrizione: " + std::string(e.what()));
            hw_.set_think_led(false);
            audio_.start_mic();
            continue;
        }
        hw_.set_think_led(false);

        log_info("👤 Tu: " + user_text);

        if (memory_.is_stop_command(user_text)) {
            log_info("🤖 Comando di spegnimento riconosciuto.");
            break;
        }

        memory_.add_user_message(user_text);
        log_info("🧠 Elaborazione risposta...");

        std::string ai_text;
        try {
            ai_text = llm_.generate_chat_response(memory_.get_history());
        } catch (const std::exception& e) {
            log_error("Errore LLM: " + std::string(e.what()));
            audio_.start_mic();
            continue;
        }

        if (memory_.is_hallucination(ai_text)) {
            memory_.revert_last_user_message();
            continue;
        }

        log_info("🤖 Jarvis: " + ai_text);
        memory_.add_assistant_message(ai_text);

        audio_.start_mic();
        log_info(
            "🗣️ Riproduzione in corso (pronuncia 'Hey Jarvis' per "
            "interrompere)...");

        bool interrupted = false;
        try {
            auto tts = llm_.stream_tts(ai_text);
            interrupted = audio_.play_tts_stream_with_barge_in(
                *tts, [this](const std::vector<uint8_t>& chunk) {
                    return handle_barge_in(chunk);
                });
        } catch (const std::exception& e) {
            log_error("Errore TTS: " + std::string(e.what()));
            continue;
        }

        if (interrupted) {
            wakeword_.reset_state();
            hw_.set_listen_led(true);
            log_emphasize("\n👂 Prontissimo! Dimmi pure il nuovo comando...");
            continue;
        } else {
            log_emphasize("\n👂 In attesa del prossimo turno...");
        }
    }

    audio_.start_mic();
    log_emphasize("\n🤖 Torno in STANDBY. In attesa di 'Hey Jarvis'...");
}

void AssistantOrchestrator::shutdown() {
    audio_.stop();
    hw_.cleanup();
}

}  // namespace jarvis
