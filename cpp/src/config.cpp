#include "jarvis/config.h"

#include <cstdlib>
#include <stdexcept>

#include "jarvis/env.h"
#include "jarvis/logger.h"

namespace jarvis {

#ifndef JARVIS_REPO_ROOT
#define JARVIS_REPO_ROOT "."
#endif
#ifndef JARVIS_CPP_DIR
#define JARVIS_CPP_DIR "."
#endif

Config Config::load() {
    Config cfg;

    // Compile-time repo layout (baked in by CMake), overridable via env vars
    // for unusual installs.
    const char* root_env = std::getenv("JARVIS_ROOT");
    cfg.root_dir = root_env ? root_env : JARVIS_REPO_ROOT;
    cfg.cpp_dir = JARVIS_CPP_DIR;

    cfg.model_dir = cfg.root_dir + "/models";
    cfg.audio_dir = cfg.root_dir + "/audio";

    // .env lives at the repo root, next to the Python sources.
    auto dotenv = load_dotenv(cfg.root_dir + "/.env");
    cfg.api_key = get_env(dotenv, "OPENAI_API_KEY");
    if (cfg.api_key.empty()) {
        throw std::runtime_error(
            "OPENAI_API_KEY non trovata: crea un file .env con "
            "OPENAI_API_KEY=... nella root del progetto.");
    }

    cfg.silero_model_path = cfg.model_dir + "/silero_vad.onnx";
    cfg.wake_word_model_json = cfg.cpp_dir + "/models/hey_jarvis.json";
    cfg.wake_word_model_tflite = cfg.cpp_dir + "/models/hey_jarvis.tflite";

    // Optional overrides.
    const char* model_dir = std::getenv("JARVIS_MODEL_DIR");
    if (model_dir) {
        cfg.model_dir = model_dir;
        cfg.silero_model_path = cfg.model_dir + "/silero_vad.onnx";
    }

    return cfg;
}

}  // namespace jarvis
