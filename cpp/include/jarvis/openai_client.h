#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "jarvis/types.h"

namespace jarvis {

// OpenAI REST client (libcurl + nlohmann/json). Mirrors src2/openai_client.py:
// Whisper transcription, chat completions, and streaming PCM TTS.
class OpenAIClient {
public:
    explicit OpenAIClient(std::string api_key);
    ~OpenAIClient();

    // wav_bytes: complete WAV file contents. Returns the transcribed text.
    std::string transcribe_audio(const std::vector<uint8_t>& wav_bytes);

    std::string generate_chat_response(const std::vector<ChatMessage>& history);

    // Returns a blocking pull-source of 24 kHz PCM chunks. Closing the source
    // (or destroying it) cancels the in-flight transfer.
    std::unique_ptr<TtsChunkSource> stream_tts(const std::string& text);

private:
    std::string api_key_;
};

}  // namespace jarvis
