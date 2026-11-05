#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace jarvis {

// One chat-completions message (role/content pair).
struct ChatMessage {
    std::string role;
    std::string content;
};

// Pull-based source of streaming TTS PCM chunks (16-bit mono @ 24 kHz).
// nextChunk() blocks until data is available and returns false at end of
// stream or on a transport error. Chunks are always non-empty.
class TtsChunkSource {
public:
    virtual ~TtsChunkSource() = default;
    virtual bool nextChunk(std::vector<uint8_t>& out) = 0;
};

}  // namespace jarvis
