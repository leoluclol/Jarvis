#pragma once

#include <cstdint>
#include <vector>

#include "jarvis/audio_controller.h"
#include "jarvis/config.h"
#include "jarvis/conversation_manager.h"
#include "jarvis/hardware_controller.h"
#include "jarvis/openai_client.h"
#include "jarvis/vad.h"
#include "jarvis/wakeword_engine.h"

namespace jarvis {

// High-level state machine tying together the hardware, audio, wake-word, VAD,
// LLM and memory components. Mirrors src2/assistant_orchestrator.py.
class AssistantOrchestrator {
public:
    AssistantOrchestrator();
    ~AssistantOrchestrator();

    // Runs standby loop -> active conversation loop until interrupted.
    void run();

private:
    float handle_barge_in(const std::vector<uint8_t>& pcm_chunk);
    void standby_loop();
    void active_conversation_loop();
    void shutdown();

    Config config_;
    HardwareController hw_;
    AudioController audio_;
    WakeWordEngine wakeword_;
    VoiceActivityDetector vad_;
    OpenAIClient llm_;
    ConversationManager memory_;
};

}  // namespace jarvis
