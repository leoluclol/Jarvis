#pragma once

#include <string>
#include <vector>

#include "jarvis/types.h"

namespace jarvis {

// Rolling conversation memory with stop-command and hallucination guards.
// Mirrors src2/conversation_manager.py.
class ConversationManager {
public:
    explicit ConversationManager(int max_history);

    void add_user_message(const std::string& text);
    void add_assistant_message(const std::string& text);
    const std::vector<ChatMessage>& get_history() const { return history_; }
    void clear_history();
    void revert_last_user_message();

    bool is_stop_command(const std::string& text) const;
    bool is_hallucination(const std::string& text) const;

private:
    void trim_history();
    static std::string normalize(const std::string& text);

    int max_history_;
    std::vector<ChatMessage> history_;
};

}  // namespace jarvis
