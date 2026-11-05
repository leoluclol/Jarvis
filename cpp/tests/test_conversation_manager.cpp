#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

#include "jarvis/conversation_manager.h"

using jarvis::ConversationManager;

namespace {
int failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::cerr << "CHECK failed: " #cond " @ line " << __LINE__     \
                      << std::endl;                                        \
            ++failures;                                                    \
        }                                                                  \
    } while (0)
}  // namespace

int main() {
    // System prompt + history trimming.
    ConversationManager cm(6);
    CHECK(cm.get_history().size() == 1);
    CHECK(cm.get_history()[0].role == "system");

    cm.add_user_message("Buongiorno");
    cm.add_assistant_message("Ciao!");
    CHECK(cm.get_history().size() == 3);

    // Stop commands (with punctuation / case normalization).
    CHECK(cm.is_stop_command("Stop!"));
    CHECK(cm.is_stop_command("basta."));
    CHECK(cm.is_stop_command("Stai zitto."));
    CHECK(cm.is_stop_command("per favore fermati"));
    CHECK(!cm.is_stop_command("Puoi fermarti domani mattina?"));
    CHECK(!cm.is_stop_command("Che tempo fa oggi?"));
    CHECK(!cm.is_stop_command(""));

    // Hallucination guards (raw substring check, like the Python version).
    CHECK(cm.is_hallucination("Guarda il video su amara.org"));
    CHECK(cm.is_hallucination("qTSS non e' disponibile"));
    CHECK(cm.is_hallucination("amara.org"));
    CHECK(!cm.is_hallucination("Una risposta normale"));

    // revert_last_user_message() drops the most recent message (used to undo
    // the user turn when the LLM output is a hallucination).
    cm.revert_last_user_message();
    CHECK(cm.get_history().back().role == "user");
    CHECK(cm.get_history().back().content == "Buongiorno");

    // History trim keeps the system prompt plus the last N messages.
    for (int i = 0; i < 20; ++i) {
        cm.add_user_message("u" + std::to_string(i));
        cm.add_assistant_message("a" + std::to_string(i));
    }
    CHECK(cm.get_history().size() == 7);  // system + 6
    CHECK(cm.get_history().front().role == "system");
    CHECK(cm.get_history().back().role == "assistant");
    CHECK(cm.get_history().back().content == "a19");

    // clear_history() resets to just the system prompt.
    cm.clear_history();
    CHECK(cm.get_history().size() == 1);
    CHECK(cm.get_history()[0].role == "system");

    if (failures > 0) {
        std::cerr << failures << " CHECK(s) failed" << std::endl;
        return 1;
    }
    std::cout << "test_conversation_manager: OK" << std::endl;
    return 0;
}
