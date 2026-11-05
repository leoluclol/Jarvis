#include "jarvis/conversation_manager.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace jarvis {

namespace {

const std::vector<std::string> kStopCommands = {
    "stop", "basta", "stai zitto", "zitto", "fermati", "spegniti"};

}  // namespace

ConversationManager::ConversationManager(int max_history)
    : max_history_(max_history) {
    clear_history();
}

void ConversationManager::add_user_message(const std::string& text) {
    history_.push_back({"user", text});
}

void ConversationManager::add_assistant_message(const std::string& text) {
    history_.push_back({"assistant", text});
    trim_history();
}

void ConversationManager::clear_history() {
    history_.clear();
    history_.push_back(
        {"system",
         "Sei Jarvis, un assistente vocale per la casa. Sii conciso e diretto. "
         "Rispondi in italiano."});
}

void ConversationManager::revert_last_user_message() {
    if (!history_.empty()) history_.pop_back();
}

std::string ConversationManager::normalize(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        const int lower = std::tolower(c);
        if (std::isalnum(lower) || lower == '_' || std::isspace(lower)) {
            out.push_back(static_cast<char>(lower));
        }
    }
    // Trim surrounding whitespace.
    auto b = out.find_first_not_of(" \t\r\n");
    auto e = out.find_last_not_of(" \t\r\n");
    return (b == std::string::npos) ? "" : out.substr(b, e - b + 1);
}

bool ConversationManager::is_stop_command(const std::string& text) const {
    const std::string clean = normalize(text);
    if (clean.empty()) return false;

    bool contains_stop = false;
    for (const auto& cmd : kStopCommands) {
        if (clean.find(cmd) != std::string::npos) {
            contains_stop = true;
            break;
        }
    }
    if (!contains_stop) return false;

    // Word count (whitespace separated) must be <= 4.
    std::istringstream iss(clean);
    std::string word;
    int words = 0;
    while (iss >> word) ++words;
    return words <= 4;
}

bool ConversationManager::is_hallucination(const std::string& text) const {
    // Mirrors the Python: lowercase the raw text and check the substrings
    // verbatim (the '.' in "amara.org" must be preserved).
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lower.find("amara.org") != std::string::npos ||
           lower.find("qtss") != std::string::npos;
}

void ConversationManager::trim_history() {
    if (static_cast<int>(history_.size()) > max_history_ + 1) {
        std::vector<ChatMessage> trimmed;
        trimmed.reserve(max_history_ + 1);
        trimmed.push_back(history_.front());
        trimmed.insert(trimmed.end(), history_.end() - max_history_,
                       history_.end());
        history_ = std::move(trimmed);
    }
}

}  // namespace jarvis
