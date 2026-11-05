#pragma once

#include <string>
#include <unordered_map>

namespace jarvis {

// Minimal .env loader (KEY=VALUE lines, '#' comments, optional 'export ',
// no shell interpolation). Mirrors python-dotenv for the fields we use.
std::unordered_map<std::string, std::string> load_dotenv(const std::string& path);

// Returns env[KEY] if set, otherwise the value from the .env file.
std::string get_env(
    const std::unordered_map<std::string, std::string>& dotenv,
    const char* key,
    const std::string& fallback = "");

}  // namespace jarvis
