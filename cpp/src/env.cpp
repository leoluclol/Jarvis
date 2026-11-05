#include "jarvis/env.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace jarvis {

namespace {

std::string trim(const std::string& s) {
    const auto is_space = [](unsigned char c) {
        return std::isspace(c) != 0;
    };
    auto b = std::find_if_not(s.begin(), s.end(), is_space);
    auto e = std::find_if_not(s.rbegin(), s.rend(), is_space).base();
    return (b < e) ? std::string(b, e) : std::string();
}

// Removes surrounding double or single quotes, if present.
std::string unquote(const std::string& s) {
    if (s.size() >= 2 &&
        ((s.front() == '"' && s.back() == '"') ||
         (s.front() == '\'' && s.back() == '\''))) {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

}  // namespace

std::unordered_map<std::string, std::string> load_dotenv(const std::string& path) {
    std::unordered_map<std::string, std::string> out;
    std::ifstream in(path);
    if (!in.is_open()) return out;

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        if (line.rfind("export ", 0) == 0) line = line.substr(7);

        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));
        if (key.empty()) continue;

        // Strip trailing inline comments, but keep '#' inside quotes.
        if (value.size() >= 2 && value.front() == '"') {
            value = unquote(value);
        } else {
            auto hash = value.find(" #");
            if (hash != std::string::npos) value = value.substr(0, hash);
            value = unquote(trim(value));
        }
        out[key] = value;
    }
    return out;
}

std::string get_env(const std::unordered_map<std::string, std::string>& dotenv,
                    const char* key, const std::string& fallback) {
    if (const char* v = std::getenv(key)) return v;
    auto it = dotenv.find(key);
    if (it != dotenv.end()) return it->second;
    return fallback;
}

}  // namespace jarvis
