#pragma once

#include <iostream>
#include <string>

namespace jarvis {

// Small logging helpers. Emoji prefixes mirror the original Python output.
inline void log_info(const std::string& msg) { std::cout << msg << std::endl; }
inline void log_error(const std::string& msg) { std::cerr << msg << std::endl; }
inline void log_emphasize(const std::string& msg) {
    std::cout << "\n" << msg << std::endl;
}

}  // namespace jarvis
