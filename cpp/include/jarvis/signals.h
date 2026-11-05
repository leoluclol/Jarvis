#pragma once

#include <atomic>

namespace jarvis {

// Global interrupt flag, set by the SIGINT/SIGTERM handler and polled by the
// main loops (C++ equivalent of Python's KeyboardInterrupt handling).
inline std::atomic<bool>& interrupted_flag() {
    static std::atomic<bool> flag{false};
    return flag;
}

inline bool interrupted() { return interrupted_flag().load(); }
inline void request_stop() { interrupted_flag().store(true); }

}  // namespace jarvis
