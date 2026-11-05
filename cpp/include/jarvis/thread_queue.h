#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>

namespace jarvis {

// Bounded-free thread-safe queue used for streaming mic audio and TTS chunks.
// `close()` wakes any blocked popper; a pop on a closed-and-empty queue
// returns false.
template <typename T>
class ThreadQueue {
public:
    void push(T&& value) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (closed_) return;
            queue_.push_back(std::move(value));
        }
        cv_.notify_one();
    }

    // Blocks until an item is available, the queue is closed, or timeout_ms
    // elapses (only honoured when timeout_ms >= 0). Returns false otherwise.
    bool pop(T& out, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mu_);
        if (queue_.empty() && !closed_ && timeout_ms >= 0) {
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms));
        } else if (queue_.empty() && !closed_) {
            cv_.wait(lock, [this] { return closed_ || !queue_.empty(); });
        }
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    // Non-blocking variant.
    bool popNowait(T& out) {
        std::lock_guard<std::mutex> lock(mu_);
        if (queue_.empty()) return false;
        out = std::move(queue_.front());
        queue_.pop_front();
        return true;
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.clear();
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            closed_ = true;
        }
        cv_.notify_all();
    }

    bool closed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return closed_;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mu_);
        return queue_.size();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::deque<T> queue_;
    bool closed_ = false;
};

}  // namespace jarvis
