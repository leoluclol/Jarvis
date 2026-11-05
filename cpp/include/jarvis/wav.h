#pragma once

#include <cstdint>
#include <vector>

namespace jarvis {

// Serializes raw little-endian PCM (int16 mono) into a standard RIFF/WAVE
// file in memory, mirroring Python's `wave` module usage.
std::vector<uint8_t> build_wav(
    const std::vector<uint8_t>& pcm_bytes,
    int sample_rate,
    int channels = 1,
    int sample_width = 2);

// Convenience overload taking int16 samples directly.
std::vector<uint8_t> build_wav(
    const std::vector<int16_t>& pcm,
    int sample_rate,
    int channels = 1);

}  // namespace jarvis
