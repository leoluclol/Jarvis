#include "jarvis/wav.h"

#include <cstring>
#include <string>

namespace jarvis {

namespace {

void write_le32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
}

void write_le16(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>(v & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
}

}  // namespace

std::vector<uint8_t> build_wav(const std::vector<uint8_t>& pcm_bytes,
                               int sample_rate, int channels, int sample_width) {
    std::vector<uint8_t> out;
    const uint32_t data_size = static_cast<uint32_t>(pcm_bytes.size());
    const uint32_t byte_rate =
        static_cast<uint32_t>(sample_rate * channels * sample_width);

    // RIFF header
    const std::string riff = "RIFF";
    out.insert(out.end(), riff.begin(), riff.end());
    write_le32(out, 36 + data_size);
    const std::string wave = "WAVE";
    out.insert(out.end(), wave.begin(), wave.end());

    // fmt chunk
    const std::string fmt = "fmt ";
    out.insert(out.end(), fmt.begin(), fmt.end());
    write_le32(out, 16);                    // fmt chunk size
    write_le16(out, 1);                     // PCM
    write_le16(out, static_cast<uint16_t>(channels));
    write_le32(out, static_cast<uint32_t>(sample_rate));
    write_le32(out, byte_rate);
    write_le16(out, static_cast<uint16_t>(channels * sample_width));  // block align
    write_le16(out, static_cast<uint16_t>(sample_width * 8));         // bits/sample

    // data chunk
    const std::string data = "data";
    out.insert(out.end(), data.begin(), data.end());
    write_le32(out, data_size);
    out.insert(out.end(), pcm_bytes.begin(), pcm_bytes.end());
    return out;
}

std::vector<uint8_t> build_wav(const std::vector<int16_t>& pcm,
                               int sample_rate, int channels) {
    std::vector<uint8_t> bytes;
    bytes.reserve(pcm.size() * 2);
    for (int16_t s : pcm) {
        bytes.push_back(static_cast<uint8_t>(s & 0xFF));
        bytes.push_back(static_cast<uint8_t>((s >> 8) & 0xFF));
    }
    return build_wav(bytes, sample_rate, channels, 2);
}

}  // namespace jarvis
