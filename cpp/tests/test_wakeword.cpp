#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>

#include "jarvis/wakeword_engine.h"

#ifndef JARVIS_CPP_DIR
#define JARVIS_CPP_DIR "."
#endif

namespace jarvis {

struct RefData {
    std::vector<int16_t> pcm;
    std::vector<std::vector<float>> features;
    std::vector<float> probs;
};

bool load_ref(const std::string& path, RefData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    uint32_t n = 0;
    auto read = [&](void* dst, size_t bytes) {
        return f.read(static_cast<char*>(dst),
                      static_cast<std::streamsize>(bytes)).gcount() ==
               static_cast<std::streamsize>(bytes);
    };

    if (!read(&n, 4)) return false;
    out.pcm.resize(n);
    if (!read(out.pcm.data(), n * 2)) return false;

    if (!read(&n, 4)) return false;
    out.features.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        out.features[i].resize(40);
        if (!read(out.features[i].data(), 40 * 4)) return false;
    }

    if (!read(&n, 4)) return false;
    out.probs.resize(n);
    return read(out.probs.data(), n * 4);
}

// Friend of WakeWordEngine: can drive the private per-frame probability path.
class WakeWordTest {
public:
    static int run() {
        RefData ref;
        const std::string ref_path =
            std::string(JARVIS_CPP_DIR) + "/tests/data/ref_wakeword.bin";
        if (!load_ref(ref_path, ref)) {
            std::cerr << "Cannot load reference: " << ref_path << std::endl;
            return 1;
        }

        // ---- 1) Frontend features must match pymicro_features exactly. ----
        MicroFrontend frontend;
        std::vector<std::vector<float>> feats;
        for (size_t i = 0; i < ref.pcm.size(); i += 160) {
            const size_t n = std::min<size_t>(160, ref.pcm.size() - i);
            frontend.process_samples(ref.pcm.data() + i, n, feats);
        }
        if (feats.size() != ref.features.size()) {
            std::cerr << "Feature frame count mismatch: got " << feats.size()
                      << ", expected " << ref.features.size() << std::endl;
            return 1;
        }
        double max_feat_err = 0.0;
        for (size_t i = 0; i < feats.size(); ++i) {
            for (size_t j = 0; j < feats[i].size(); ++j) {
                max_feat_err = std::max(
                    max_feat_err,
                    std::fabs(static_cast<double>(feats[i][j]) -
                              ref.features[i][j]));
            }
        }
        std::cout << "frontend: " << feats.size()
                  << " frames, max abs err = " << max_feat_err << std::endl;
        if (max_feat_err > 0.01) {
            std::cerr << "FEATURE MISMATCH" << std::endl;
            return 1;
        }

        // ---- 2) Wake-word probabilities must match the TFLite path. ----
        WakeWordEngine ww(std::string(JARVIS_CPP_DIR) +
                          "/models/hey_jarvis.json");
        std::vector<float> my_probs;
        for (size_t i = 0; i < ref.pcm.size(); i += 160) {
            const size_t n = std::min<size_t>(160, ref.pcm.size() - i);
            std::vector<std::vector<float>> frames;
            ww.frontend_->process_samples(ref.pcm.data() + i, n, frames);
            for (const auto& f : frames) {
                my_probs.push_back(ww.process_streaming_prob(f));
            }
        }
        if (my_probs.size() != ref.probs.size()) {
            std::cerr << "Probability count mismatch: got " << my_probs.size()
                      << ", expected " << ref.probs.size() << std::endl;
            return 1;
        }
        double max_prob_err = 0.0;
        for (size_t i = 0; i < my_probs.size(); ++i) {
            max_prob_err = std::max(
                max_prob_err,
                std::fabs(static_cast<double>(my_probs[i]) - ref.probs[i]));
        }
        std::cout << "wakeword: " << my_probs.size()
                  << " probs, max abs err = " << max_prob_err << std::endl;
        if (max_prob_err > 0.01) {
            std::cerr << "PROBABILITY MISMATCH" << std::endl;
            return 1;
        }

        std::cout << "test_wakeword: OK" << std::endl;
        return 0;
    }
};

}  // namespace jarvis

int main() { return jarvis::WakeWordTest::run(); }
