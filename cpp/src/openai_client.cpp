#include "jarvis/openai_client.h"

#include <curl/curl.h>

#include <atomic>
#include <stdexcept>
#include <thread>

#include <nlohmann/json.hpp>

#include "jarvis/logger.h"
#include "jarvis/thread_queue.h"

namespace jarvis {

namespace {

void ensure_curl_global_init() {
    static const bool initialized = [] {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        return true;
    }();
    (void)initialized;
}

struct ResponseBuffer {
    std::string data;
};

size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* buf = static_cast<ResponseBuffer*>(userdata);
    buf->data.append(ptr, size * nmemb);
    return size * nmemb;
}

// POST a JSON body and return the raw response body.
std::string post_json(const std::string& url, const nlohmann::json& body,
                      const std::string& api_key, long timeout_ms) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("curl_easy_init fallita");
    }

    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers,
                                ("Authorization: Bearer " + api_key).c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");

    const std::string payload = body.dump();
    ResponseBuffer resp;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE,
                      static_cast<curl_off_t>(payload.size()));
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);

    const CURLcode rc = curl_easy_perform(curl);
    long http = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK || http >= 400) {
        throw std::runtime_error("Richiesta OpenAI fallita (HTTP " +
                                 std::to_string(http) + "): " + resp.data);
    }
    return resp.data;
}

// Streaming TTS: runs the curl transfer on a background thread and exposes the
// PCM chunks through a pull-based queue (mirrors requests' iter_content()).
class StreamingTtsSource : public TtsChunkSource {
public:
    StreamingTtsSource(std::string api_key, nlohmann::json body)
        : api_key_(std::move(api_key)), body_(std::move(body)) {
        thread_ = std::thread([this] { run_transfer(); });
    }

    ~StreamingTtsSource() override {
        queue_.close();
        if (thread_.joinable()) thread_.join();
    }

    bool nextChunk(std::vector<uint8_t>& out) override {
        return queue_.pop(out);
    }

private:
    void run_transfer() {
        CURL* curl = curl_easy_init();
        if (!curl) {
            queue_.close();
            return;
        }

        curl_slist* headers = nullptr;
        headers = curl_slist_append(
            headers, ("Authorization: Bearer " + api_key_).c_str());
        headers = curl_slist_append(headers, "Content-Type: application/json");

        const std::string payload = body_.dump();
        curl_easy_setopt(curl, CURLOPT_URL,
                         "https://api.openai.com/v1/audio/speech");
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE,
                          static_cast<curl_off_t>(payload.size()));
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, chunk_write_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, this);
        curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 4096L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 120000L);

        const CURLcode rc = curl_easy_perform(curl);
        long http = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);
        if (rc != CURLE_OK || http >= 400) {
            log_error("Errore durante lo streaming TTS (HTTP " +
                      std::to_string(http) + ")");
        }

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        queue_.close();
    }

    static size_t chunk_write_callback(char* ptr, size_t size, size_t nmemb,
                                       void* userdata) {
        auto* self = static_cast<StreamingTtsSource*>(userdata);
        const size_t total = size * nmemb;
        std::vector<uint8_t> chunk(ptr, ptr + total);
        self->queue_.push(std::move(chunk));
        return total;
    }

    std::string api_key_;
    nlohmann::json body_;
    std::thread thread_;
    ThreadQueue<std::vector<uint8_t>> queue_;
};

}  // namespace

OpenAIClient::OpenAIClient(std::string api_key) : api_key_(std::move(api_key)) {
    ensure_curl_global_init();
}

OpenAIClient::~OpenAIClient() = default;

std::string OpenAIClient::transcribe_audio(
    const std::vector<uint8_t>& wav_bytes) {
    ensure_curl_global_init();
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("curl_easy_init fallita");
    }

    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers,
                                ("Authorization: Bearer " + api_key_).c_str());

    curl_mime* form = curl_mime_init(curl);
    curl_mimepart* file = curl_mime_addpart(form);
    curl_mime_name(file, "file");
    curl_mime_filename(file, "command.wav");
    curl_mime_type(file, "audio/wav");
    curl_mime_data(file, reinterpret_cast<const char*>(wav_bytes.data()),
                   wav_bytes.size());

    auto add_field = [&](const char* name, const char* value) {
        curl_mimepart* part = curl_mime_addpart(form);
        curl_mime_name(part, name);
        curl_mime_data(part, value, CURL_ZERO_TERMINATED);
    };
    add_field("model", "whisper-1");
    add_field("language", "it");
    add_field("temperature", "0.0");

    ResponseBuffer resp;
    curl_easy_setopt(curl, CURLOPT_URL,
                     "https://api.openai.com/v1/audio/transcriptions");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &resp);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 20000L);

    const CURLcode rc = curl_easy_perform(curl);
    long http = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http);

    curl_mime_free(form);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (rc != CURLE_OK || http >= 400) {
        throw std::runtime_error("Trascrizione fallita (HTTP " +
                                 std::to_string(http) + "): " + resp.data);
    }

    auto json = nlohmann::json::parse(resp.data);
    std::string text = json.value("text", "");
    // trim
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
        text.erase(text.begin());
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
        text.pop_back();
    return text;
}

std::string OpenAIClient::generate_chat_response(
    const std::vector<ChatMessage>& history) {
    nlohmann::json body;
    body["model"] = "gpt-4o-mini";
    nlohmann::json messages = nlohmann::json::array();
    for (const auto& m : history) {
        messages.push_back({{"role", m.role}, {"content", m.content}});
    }
    body["messages"] = messages;

    const std::string raw =
        post_json("https://api.openai.com/v1/chat/completions", body, api_key_,
                  60000);
    const auto json = nlohmann::json::parse(raw);
    return json["choices"][0]["message"]["content"].get<std::string>();
}

std::unique_ptr<TtsChunkSource> OpenAIClient::stream_tts(const std::string& text) {
    ensure_curl_global_init();
    nlohmann::json body;
    body["model"] = "gpt-4o-mini-tts";
    body["voice"] = "onyx";
    body["input"] = text;
    body["response_format"] = "pcm";
    body["speed"] = 1.3;
    return std::make_unique<StreamingTtsSource>(api_key_, body);
}

}  // namespace jarvis
