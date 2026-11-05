#include <csignal>
#include <string>

#include "jarvis/assistant_orchestrator.h"
#include "jarvis/logger.h"
#include "jarvis/signals.h"

namespace {

void handle_signal(int) { jarvis::request_stop(); }

}  // namespace

int main() {
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    try {
        jarvis::AssistantOrchestrator app;
        app.run();
    } catch (const std::exception& e) {
        jarvis::log_error("Errore fatale: " + std::string(e.what()));
        return 1;
    }
    return 0;
}
