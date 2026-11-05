#include "jarvis/hardware_controller.h"

#include <fstream>

#include "jarvis/logger.h"

namespace jarvis {

namespace {
const char* kGpioBase = "/sys/class/gpio";
}  // namespace

HardwareController::HardwareController(const Config& config) : config_(config) {}

HardwareController::~HardwareController() { cleanup(); }

bool HardwareController::write_gpio(const std::string& path,
                                    const std::string& value) const {
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << value;
    return static_cast<bool>(out);
}

bool HardwareController::export_pin(int pin) {
    return write_gpio(std::string(kGpioBase) + "/export", std::to_string(pin));
}

bool HardwareController::set_pin_direction(int pin, const std::string& dir) {
    return write_gpio(std::string(kGpioBase) + "/gpio" + std::to_string(pin) +
                          "/direction",
                      dir);
}

bool HardwareController::set_pin_value(int pin, bool high) {
    return write_gpio(std::string(kGpioBase) + "/gpio" + std::to_string(pin) +
                          "/value",
                      high ? "1" : "0");
}

void HardwareController::unexport_pin(int pin) {
    write_gpio(std::string(kGpioBase) + "/unexport", std::to_string(pin));
}

void HardwareController::setup() {
    // Export is only possible with root/GPIO group privileges; on machines
    // without sysfs GPIO this fails and the controller stays inert.
    if (export_pin(config_.led_think) &&
        export_pin(config_.led_listen)) {
        set_pin_direction(config_.led_think, "out");
        set_pin_direction(config_.led_listen, "out");
        available_ = true;
        turn_off_all();
    } else {
        available_ = false;
        log_info("⚠️ GPIO non disponibile: indicatori LED disattivati.");
    }
}

void HardwareController::set_think_led(bool on) {
    if (!available_) return;
    set_pin_value(config_.led_think, on);
}

void HardwareController::set_listen_led(bool on) {
    if (!available_) return;
    set_pin_value(config_.led_listen, on);
}

void HardwareController::turn_off_all() {
    if (!available_) return;
    set_pin_value(config_.led_think, false);
    set_pin_value(config_.led_listen, false);
}

void HardwareController::cleanup() {
    if (!available_) return;
    turn_off_all();
    unexport_pin(config_.led_think);
    unexport_pin(config_.led_listen);
    available_ = false;
}

}  // namespace jarvis
