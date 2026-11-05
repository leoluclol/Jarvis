#pragma once

#include "jarvis/config.h"

namespace jarvis {

// Raspberry Pi LED control via the legacy sysfs GPIO interface (BCM numbers).
// On machines without sysfs GPIO (e.g. the dev box) every call is a safe
// no-op, exactly like the Python RPi.GPIO try/except fallback.
class HardwareController {
public:
    explicit HardwareController(const Config& config);
    ~HardwareController();

    // Attempts to export the LED pins and mark the controller available.
    void setup();
    void set_think_led(bool on);
    void set_listen_led(bool on);
    void turn_off_all();
    void cleanup();

    bool available() const { return available_; }

private:
    bool write_gpio(const std::string& path, const std::string& value) const;
    bool export_pin(int pin);
    bool set_pin_direction(int pin, const std::string& dir);
    bool set_pin_value(int pin, bool high);
    void unexport_pin(int pin);

    Config config_;
    bool available_ = false;
};

}  // namespace jarvis
