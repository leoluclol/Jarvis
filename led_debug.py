#!/usr/bin/env python3
"""Debug script to verify Raspberry Pi LEDs on BCM pins 21 and 23."""

import time

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError) as exc:
    raise SystemExit(
        "RPi.GPIO is not available. Run this on a Raspberry Pi with Python3 and the RPi.GPIO package installed. ``pip install RPi.GPIO`` if needed.\n"
        f"Details: {exc}"
    )

LED_THINK = 21
LED_LISTEN = 23


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_THINK, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LED_LISTEN, GPIO.OUT, initial=GPIO.LOW)


def cleanup_gpio():
    GPIO.output(LED_THINK, GPIO.LOW)
    GPIO.output(LED_LISTEN, GPIO.LOW)
    GPIO.cleanup()


def blink(pin: int, label: str, count: int = 3, interval: float = 0.5):
    print(f"Blinking {label} LED on GPIO {pin} {count} times...")
    for _ in range(count):
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(interval)
        GPIO.output(pin, GPIO.LOW)
        time.sleep(interval)


def main():
    print("Starting LED debug script for GPIO 21 and 23.")
    setup_gpio()

    try:
        blink(LED_THINK, "THINK")
        blink(LED_LISTEN, "LISTEN")

        print("Turning both LEDs on for 2 seconds...")
        GPIO.output(LED_THINK, GPIO.HIGH)
        GPIO.output(LED_LISTEN, GPIO.HIGH)
        time.sleep(2)
        print("Turning both LEDs off.")
        GPIO.output(LED_THINK, GPIO.LOW)
        GPIO.output(LED_LISTEN, GPIO.LOW)

        print("LED test completed successfully.")
    finally:
        cleanup_gpio()


if __name__ == "__main__":
    main()
