class LedManager:
    self.gpio_available = False
    
    
    def __init__(gpio_available_value, led_think_value, led_listen_value):
        self.gpio_available = gpio_available_value
        self.led_think = led_think_value
        self.led_listen = led_listen_value

    def init_gpio():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_THINK, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LED_LISTEN, GPIO.OUT, initial=GPIO.LOW)


def set_led(pin: int, state: bool):
    if not GPIO_AVAILABLE:
        return
    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


def all_leds_off():
    if not GPIO_AVAILABLE:
        return
    GPIO.output(LED_THINK, GPIO.LOW)
    GPIO.output(LED_LISTEN, GPIO.LOW)


def cleanup_gpio():
    if not GPIO_AVAILABLE:
        return
    all_leds_off()
    GPIO.cleanup()