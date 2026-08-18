from main import Config

class HardwareController:
    def __init__(self, config: Config):
        self.config = config
        self.available = False
        try:
            import RPi.GPIO as GPIO
            self.GPIO = GPIO
            self.available = True
        except (ImportError, RuntimeError):
            pass

    def setup(self):
        if not self.available:
            return
        self.GPIO.setmode(self.GPIO.BCM)
        self.GPIO.setup(self.config.LED_THINK, self.GPIO.OUT, initial=self.GPIO.LOW)
        self.GPIO.setup(self.config.LED_LISTEN, self.GPIO.OUT, initial=self.GPIO.LOW)

    def set_think_led(self, state: bool):
        if not self.available: return
        self.GPIO.output(self.config.LED_THINK, self.GPIO.HIGH if state else self.GPIO.LOW)

    def set_listen_led(self, state: bool):
        if not self.available: return
        self.GPIO.output(self.config.LED_LISTEN, self.GPIO.HIGH if state else self.GPIO.LOW)

    def turn_off_all(self):
        if not self.available: return
        self.GPIO.output(self.config.LED_THINK, self.GPIO.LOW)
        self.GPIO.output(self.config.LED_LISTEN, self.GPIO.LOW)

    def cleanup(self):
        if not self.available: return
        self.turn_off_all()
        self.GPIO.cleanup()