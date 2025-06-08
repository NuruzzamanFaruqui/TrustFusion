# gpio_auth_interface.py
import RPi.GPIO as GPIO
import time

# Pin configuration (update based on your wiring)
FINGERPRINT_SENSOR_PIN = 17  # Input pin for fingerprint sensor trigger
ACCESS_RELAY_PIN = 27        # Output pin for door unlock mechanism

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(FINGERPRINT_SENSOR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(ACCESS_RELAY_PIN, GPIO.OUT)
    GPIO.output(ACCESS_RELAY_PIN, GPIO.LOW)

def wait_for_auth_event():
    print("Waiting for fingerprint scan...")
    while True:
        if GPIO.input(FINGERPRINT_SENSOR_PIN) == GPIO.HIGH:
            print("Fingerprint scanned.")
            return True
        time.sleep(0.1)

def trigger_access_grant(duration=3):
    print("Access granted. Unlocking...")
    GPIO.output(ACCESS_RELAY_PIN, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(ACCESS_RELAY_PIN, GPIO.LOW)
    print("Access locked.")

def cleanup():
    GPIO.cleanup()
