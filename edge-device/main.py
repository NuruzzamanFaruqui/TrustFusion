# main.py
import time
from gpio_auth_interface import setup_gpio, wait_for_auth_event, trigger_access_grant, cleanup
from user_input_simulator import get_user_sample
from device_metadata import get_device_metadata
from api_client import send_to_backend

def main():
    setup_gpio()
    try:
        while True:
            if wait_for_auth_event():
                user_sample = get_user_sample()
                device_context = get_device_metadata()
                user_sample.update(device_context)

                trust_tier, decision = send_to_backend(user_sample)
                print(f"Trust Tier: {trust_tier}, Decision: {decision}")

                if decision == "grant":
                    trigger_access_grant()
                else:
                    print("Access denied.")

            time.sleep(1)

    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
