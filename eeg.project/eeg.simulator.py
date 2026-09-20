import serial
import time
import math
import random

# ============================================================
# SETTINGS
# ============================================================

# IMPORTANT:
# Change this to the Arduino port that worked for you.
ARDUINO_PORT = "/dev/cu.usbmodem101"

BAUD_RATE = 115200

# Same thresholds used in eeg_live.py
ENTER_THRESHOLD = 0.10
EXIT_THRESHOLD = 0.05

# How often the simulated EEG updates
UPDATE_INTERVAL = 0.20


# ============================================================
# CONNECT TO ARDUINO
# ============================================================

print("Connecting to Arduino...")

arduino = serial.Serial(
    ARDUINO_PORT,
    BAUD_RATE,
    timeout=1
)

# Arduino resets when serial connection opens
time.sleep(2)

print("Arduino connected!")
print()
print("Starting simulated EEG...")
print("Press Ctrl+C to stop.")
print()


# ============================================================
# SIMULATED EEG SIGNAL
# ============================================================

def generate_fake_directional_index(t):
    """
    Simulates the directional index that would normally
    come from the real EEG.

    Positive = LEFT
    Negative = RIGHT
    Near zero = NEUTRAL
    """

    # Slowly oscillating neural signal
    signal = 0.16 * math.sin(t * 0.7)

    # Add small random fluctuations
    noise = random.uniform(-0.025, 0.025)

    return signal + noise


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_direction(index, previous_state):

    # Strong LEFT signal
    if index > ENTER_THRESHOLD:
        return "LEFT"

    # Strong RIGHT signal
    elif index < -ENTER_THRESHOLD:
        return "RIGHT"

    # If we're already in LEFT/RIGHT, don't immediately
    # switch because of tiny fluctuations.
    elif previous_state == "LEFT" and index > -EXIT_THRESHOLD:
        return "LEFT"

    elif previous_state == "RIGHT" and index < EXIT_THRESHOLD:
        return "RIGHT"

    # Otherwise neutral
    else:
        return "NEUTRAL"


# ============================================================
# SEND COMMAND TO ARDUINO
# ============================================================

def send_command(state):

    if state == "LEFT":
        arduino.write(b"L\n")

    elif state == "RIGHT":
        arduino.write(b"R\n")

    elif state == "NEUTRAL":
        arduino.write(b"N\n")


# ============================================================
# MAIN SIMULATION
# ============================================================

previous_state = "NEUTRAL"

start_time = time.time()

try:

    while True:

        # Current simulation time
        t = time.time() - start_time

        # Generate fake EEG-derived index
        directional_index = generate_fake_directional_index(t)

        # Convert index into LEFT / RIGHT / NEUTRAL
        state = classify_direction(
            directional_index,
            previous_state
        )

        # Only send a new command when the state changes
        if state != previous_state:

            send_command(state)

            print(
                f"Directional Index: {directional_index:+.3f}"
                f"   →   {state}"
            )

            previous_state = state

        # Keep updating
        time.sleep(UPDATE_INTERVAL)


except KeyboardInterrupt:

    print()
    print("Stopping simulation...")

    # Return LEDs to neutral
    arduino.write(b"N\n")

    time.sleep(0.5)

    arduino.close()

    print("Arduino disconnected.")
    print("Simulation complete!")