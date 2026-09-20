import csv
import time
import serial
import numpy as np
from scipy.signal import butter, sosfilt


# ============================================================
# SETTINGS
# ============================================================

EEG_FILE = "/Users/haileylewicki/cheryls-womb/neurohack/.venv/eeg.project/example_eeg.txt"

# CHANGE THIS to the Arduino port that worked
ARDUINO_PORT = "/dev/cu.usbmodem101"

BAUD_RATE = 115200

SAMPLE_RATE = 250

ALPHA_LOW = 8
ALPHA_HIGH = 12

WINDOW_SECONDS = 1.0
UPDATE_SECONDS = 0.20

BASELINE_SECONDS = 10.0

ENTER_THRESHOLD = 0.10
EXIT_THRESHOLD = 0.05


# ============================================================
# CHANNEL MAPPING
# ============================================================

# Columns in the file:
#
# 0 = Sample Index
# 1 = EXG Channel 0
# 2 = EXG Channel 1
# ...
# 8 = EXG Channel 7

EXG_COLUMNS = list(range(1, 9))

LEFT_CHANNELS = [0, 2, 4]
RIGHT_CHANNELS = [1, 3, 5]


# ============================================================
# LOAD EEG FILE
# ============================================================

def load_eeg(filename):

    print(f"Loading EEG file: {filename}")

    rows = []

    with open(filename, "r") as f:

        # Skip OpenBCI metadata lines
        for line in f:

            if line.startswith("%"):
                continue

            # Skip header
            if line.startswith("Sample Index"):
                continue

            # Skip blank lines
            if not line.strip():
                continue

            try:
                values = next(csv.reader([line]))

                # We only need the 8 EXG channels
                eeg_values = [
                    float(values[i])
                    for i in EXG_COLUMNS
                ]

                rows.append(eeg_values)

            except (ValueError, IndexError):
                continue

    eeg = np.array(rows)

    print(f"Loaded {len(eeg)} EEG samples.")
    print(f"Duration: {len(eeg) / SAMPLE_RATE:.2f} seconds")
    print(f"Channels: {eeg.shape[1]}")

    return eeg


# ============================================================
# FILTER
# ============================================================

def create_alpha_filter():

    return butter(
        4,
        [ALPHA_LOW, ALPHA_HIGH],
        btype="bandpass",
        fs=SAMPLE_RATE,
        output="sos"
    )


# ============================================================
# ALPHA POWER
# ============================================================

def calculate_alpha_power(eeg_window, sos):

    # eeg_window shape:
    # channels × samples

    filtered = sosfilt(
        sos,
        eeg_window,
        axis=1
    )

    power = np.mean(
        filtered ** 2,
        axis=1
    )

    return power


# ============================================================
# DIRECTIONAL INDEX
# ============================================================

def calculate_directional_index(
    alpha_power,
    baseline_alpha
):

    eps = 1e-12

    # Log ratio relative to baseline
    relative_power = np.log(
        (alpha_power + eps) /
        (baseline_alpha + eps)
    )

    left_power = np.mean(
        relative_power[LEFT_CHANNELS]
    )

    right_power = np.mean(
        relative_power[RIGHT_CHANNELS]
    )

    directional_index = (
        left_power - right_power
    )

    return directional_index


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_direction(index, previous_state):

    if index > ENTER_THRESHOLD:
        return "LEFT"

    elif index < -ENTER_THRESHOLD:
        return "RIGHT"

    elif previous_state == "LEFT" and index > -EXIT_THRESHOLD:
        return "LEFT"

    elif previous_state == "RIGHT" and index < EXIT_THRESHOLD:
        return "RIGHT"

    else:
        return "NEUTRAL"


# ============================================================
# SEND TO ARDUINO
# ============================================================

def send_to_arduino(arduino, state):

    if state == "LEFT":
        arduino.write(b"L\n")

    elif state == "RIGHT":
        arduino.write(b"R\n")

    elif state == "NEUTRAL":
        arduino.write(b"N\n")


# ============================================================
# MAIN
# ============================================================

print()
print("======================================")
print(" EEG REPLAY MODE")
print("======================================")
print()

# Load recording
eeg = load_eeg(EEG_FILE)

# Create filter
sos = create_alpha_filter()

# Connect Arduino
print()
print("Connecting to Arduino...")

arduino = serial.Serial(
    ARDUINO_PORT,
    BAUD_RATE,
    timeout=1
)

time.sleep(2)

print("Arduino connected!")
print()

# ------------------------------------------------------------
# BASELINE
# ------------------------------------------------------------

baseline_samples = int(
    BASELINE_SECONDS * SAMPLE_RATE
)

window_samples = int(
    WINDOW_SECONDS * SAMPLE_RATE
)

update_samples = int(
    UPDATE_SECONDS * SAMPLE_RATE
)

if len(eeg) < baseline_samples + window_samples:

    raise ValueError(
        "EEG file is too short for the requested baseline."
    )


print(
    f"Using first {BASELINE_SECONDS} seconds "
    "as baseline..."
)

baseline_data = eeg[:baseline_samples].T

baseline_alpha = calculate_alpha_power(
    baseline_data,
    sos
)

print("Baseline established!")
print()

# ------------------------------------------------------------
# REPLAY
# ------------------------------------------------------------

previous_state = "NEUTRAL"

current_sample = baseline_samples

try:

    while current_sample + window_samples <= len(eeg):

        # Grab a 1-second window
        window = eeg[
            current_sample:
            current_sample + window_samples
        ].T

        # Calculate alpha power
        alpha_power = calculate_alpha_power(
            window,
            sos
        )

        # Calculate directional index
        directional_index = calculate_directional_index(
            alpha_power,
            baseline_alpha
        )

        # Convert to LEFT / RIGHT / NEUTRAL
        state = classify_direction(
            directional_index,
            previous_state
        )

        # Only send when state changes
        if state != previous_state:

            send_to_arduino(
                arduino,
                state
            )

            print(
                f"Time: "
                f"{current_sample / SAMPLE_RATE:6.2f}s"
                f" | Index: "
                f"{directional_index:+.3f}"
                f" | Signal: {state}"
            )

            previous_state = state

        # Move forward in the recording
        current_sample += update_samples

        # Pretend this is real-time
        time.sleep(UPDATE_SECONDS)


except KeyboardInterrupt:

    print()
    print("Replay stopped.")


finally:

    # Return LEDs to neutral
    try:
        arduino.write(b"N\n")
        time.sleep(0.2)
        arduino.close()
    except:
        pass

    print("Arduino disconnected.")
    print("Replay complete.")