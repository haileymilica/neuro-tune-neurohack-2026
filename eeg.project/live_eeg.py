"""
LIVE EEG → F3/F4 ALPHA ASYMMETRY → ARDUINO LED

FAST / LOW-LATENCY VERSION

Pipeline:

OpenBCI EEG
    ↓
F3 / F4
    ↓
8–12 Hz alpha band
    ↓
stateful live filter
    ↓
0.5-second rolling window
    ↓
alpha power
    ↓
rolling baseline normalization
    ↓
F3 vs F4 asymmetry
    ↓
light smoothing
    ↓
LEFT / RIGHT / NEUTRAL
    ↓
Arduino LED


BrainFlow EEG channels:

0 = Fp1
1 = Fp2
2 = F3
3 = F4
4 = T7
5 = T8
6 = Pz
7 = Cz


Arduino:

D8  = LEFT
D9  = RIGHT
D10 = NEUTRAL

Commands:

L = LEFT
R = RIGHT
N = NEUTRAL
"""


import time
import numpy as np
import serial

from scipy.signal import butter, sosfilt, sosfilt_zi

from brainflow.board_shim import (
    BoardShim,
    BrainFlowInputParams,
    BoardIds
)


# ============================================================
# CONNECTION SETTINGS
# ============================================================

BOARD_ID = BoardIds.CYTON_BOARD.value

OPENBCI_PORT = "/dev/cu.usbserial-DQ00859S"

ARDUINO_PORT = "/dev/cu.usbmodem101"

ARDUINO_BAUD = 115200


# ============================================================
# EEG SETTINGS
# ============================================================

LOW_FREQ = 8.0
HIGH_FREQ = 12.0


# ============================================================
# SPEED SETTINGS
# ============================================================

# How much EEG is used for each decision.
#
# Smaller = faster but noisier.
#
# 0.5 seconds is a good compromise for the demo.

WINDOW_SECONDS = 0.50


# How often the classifier updates.
#
# This can be faster than the window length because
# the windows overlap.

UPDATE_SECONDS = 0.10


# Number of recent directional-index values to average.
#
# Smaller = faster response.
# Larger = smoother but more lag.

SMOOTHING_POINTS = 3


# ============================================================
# INITIAL BASELINE
# ============================================================

INITIAL_BASELINE_SECONDS = 10.0


# ============================================================
# ROLLING BASELINE
# ============================================================

# The baseline adapts slowly so normal EEG drift does not
# permanently push the classifier toward LEFT or RIGHT.
#
# Smaller = more stable.
# Larger = adapts faster.

BASELINE_ADAPT_RATE = 0.01


# ============================================================
# CLASSIFICATION THRESHOLDS
# ============================================================

# Positive index = LEFT
# Negative index = RIGHT

ENTER_THRESHOLD = 0.015

EXIT_THRESHOLD = 0.007


# ============================================================
# PERSISTENCE
# ============================================================

# Number of consecutive classifier updates required
# before changing the stable state.

PERSISTENCE_UPDATES = 1


# ============================================================
# NUMERICAL SAFETY
# ============================================================

EPSILON = 1e-12


# ============================================================
# CHANNELS
# ============================================================

CHANNEL_NAMES = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "T7",
    "T8",
    "Pz",
    "Cz"
]


# F3 = BrainFlow EEG channel index 2
# F4 = BrainFlow EEG channel index 3

LEFT_CHANNEL = 2
RIGHT_CHANNEL = 3


# ============================================================
# GLOBAL CLASSIFIER STATE
# ============================================================

current_state = "NEUTRAL"

candidate_state = "NEUTRAL"

candidate_count = 0


# ============================================================
# CREATE ALPHA FILTER
# ============================================================

def create_alpha_filter(sample_rate):

    """
    Create a 4th-order Butterworth band-pass filter
    for the alpha band: 8–12 Hz.
    """

    sos = butter(
        4,
        [LOW_FREQ, HIGH_FREQ],
        btype="bandpass",
        fs=sample_rate,
        output="sos"
    )

    return sos


# ============================================================
# INITIALIZE FILTER STATE
# ============================================================

def initialize_filter_state(
    sos,
    eeg_data
):

    """
    Initialize the IIR filter state using calibration data.

    This prevents a large filter transient when live
    processing begins.
    """

    zi = np.zeros(
        (
            sos.shape[0],
            eeg_data.shape[0],
            2
        )
    )


    for channel in range(
        eeg_data.shape[0]
    ):

        zi[:, channel, :] = (
            sosfilt_zi(sos)
            *
            eeg_data[channel, -1]
        )


    return zi


# ============================================================
# STATEFUL FILTER
# ============================================================

def filter_live_data(
    eeg_chunk,
    sos,
    filter_state
):

    """
    Filter new EEG data while preserving the filter state.

    The filter state is carried from one update to the next.
    """

    filtered = np.zeros_like(
        eeg_chunk
    )

    new_state = np.zeros_like(
        filter_state
    )


    for channel in range(
        eeg_chunk.shape[0]
    ):

        filtered[channel], new_state[:, channel, :] = (
            sosfilt(
                sos,
                eeg_chunk[channel],
                zi=filter_state[:, channel, :]
            )
        )


    return filtered, new_state


# ============================================================
# ALPHA POWER
# ============================================================

def calculate_alpha_power(
    filtered_window
):

    """
    Calculate mean squared amplitude
    of the filtered alpha signal.
    """

    return np.mean(
        filtered_window ** 2,
        axis=1
    )


# ============================================================
# INITIAL BASELINE
# ============================================================

def calculate_initial_baseline(
    calibration_data,
    sos
):

    """
    Calculate resting alpha-power baseline
    for F3 and F4.
    """

    filtered = np.zeros_like(
        calibration_data
    )


    for channel in range(
        calibration_data.shape[0]
    ):

        filtered[channel] = sosfilt(
            sos,
            calibration_data[channel]
        )


    alpha_power = calculate_alpha_power(
        filtered
    )


    left_baseline = alpha_power[
        LEFT_CHANNEL
    ]


    right_baseline = alpha_power[
        RIGHT_CHANNEL
    ]


    return (
        left_baseline,
        right_baseline
    )


# ============================================================
# DIRECTIONAL INDEX
# ============================================================

def calculate_directional_index(
    alpha_power,
    left_baseline,
    right_baseline
):

    """
    Normalize F3 and F4 independently against their
    respective baselines.

    Then calculate:

        DI = (F3 - F4) / (F3 + F4)

    Positive = F3-dominant
    Negative = F4-dominant
    """

    left_normalized = (
        alpha_power[LEFT_CHANNEL]
        /
        (
            left_baseline
            +
            EPSILON
        )
    )


    right_normalized = (
        alpha_power[RIGHT_CHANNEL]
        /
        (
            right_baseline
            +
            EPSILON
        )
    )


    denominator = (
        left_normalized
        +
        right_normalized
        +
        EPSILON
    )


    directional_index = (
        left_normalized
        -
        right_normalized
    ) / denominator


    return (
        directional_index,
        left_normalized,
        right_normalized
    )


# ============================================================
# CLASSIFIER
# ============================================================

def classify_direction(
    index,
    current_state
):

    """
    Convert the directional index into:

        LEFT
        RIGHT
        NEUTRAL

    Hysteresis prevents rapid flickering around zero.
    """


    # --------------------------------------------------------
    # CURRENTLY LEFT
    # --------------------------------------------------------

    if current_state == "LEFT":

        if index < EXIT_THRESHOLD:

            if index < -ENTER_THRESHOLD:

                return "RIGHT"

            return "NEUTRAL"

        return "LEFT"


    # --------------------------------------------------------
    # CURRENTLY RIGHT
    # --------------------------------------------------------

    if current_state == "RIGHT":

        if index > -EXIT_THRESHOLD:

            if index > ENTER_THRESHOLD:

                return "LEFT"

            return "NEUTRAL"

        return "RIGHT"


    # --------------------------------------------------------
    # CURRENTLY NEUTRAL
    # --------------------------------------------------------

    if index > ENTER_THRESHOLD:

        return "LEFT"


    if index < -ENTER_THRESHOLD:

        return "RIGHT"


    return "NEUTRAL"


# ============================================================
# UPDATE STABLE STATE
# ============================================================

def update_state(
    new_candidate
):

    """
    Apply persistence before changing the stable state.
    """

    global current_state
    global candidate_state
    global candidate_count


    # Already in requested state.

    if new_candidate == current_state:

        candidate_state = new_candidate

        candidate_count = 0

        return current_state


    # Same candidate as previous update.

    if new_candidate == candidate_state:

        candidate_count += 1

    else:

        candidate_state = new_candidate

        candidate_count = 1


    # Candidate has persisted long enough.

    if candidate_count >= PERSISTENCE_UPDATES:

        current_state = candidate_state

        candidate_count = 0

        print()

        print(
            ">>> STATE CHANGED:",
            current_state
        )

        print()


    return current_state


# ============================================================
# SEND COMMAND TO ARDUINO
# ============================================================

def send_to_arduino(
    arduino,
    state
):

    """
    Send the appropriate command to Arduino.
    """

    if state == "LEFT":

        arduino.write(
            b"L\n"
        )


    elif state == "RIGHT":

        arduino.write(
            b"R\n"
        )


    else:

        arduino.write(
            b"N\n"
        )


# ============================================================
# PRINT SERIAL PORTS
# ============================================================

def print_available_ports():

    import serial.tools.list_ports


    print()

    print(
        "AVAILABLE SERIAL PORTS"
    )

    print(
        "-----------------------"
    )


    ports = list(
        serial.tools.list_ports.comports()
    )


    if not ports:

        print(
            "No serial ports found."
        )

    else:

        for port in ports:

            print(
                f"{port.device} | "
                f"{port.description}"
            )


    print()


# ============================================================
# MAIN
# ============================================================

def main():

    global current_state


    print()

    print(
        "=" * 65
    )

    print(
        "FAST LIVE EEG COMMUNICATION PROTOTYPE"
    )

    print(
        "=" * 65
    )

    print()


    print(
        "NEURAL INPUT"
    )

    print(
        "------------"
    )

    print(
        "LEFT  = F3"
    )

    print(
        "RIGHT = F4"
    )

    print()


    print(
        "SIGNAL"
    )

    print(
        "------"
    )

    print(
        f"Alpha band:       "
        f"{LOW_FREQ:.0f}–{HIGH_FREQ:.0f} Hz"
    )

    print(
        f"Window:           "
        f"{WINDOW_SECONDS:.2f} sec"
    )

    print(
        f"Update interval:  "
        f"{UPDATE_SECONDS:.2f} sec"
    )

    print(
        f"Smoothing:        "
        f"{SMOOTHING_POINTS} points"
    )

    print()


    print_available_ports()


    # ========================================================
    # CONNECT ARDUINO
    # ========================================================

    print(
        "Connecting to Arduino..."
    )


    try:

        arduino = serial.Serial(
            ARDUINO_PORT,
            ARDUINO_BAUD,
            timeout=1
        )


        # Give Arduino time to reset.

        time.sleep(2)


        # Start neutral.

        send_to_arduino(
            arduino,
            "NEUTRAL"
        )


        print(
            "Arduino connected."
        )


    except Exception as e:

        print()

        print(
            "ERROR CONNECTING TO ARDUINO:"
        )

        print(e)

        return


    # ========================================================
    # CONNECT OPENBCI
    # ========================================================

    print()

    print(
        "Connecting to OpenBCI..."
    )


    params = BrainFlowInputParams()

    params.serial_port = OPENBCI_PORT


    board = BoardShim(
        BOARD_ID,
        params
    )


    try:

        board.prepare_session()

        board.start_stream()


        print(
            "OpenBCI connected."
        )

        print()


        # ====================================================
        # SAMPLE RATE
        # ====================================================

        sample_rate = (
            BoardShim.get_sampling_rate(
                BOARD_ID
            )
        )


        print(
            f"Sampling rate: "
            f"{sample_rate} Hz"
        )

        print()


        # ====================================================
        # EEG CHANNELS
        # ====================================================

        eeg_channels = (
            BoardShim.get_eeg_channels(
                BOARD_ID
            )
        )


        print(
            "EEG channels:",
            eeg_channels
        )

        print()


        # ====================================================
        # FILTER
        # ====================================================

        sos = create_alpha_filter(
            sample_rate
        )


        # ====================================================
        # INITIAL CALIBRATION
        # ====================================================

        print(
            "=" * 65
        )

        print(
            "INITIAL RESTING BASELINE"
        )

        print(
            "=" * 65
        )

        print()

        print(
            f"Collecting "
            f"{INITIAL_BASELINE_SECONDS:.0f} "
            f"seconds of resting EEG..."
        )

        print()

        print(
            "Sit comfortably."
        )

        print(
            "Keep your head still."
        )

        print(
            "Do not intentionally produce LEFT or RIGHT."
        )

        print()


        calibration_chunks = []


        calibration_start = time.time()


        while (
            time.time()
            -
            calibration_start
            <
            INITIAL_BASELINE_SECONDS
        ):

            data = (
                board.get_current_board_data(
                    int(sample_rate * 0.5)
                )
            )


            if data.shape[1] > 0:

                calibration_chunks.append(
                    data[
                        eeg_channels,
                        :
                    ]
                )


            time.sleep(0.05)


        if not calibration_chunks:

            raise RuntimeError(
                "No EEG data received during calibration."
            )


        calibration_data = (
            np.concatenate(
                calibration_chunks,
                axis=1
            )
        )


        required_samples = int(
            INITIAL_BASELINE_SECONDS
            *
            sample_rate
        )


        calibration_data = (
            calibration_data[
                :,
                -required_samples:
            ]
        )


        print(
            f"Calibration samples: "
            f"{calibration_data.shape[1]}"
        )


        # ====================================================
        # BASELINE
        # ====================================================

        (
            left_baseline,
            right_baseline
        ) = calculate_initial_baseline(
            calibration_data,
            sos
        )


        print()

        print(
            "INITIAL BASELINE"
        )

        print(
            "----------------"
        )

        print(
            f"F3 baseline: "
            f"{left_baseline:.6f}"
        )

        print(
            f"F4 baseline: "
            f"{right_baseline:.6f}"
        )


        baseline_ratio = (
            left_baseline
            /
            (
                right_baseline
                +
                EPSILON
            )
        )


        print(
            f"F3/F4 ratio: "
            f"{baseline_ratio:.3f}"
        )


        # ====================================================
        # FILTER STATE
        # ====================================================

        filter_state = (
            initialize_filter_state(
                sos,
                calibration_data
            )
        )


        # ====================================================
        # LIVE WINDOW
        # ====================================================

        window_samples = int(
            WINDOW_SECONDS
            *
            sample_rate
        )


        # ====================================================
        # DIRECTIONAL HISTORY
        # ====================================================

        directional_history = []


        # ====================================================
        # TIMER
        # ========================================================

        last_update = 0


        # ====================================================
        # LIVE MODE
        # ====================================================

        print()

        print(
            "=" * 65
        )

        print(
            "LIVE MODE"
        )

        print(
            "=" * 65
        )

        print()

        print(
            "Positive index → LEFT"
        )

        print(
            "Negative index → RIGHT"
        )

        print()

        print(
            "Press CTRL+C to stop."
        )

        print()


        # ====================================================
        # MAIN LIVE LOOP
        # ====================================================

        while True:

            now = time.time()


            # -----------------------------------------------
            # UPDATE TIMER
            # -----------------------------------------------

            if (
                now - last_update
                <
                UPDATE_SECONDS
            ):

                time.sleep(0.005)

                continue


            last_update = now


            # -----------------------------------------------
            # GET ROLLING EEG WINDOW
            # -----------------------------------------------

            data = (
                board.get_current_board_data(
                    window_samples
                )
            )


            if data.shape[1] < window_samples:

                time.sleep(0.02)

                continue


            eeg_window = (
                data[
                    eeg_channels,
                    :
                ]
            )


            # -----------------------------------------------
            # FILTER
            # -----------------------------------------------

            filtered_window, filter_state = (
                filter_live_data(
                    eeg_window,
                    sos,
                    filter_state
                )
            )


            # -----------------------------------------------
            # ALPHA POWER
            # -----------------------------------------------

            alpha_power = (
                calculate_alpha_power(
                    filtered_window
                )
            )


            current_left_power = (
                alpha_power[
                    LEFT_CHANNEL
                ]
            )


            current_right_power = (
                alpha_power[
                    RIGHT_CHANNEL
                ]
            )


            # -----------------------------------------------
            # ROLLING BASELINE
            # -----------------------------------------------

            left_baseline = (
                (1 - BASELINE_ADAPT_RATE)
                *
                left_baseline
                +
                BASELINE_ADAPT_RATE
                *
                current_left_power
            )


            right_baseline = (
                (1 - BASELINE_ADAPT_RATE)
                *
                right_baseline
                +
                BASELINE_ADAPT_RATE
                *
                current_right_power
            )


            # -----------------------------------------------
            # DIRECTIONAL INDEX
            # -----------------------------------------------

            (
                raw_index,
                left_score,
                right_score
            ) = calculate_directional_index(
                alpha_power,
                left_baseline,
                right_baseline
            )


            # -----------------------------------------------
            # SMOOTHING
            # -----------------------------------------------

            directional_history.append(
                raw_index
            )


            if len(
                directional_history
            ) > SMOOTHING_POINTS:

                directional_history.pop(0)


            smoothed_index = np.mean(
                directional_history
            )


            # -----------------------------------------------
            # CLASSIFY
            # -----------------------------------------------

            requested_state = (
                classify_direction(
                    smoothed_index,
                    current_state
                )
            )


            # -----------------------------------------------
            # UPDATE STATE
            # -----------------------------------------------

            stable_state = update_state(
                requested_state
            )


            # -----------------------------------------------
            # SEND TO ARDUINO
            # -----------------------------------------------

            send_to_arduino(
                arduino,
                stable_state
            )


            # -----------------------------------------------
            # DISPLAY
            # -----------------------------------------------

            print(
                "\033[2J\033[H",
                end=""
            )


            print(
                "=" * 65
            )

            print(
                "FAST LIVE EEG COMMUNICATION PROTOTYPE"
            )

            print(
                "=" * 65
            )

            print()


            print(
                "F3 → LEFT"
            )

            print(
                "F4 → RIGHT"
            )

            print()


            print(
                "SIGNAL"
            )

            print(
                "------"
            )

            print(
                f"Window:           "
                f"{WINDOW_SECONDS:.2f} sec"
            )

            print(
                f"Update:           "
                f"{UPDATE_SECONDS:.2f} sec"
            )

            print()


            print(
                "ALPHA POWER"
            )

            print(
                "-----------"
            )

            print(
                f"F3 alpha:         "
                f"{current_left_power:.6f}"
            )

            print(
                f"F4 alpha:         "
                f"{current_right_power:.6f}"
            )

            print()


            print(
                "NORMALIZED"
            )

            print(
                "----------"
            )

            print(
                f"F3 normalized:    "
                f"{left_score:.3f}"
            )

            print(
                f"F4 normalized:    "
                f"{right_score:.3f}"
            )

            print()


            print(
                "DIRECTION"
            )

            print(
                "---------"
            )

            print(
                f"Raw index:        "
                f"{raw_index:+.4f}"
            )

            print(
                f"Smoothed index:   "
                f"{smoothed_index:+.4f}"
            )

            print()


            print(
                "CLASSIFICATION"
            )

            print(
                "--------------"
            )

            print(
                f"Threshold:        "
                f"±{ENTER_THRESHOLD:.3f}"
            )

            print(
                f"Requested:        "
                f"{requested_state}"
            )

            print(
                f"Stable state:     "
                f"{stable_state}"
            )

            print()


            # -----------------------------------------------
            # LED DISPLAY
            # -----------------------------------------------

            if stable_state == "LEFT":

                print(
                    "LED:              ← LEFT"
                )

            elif stable_state == "RIGHT":

                print(
                    "LED:              RIGHT →"
                )

            else:

                print(
                    "LED:              — NEUTRAL —"
                )


            print()

            print(
                "Arduino:          ",
                stable_state
            )

            print()

            print(
                f"Baseline adapt:   "
                f"{BASELINE_ADAPT_RATE}"
            )

            print(
                f"Smoothing:        "
                f"{SMOOTHING_POINTS}"
            )

            print()

            print(
                "=" * 65
            )


    # ========================================================
    # CTRL+C
    # ========================================================

    except KeyboardInterrupt:

        print()

        print(
            "Stopping..."
        )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    except Exception as e:

        print()

        print(
            "ERROR:"
        )

        print(e)

        print()


    # ========================================================
    # CLEAN SHUTDOWN
    # ========================================================

    finally:

        try:

            send_to_arduino(
                arduino,
                "NEUTRAL"
            )

            arduino.close()

        except Exception:

            pass


        try:

            board.stop_stream()

        except Exception:

            pass


        try:

            board.release_session()

        except Exception:

            pass


        print()

        print(
            "EEG stopped."
        )

        print(
            "Arduino set to NEUTRAL."
        )

        print(
            "Done."
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()