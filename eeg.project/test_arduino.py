import serial
import time

# CHANGE THIS to the Arduino port that worked for you
ARDUINO_PORT = "/dev/cu.usbmodem101"

BAUD_RATE = 115200

arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)

# Give Arduino time to reset after opening serial
time.sleep(2)

print("Arduino connected!")
print("Testing LEFT...")
arduino.write(b"L\n")
time.sleep(2)

print("Testing NEUTRAL...")
arduino.write(b"N\n")
time.sleep(2)

print("Testing RIGHT...")
arduino.write(b"R\n")
time.sleep(2)

print("Testing NEUTRAL...")
arduino.write(b"N\n")
time.sleep(2)

arduino.close()

print("Test complete!")