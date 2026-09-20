Hello!

Welcome to the code for NeuroTune. This code works by branching arduino, and EEG! 
1. after setting up EEG, plug it into your computer, note down the serial port in the proper location.
2. build Arduino circuit, this is a simple, grounded, 3 LED circuit, Green led goes to digital pin 8, 
yellow to Digital pin 10 and red to digital pin 9. 
3. plug the Arduino into the computer and update the serial inputs at the top of the code. 
4. insure the eeg is placed correctly on the head (we used the GUI to ensure that we were getting normalised waves)
5. run the code in live_eeg.py. the code will run indefinitely


Notes!
eeg_replay.py, is the file I used to run pre recorded data through to test my code and arduino set up
test_arduino.py is the file I used to blink the Arduino to see if the lights were working. 
