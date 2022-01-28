# Raspberry pi Neural
# Libraries
import RPi.GPIO as GPIO
from lib_nrf24 import NRF24
import time
import spidev
import os
import tflite_runtime.interpreter as tflite
import numpy as np
import psutil

GPIO.setmode(GPIO.BCM);

# Global Variables
global_master_message = ["stateRequest", "0.0"]
global_packets_lost = 0

# Transceiver initialization
# Pipes setup
pipes = [[0xe7, 0xe7, 0xe7, 0xe7, 0xe7], [0xc2, 0xc2, 0xc2, 0xc2, 0xc2]]
# Radio setup
radio = NRF24(GPIO, spidev.SpiDev());
radio.begin(0, 5);
# Stabilization delay
time.sleep(1);
# Radio setup
radio.setRetries(15,15);
# Set to maximum dynamic payload according to received_messagesheet
radio.setPayloadSize(32);
# Random channel we have 126 channels to choose from
radio.setChannel(0x64);
radio.setDataRate(NRF24.BR_1MBPS);
radio.setPALevel(NRF24.PA_LOW);
radio.setAutoAck(False);
radio.enableDynamicPayloads();
# radio.enableAckPayload();
# Open radio pipes
radio.openWritingPipe(pipes[0])
radio.openReadingPipe(1, pipes[1])
# Stop listening since this py will act as a master 
radio.stopListening();

# Neural network initialization
# Get path to current working directory
CWD_PATH = os.getcwd()
# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,"model.tflite")
# Interpreter
interpreter = tflite.Interpreter(model_path=PATH_TO_CKPT)
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Transceiver functions
# Function used to request state and transfer failure risk
def radio_send_data():
    separator = ","
    master_message = separator.join(global_master_message)
    master_message = list(master_message)
    master_message.append(0)
    radio.write(master_message)
    
# Function used to decode received messages from the slave
def radio_decode(message):
    global global_packets_lost
    if len(message) > 0:
        # Form a slave_state list
        slave_state = message.split(",")
        slave_state.append(str(global_packets_lost))
        # Print device state
        print(f'Temperature [°C]: {slave_state[0]:<6} - Battery charge [%]: {slave_state[1]:<6} - '
              f'CPU Utilization [%]: {slave_state[2]:<6} - Response time [ms]: {slave_state[3]:<6} - Packets lost: {slave_state[4]}')
        # Set global packets lost to 0 after there's a reply from the slave
        global_packets_lost = 0
        # Neural network
        neural_network(slave_state)
        
def neural_network(slave_state):
    global global_master_message
    # Battery charge, battery temperature, cpu utilization, response time
    input_data = np.array([[float(slave_state[1]),float(slave_state[0]),float(slave_state[2]),float(slave_state[3])]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # CPU utilization
    cpu_utilization = psutil.cpu_percent(interval = 0.1)

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f'Neural network risk of failure prediction: {output_data[0][0]: .2f}  -  CPU utilizatio master: {cpu_utilization}')
    # Update master message
    global_master_message[1] = f'{output_data[0][0]: .2f}'
    
# Main loop executed every second
while True:
    initial_time = time.time()
    # Send request
    radio_send_data()
    # Start listening for a response
    radio.startListening()
    # Wait for the radio to get a response if there's no response in 1 sec we raise a flag
    while not radio.available():
        time.sleep(0.01)
        if(time.time() - initial_time) > 1:
            # Update global packets lost if there's no reply from the slave
            global_packets_lost += 1
            print("Packet lost")
            break
    # Read received_message
    received_message = []
    radio.read(received_message, radio.getDynamicPayloadSize())
    string_message = ""
    # Loop over the received received_message
    for n in received_message:
        # We discard any special character, check ascii code for more info
        if(n >= 32 and n <= 126):
            string_message += chr(n)
    radio_decode(string_message)
    # Stop listening
    radio.stopListening()
    # Clean radio buffer
    radio.flush_rx()
    # Sleep for five seconds
    time.sleep(1);

GPIO.cleanup();
        
        


