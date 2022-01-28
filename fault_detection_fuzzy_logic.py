# Fault detection fuzzy logic with real data
# Libraries
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import RPi.GPIO as GPIO
from lib_nrf24 import NRF24
import time
import spidev
import os
import tflite_runtime.interpreter as tflite
import psutil

# Fuzzy setup and variables 
# Univers variables
battery_universe=[0,100,1]
temp_universe=[0,40,1]
cpu_util_universe=[0,100,1]
response_time_universe=[0,4000,1]
output_fail_universe=[0,100,1]

# Inputs universe variables and membership
battery = ctrl.Antecedent(np.arange(battery_universe[0], battery_universe[1], battery_universe[2]), "Battery charge level")
temp = ctrl.Antecedent(np.arange(temp_universe[0], temp_universe[1], temp_universe[2]), "Battery temperature")
cpu_util=ctrl.Antecedent(np.arange(cpu_util_universe[0], cpu_util_universe[1], cpu_util_universe[2]),"CPU utilization")
response_time=ctrl.Antecedent(np.arange(response_time_universe[0],response_time_universe[1],response_time_universe[2]),"Response time")

# Output universe variable and membership
output_fail = ctrl.Consequent(np.arange(output_fail_universe[0], output_fail_universe[1], output_fail_universe[2]), "Failure-risk")

# Custom memebership functions
battery["poor"] = fuzz.trapmf(battery.universe, [0, 0, 20, 40])
battery["normal"] = fuzz.trapmf(battery.universe, [30, 35, 75, 80])
battery["optimum"] = fuzz.trapmf(battery.universe, [70, 80, 90, 100])

temp["optimum"] = fuzz.trapmf(temp.universe, [0, 0, 20, 25])
temp["normal"] = fuzz.trapmf(temp.universe, [20, 25, 30, 35])
temp["poor"] = fuzz.trapmf(temp.universe, [30, 35, 40, 40])

cpu_util["optimum"] = fuzz.trapmf(cpu_util.universe, [0, 0, 20, 30])
cpu_util["normal"] = fuzz.trapmf(cpu_util.universe, [20, 30, 60, 70])
cpu_util["poor"] = fuzz.trapmf(cpu_util.universe, [60, 70, 90, 100])

response_time["optimum"] = fuzz.trapmf(response_time.universe, [0, 0, 100, 300])
response_time["normal"] = fuzz.trapmf(response_time.universe, [250,500,1200,1500])
response_time["poor"] = fuzz.trapmf(response_time.universe, [1400, 1500, 4000, 4000])

output_fail["very-low"] = fuzz.trapmf(output_fail.universe, [0,0,17,23])
output_fail["low"] = fuzz.trapmf(output_fail.universe, [16,24,56,64])
output_fail["high"] = fuzz.trapmf(output_fail.universe, [57,63,86,92])
output_fail["very-high"] = fuzz.trapmf(output_fail.universe, [88,90,100,100])

# Fuzzy rules
rules = []
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["poor"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["poor"] & response_time["normal"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["poor"] & response_time["optimum"], output_fail["very-high"]))

rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["normal"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["normal"] & response_time["normal"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["normal"] & response_time["optimum"], output_fail["high"]))

rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["optimum"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["optimum"] & response_time["normal"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["poor"] & cpu_util["optimum"] & response_time["optimum"], output_fail["high"]))

rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["poor"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["poor"] & response_time["normal"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["poor"] & response_time["optimum"], output_fail["high"]))

rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["normal"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["normal"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["normal"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["optimum"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["optimum"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["poor"] & battery["normal"] & cpu_util["optimum"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["poor"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["poor"] & response_time["normal"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["poor"] & response_time["optimum"], output_fail["high"]))

rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["normal"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["normal"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["normal"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["optimum"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["optimum"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["poor"] & battery["optimum"] & cpu_util["optimum"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["poor"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["poor"] & response_time["normal"], output_fail["high"]))
rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["poor"] & response_time["optimum"], output_fail["high"]))

rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["normal"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["normal"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["normal"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["optimum"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["optimum"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["poor"] & cpu_util["optimum"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["poor"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["poor"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["poor"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["normal"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["normal"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["normal"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["optimum"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["optimum"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["normal"] & cpu_util["optimum"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["poor"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["poor"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["poor"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["normal"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["normal"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["normal"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["optimum"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["optimum"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["normal"] & battery["optimum"] & cpu_util["optimum"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["poor"] & response_time["poor"], output_fail["very-high"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["poor"] & response_time["normal"], output_fail["high"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["poor"] & response_time["optimum"], output_fail["high"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["normal"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["normal"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["normal"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["optimum"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["optimum"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["poor"] & cpu_util["optimum"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["poor"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["poor"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["poor"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["normal"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["normal"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["normal"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["optimum"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["optimum"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["normal"] & cpu_util["optimum"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["poor"] & response_time["poor"], output_fail["high"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["poor"] & response_time["normal"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["poor"] & response_time["optimum"], output_fail["low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["normal"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["normal"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["normal"] & response_time["optimum"], output_fail["very-low"]))

rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["optimum"] & response_time["poor"], output_fail["low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["optimum"] & response_time["normal"], output_fail["very-low"]))
rules.append(ctrl.Rule(temp["optimum"] & battery["optimum"] & cpu_util["optimum"] & response_time["optimum"], output_fail["very-low"]))

# Control System Creation and Simulation
# Create a control system
failure_control = ctrl.ControlSystem(rules)

# Create a control system simualtion
fail = ctrl.ControlSystemSimulation(failure_control)

# GPIO mode
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
        # Set global packets lost to 0 after there's a reply from the slave
        global_packets_lost = 0
        # Fuzzy logic
        fuzzy_logic(slave_state)
        
def fuzzy_logic(slave_state):
    global global_master_message
    # Pass inputs to the ControlSystem 
    fail.input["Battery charge level"] = float(slave_state[1])
    fail.input["Battery temperature"]= float(slave_state[0])
    fail.input["CPU utilization"]= float(slave_state[2])
    fail.input["Response time"]=float(slave_state[3])
    # Compute
    fail.compute()
    # CPU utilization
    cpu_utilization = psutil.cpu_percent(interval = 0.1)
    # Print device state and prediction
    print(f'B. Temperature [°C]: {slave_state[0]:<6} - B. Charge [%]: {slave_state[1]:<6} - '
          f'CPU [%]: {slave_state[2]:<6} - Response time [ms]: {slave_state[3]:<6} - Packets lost: {slave_state[4]:<6}'
          f'Risk of failure [%]: {fail.output["Failure-risk"]:<6.2f}, CPU master[%]: {cpu_utilization:<6.2f}')
    # Update master message
    global_master_message[1] = f'{fail.output["Failure-risk"]: .2f}'
    
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
        



