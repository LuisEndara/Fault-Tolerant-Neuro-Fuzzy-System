# Raspberry pi Device

# Libraries
import subprocess
import RPi.GPIO as GPIO
import time
import spidev
import glob
import board
import psutil
import threading
from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219
from lib_nrf24 import NRF24

# Global variables
global_device_state = [0.0, 0.0, 0.0, 0.0]

# GPIO Mode
GPIO.setmode(GPIO.BCM)

# Transceiver initialization
# Pipes for communication
pipes = [[0xe7, 0xe7, 0xe7, 0xe7, 0xe7], [0xc2, 0xc2, 0xc2, 0xc2, 0xc2]]
# Radio setup
radio = NRF24(GPIO, spidev.SpiDev())
radio.begin(0, 5)
# Stabilization delay
time.sleep(1)
# Radio setup
radio.setRetries(15,15)
# Set to maximum dynamic payload according to datasheet
radio.setPayloadSize(32)
# Random channel we have 126 channels to choose from
radio.setChannel(0x64)
radio.setDataRate(NRF24.BR_1MBPS)
radio.setPALevel(NRF24.PA_LOW)
radio.setAutoAck(False)
radio.enableDynamicPayloads()
# radio.enableAckPayload()
# Open radio pipes
radio.openWritingPipe(pipes[1])
radio.openReadingPipe(1, pipes[0])

# Ina219 initialization
# Bus I2C initialization ina219
i2c_bus = board.I2C()
ina219 = INA219(i2c_bus)
# 12bit,  32 samples, 17.02ms
ina219.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina219.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina219.bus_voltage_range = BusVoltageRange.RANGE_16V

# Transceiver functions
# Function used to decode received messages from the master
def radio_decode(message):
    message = message.split(",")
    # Answer if there is a resquest from the master
    if(message[0] == "stateRequest"):
        # Stop listening
        radio.stopListening()
        # Send response
        radio_send_state();
    # Take actions depending on the risk of failure
    failure_handler(message[1])
    
        
def radio_send_state():
    # Get device state
    device_state = global_device_state
    for index in range(len(device_state)):
        device_state[index] = format(float(device_state[index]), ".2f")
    separator = ","
    device_state = separator.join(device_state)
    # Print device state
    device_state_list = device_state.split(",")
    print(f'Temperature [°C]: {device_state_list[0]:<6} - Battery charge [%]: {device_state_list[1]:<6} - CPU Utilization [%]: {device_state_list[2]:<6} - '
          f'Response time [ms]: {device_state_list[3]:<6}')
    # Prepare device_state to be sent 
    device_state = list(device_state)
    device_state.append(0)
    radio.write(device_state)
    
# DS18B20 Functions
# Get info from the file
def read_temp_raw():
    # Construct device file name
    # Define directory
    base_dir = '/sys/bus/w1/devices/'
    # Get the first available folder called '/sys/bus/w1/devices/28'
    device_folder = glob.glob(base_dir + '28*')[0]
    # Get file called '/sys/bus/w1/devices/28 "wild card"/w1_slave'
    device_file = device_folder + '/w1_slave'
    # Read file with context manager, file is closed automatically by the conext manager
    with open(device_file, 'r') as f:
        f = open(device_file, 'r')
        # Get all lines as a list
        lines = f.readlines()
    # Return list of lines
    return lines

# Get temperature from the one wire file 
def read_temp():
    lines = read_temp_raw()
    # Check wheter we have more than 1 line 
    if (len(lines) > 0):
        # Wait until there is a reading
        while lines[0].strip()[-3:] != 'YES':
            time.sleep(0.2)
            lines = read_temp_raw()
        # Get the position when temperature reading start
        equals_pos = lines[1].find('t=')
        # If the position exist a reading exists too
        if equals_pos != -1:
            temp_string = lines[1][equals_pos+2:]
            temp_c = float(temp_string) / 1000.0
            # We won't use this temperature
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            return temp_c
    else:
        return False
    
# Function used to gather sensor data
def device_get_state():
    global global_device_state
    # Temperature in degrees celsious
    local_temperature = read_temp()
    # We make sure we have a valid data storage in temperature
    temperature = local_temperature if local_temperature != False else global_device_state[0]
    # Global CPU utilization in percentage
    utilization = psutil.cpu_percent(interval = 0.1)
    # Get state inside a global variable
    global_device_state[0] = temperature
    global_device_state[2] = utilization
    
# Function used to handle the risk of failure calculated in the master
def failure_handler(failure_risk):
    	#shutdown rutine
	#if(float(failure_risk) >=90):
	#subprocess.call(["sudo", "shutdown", "-h", "now"])
	#os.system("sudo shutdown -h now")
	if(float(failure_risk) >= 10):
       
        print(f'Risk of failure calculated by master: {failure_risk}')
        


# State of charge loop (the battery must be connected before starting this script otherwise state_of_charge will get wrong stimations)
def state_charge():
    global global_device_state
    # Sensor readings
    bus_voltage = ina219.bus_voltage # voltage on V- (load side)
    shunt_voltage = ina219.shunt_voltage # voltage between V+ and V- across the shunt
    # Power supply voltage
    power_supply_unit = bus_voltage + shunt_voltage
    # Eq. for the state of charge y = -0.0119x + 7.9813 (y: discharge voltage, x: discharge capacity %), x = ( y - 7.9813 ) / -0.0119
    discharge_capacity = (power_supply_unit - 7.9813) / -0.0119
    # Clamp value
    discharge_capacity = 0 if discharge_capacity < 0 else 100 if discharge_capacity > 100 else discharge_capacity
    state_of_charge = 100 - discharge_capacity
    # Coulometer
    while True:
        # Sensor readings
        bus_voltage = ina219.bus_voltage # voltage on V- (load side)
        shunt_voltage = ina219.shunt_voltage # voltage between V+ and V- across the shunt
        # Power supply voltage
        power_supply_unit = bus_voltage + shunt_voltage
        current = ina219.current # current in mA
        # 968 mAh is the maximum capacity found through a discharge test, since we are calculating the current in mAs we have to transform it to hours so we
        # dived the number by 3600 finally we need to get a percentage so we mutiply by 100
        state_of_charge -= (current / 968.0 ) / 3600 * 100
        # We make sure the reading is right by clamping the state_of_charge between 10 and 100%
        state_of_charge = 10.0 if state_of_charge < 10.0 else 100.0 if state_of_charge > 100.0 else state_of_charge
        # Drift rises every time we use any kind of intergration, to be sure the state of charge is accurate we overwrite the state of charge when
        # the battery hits 7 volts, the equation for this region is the following y = -0.0995x +16.194 (y: discharge voltage, x: discharge capacity %)
        if power_supply_unit <= 7.0:
            discharge_capacity = (power_supply_unit - 16.194 ) / -0.0995
            discharge_capacity = 0 if discharge_capacity < 0 else 100 if discharge_capacity > 100 else discharge_capacity
            state_of_charge = 100 - discharge_capacity 
        time.sleep(1)
        # Update value in global variable
        global_device_state[1] = state_of_charge

# Sensor reading loop
def state_loop():
    while True:
        device_get_state()
        
# Main loop
def main_loop():
    global global_device_state
    while True:
        # Start listening
        radio.startListening()
        # Wait until there's data available
        if not radio.available(0):
            time.sleep(0.01)
        else:
            initialTime = time.time()
            received_message = []
            radio.read(received_message, radio.getDynamicPayloadSize())
            string_message = ""
            # Loop over the received data
            for n in received_message:
                # We discard any special character, check ascii code for more info
                if(n >= 32 and n <= 126):
                    string_message += chr(n)
            radio_decode(string_message)
            responseTime = time.time() - initialTime
            # Update response time
            global_device_state[3] = responseTime*1000
            # Clean radio buffer
            radio.flush_rx()
    GPIO.cleanup()

# Threads
# We will use three threades in order to optimize response time
state_of_charge = threading.Thread(target = state_charge)
state_of_charge.start()

state_thread = threading.Thread(target = state_loop)
state_thread.start()

main_thread = threading.Thread(target = main_loop)
main_thread.start()





        
