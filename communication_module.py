# Libraries
import RPi.GPIO as GPIO
from lib_nrf24 import NRF24
import time
import spidev

GPIO.setmode(GPIO.BCM)

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
#radio.enableAckPayload()
# Open radio pipes
radio.openWritingPipe(pipes[1])
radio.openReadingPipe(1, pipes[0])

# Print radio details
# radio.printDetails()

while True:
    # Start listening
    radio.startListening()
    
    if not radio.available(0):
        time.sleep(0.01)
    else:
        data = []
        radio.read(data, radio.getDynamicPayloadSize())
        message = ""
        # Loop over the received data
        for n in data:
            # We discard any special character, check ascii code for more info
            if(n >= 32 and n <= 126):
                message += chr(n)        
        print("{}". format(message))
        # Answer if there is a resquest from the master
        if message == "stateRequest":
            # Stop listening
            radio.stopListening()
            # Send response
            responseMessage = list("deviceResponse");
            # Add null byte at the end of the stringMessage
            responseMessage.append(0)
            radio.write(responseMessage)
            print("reponse sent")
        # Clean radio buffer
        radio.flush_rx()
GPIO.cleanup()
        
        
