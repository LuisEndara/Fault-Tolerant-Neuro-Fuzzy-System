# libraries
import time
import board
from adafruit_ina219 import ADCResolution, BusVoltageRange, INA219

# Test info
# Load: Resistive load 15.3 ohms
# Battery: LiPo 900 mAh - TCBworth

# Bus initialization
i2c_bus = board.I2C()
ina219 = INA219(i2c_bus)

# optional : change configuration to use 32 samples averaging for both bus voltage and shunt voltage
# 12bit,  32 samples, 17.02ms
ina219.bus_adc_resolution = ADCResolution.ADCRES_12BIT_32S
ina219.shunt_adc_resolution = ADCResolution.ADCRES_12BIT_32S
# optional : change voltage range to 16V
ina219.bus_voltage_range = BusVoltageRange.RANGE_16V

# Variables
discharge_test = []

# measure and display loop
while True:
    try:
        # Sensor readings
        bus_voltage = ina219.bus_voltage # voltage on V- (load side)
        shunt_voltage = ina219.shunt_voltage # voltage between V+ and V- across the shunt
        current = ina219.current # current in mA
        power_supply_unit = bus_voltage + shunt_voltage
        discharge_test.append(f'{power_supply_unit:<8.4f} , {current/1000:<8.4f} , {shunt_voltage:<8.4f} , {bus_voltage:<8.4f}')
        # INA219 measure bus voltage on the load side. So PSU voltage = bus_voltage + shunt_voltage
        print("PSU Voltage: {:6.3f} V".format(bus_voltage + shunt_voltage))
        print("Shunt Voltage: {:9.6f} V".format(shunt_voltage))
        print("Load Voltage: {:6.3f} V".format(bus_voltage))
        print("Current: {:9.6f} A".format(current/1000))
        print("")
        # End test when the battery level is bellow 3V per cell 6 volts
        if power_supply_unit < 6:
            break
        time.sleep(1)
    # Test end when user presses ctr c
    except KeyboardInterrupt:
        break

print("Test completed")
# Save data in a document
# Create and open a document with context manager called "discharge_text"
with open("discharge_test.txt", "w") as document:
    for data in discharge_test:
        document.write(data + "\n")

