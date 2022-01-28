import glob
import time

# Construct device file name
# Define directory
base_dir = '/sys/bus/w1/devices/'
# Get the first available folder called '/sys/bus/w1/devices/28'
device_folder = glob.glob(base_dir + '28*')[0]
# Get file called '/sys/bus/w1/devices/28 "wild card"/w1_slave'
device_file = device_folder + '/w1_slave'

# Get info from the file
def read_temp_raw():
    # Read file
    f = open(device_file, 'r')
    # Get all lines as a list
    lines = f.readlines()
    # Close file
    f.close()
    # Return list of lines
    return lines

# Get temperature from the one wire file 
def read_temp():
    lines = read_temp_raw()
    # Get first lines without spaces and check the 3 characters
    # We wait until we have a reading
    while lines[0].strip()[-3:] != 'YES':
        time.sleep(0.2)
        lines = read_temp_raw()
    # Get the position when temperature reading start
    equals_pos = lines[1].find('t=')
    # If the position exist a reading exists too
    if equals_pos != -1:
        temp_string = lines[1][equals_pos+2:]
        temp_c = float(temp_string) / 1000.0
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return temp_c, temp_f

# Loop 
while True:
    print(read_temp())
    time.sleep(1)