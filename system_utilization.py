# Libraries
import psutil
import time

# Loop
while True:
    cpu_utilization = psutil.cpu_percent(interval = 0.1)
    cpu_utilization_each = psutil.cpu_percent(interval = 0.1, percpu = True)
    print("Global CPU Utilizacion: {}".format(cpu_utilization))
    print("Single CPU Utilizacion {}: ".format(cpu_utilization_each))
    time.sleep(1)