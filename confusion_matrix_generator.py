# Regression neural network & fuzzy logic confusion matrix dataset generator
# Libraries
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import tflite_runtime.interpreter as tflite
import numpy as np
import matplotlib.pyplot as plt

# Fuzzy logic setup and variables
# Univers variables
battery_universe=[0,101,1]
temp_universe=[0,41,1]
cpu_util_universe=[0,101,1]
response_time_universe=[0,4001,1]
output_fail_universe=[0,101,1]

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

# Neural network variales and setup
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

# Dataset variable
dataset_data = []

# Confusion matriz variables
confusion_true_positive = 0
confusion_true_negative = 0
confusion_false_positive = 0
confusion_false_negative = 0


for index in range(2000):
    # Generate random inputs
    battery_charge_random = np.random.random()*100
    battery_temperature_random = np.random.random()*40
    cpu_utilization_random = np.random.random()*100
    response_time_random =np.random.random()*4000
    
    # Fuzzy logic risk of failure
    # Pass inputs to the ControlSystem 
    fail.input["Battery charge level"] = battery_charge_random
    fail.input["Battery temperature"]= battery_temperature_random
    fail.input["CPU utilization"]= cpu_utilization_random
    fail.input["Response time"]=response_time_random
    # Compute
    fail.compute()
    failure_risk_computed = fail.output["Failure-risk"]
    failure_risk_computed_category = 1 if failure_risk_computed >= 70 else 0
    
    # Neural network risk of failure
    # Battery charge, battery temperature, cpu, response
    input_data = np.array([[battery_charge_random,battery_temperature_random,cpu_utilization_random,response_time_random]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data[0][0]
    output_data_category = 1 if output_data >= 70 else 0
    
    # Confusion matrix variable
    confusion_label = "✓" if failure_risk_computed_category == output_data_category else "x"
    if failure_risk_computed_category == 1 and output_data_category == 1 :
        confusion_true_positive = confusion_true_positive + 1
    if failure_risk_computed_category == 0 and output_data_category == 0:
        confusion_true_negative = confusion_true_negative + 1
    if failure_risk_computed_category == 0 and output_data_category == 1:
        confusion_false_positive = confusion_false_positive + 1
    if failure_risk_computed_category == 1 and output_data_category == 0:
        confusion_false_negative = confusion_false_negative + 1
    
    # Save data
    dataset_data.append(f'{battery_charge_random:<10.4f} , {battery_temperature_random:<10.4f} , {cpu_utilization_random:<10.4f} , {response_time_random:<10.4f} ,'
                        f'{failure_risk_computed:<10.4f} , {failure_risk_computed_category:<10} ,{output_data:<10.4f}, {output_data_category :<10}'
                        f'{confusion_label}')
# Confusion matrix results
dataset_data.append(f'True positives: {confusion_true_positive:<10}, True negatives: {confusion_true_positive:<10}, False positives: {confusion_false_positive:<10},'
                    f'False negatives: {confusion_false_negative:<10}')
# Save data in a document
# Create and open a document with context manager called "failure_dataset"
with open("confusion_matrix_dataset.txt", "w") as document:
    for data in dataset_data:
        document.write(data + "\n")

# Dataset generated
print("Dataset generated")
# Pie chart
piechart_values = [confusion_true_positive, confusion_false_positive, confusion_true_negative, confusion_false_negative]
piechart_labels = ["True positives", "False positives", "True negatives", "Flase negatives"]
plt.axis("equal")
plt.title("Confusion Matrix")
plt.pie(piechart_values, labels = piechart_labels, radius = 1, autopct = '%0.2f%%')
plt.show()




