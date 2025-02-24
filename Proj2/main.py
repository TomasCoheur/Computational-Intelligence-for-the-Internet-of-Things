import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

df = pd.read_csv(input("Insert the Name of the File:"))
del df['S1Temp']
del df['S2Temp']
del df['S3Temp']
df['Persons'].replace({1: 0, 2: 0, 3: 1})

co2_var_list = []
counter = 0
ten_last_values = [0]
previous_value = 390
for value in df['CO2'].values:
    if counter == 10:
        if value - ten_last_values[0] <= -15:
            co2_var_list.append(1)
        elif value - ten_last_values[0] <= -5:
            co2_var_list.append(0)
        elif value - ten_last_values[0] == 0:
            co2_var_list.append(2)
        elif value - ten_last_values[0] >= 15:
            co2_var_list.append(4)
        elif value - ten_last_values[0] >= 5:
            co2_var_list.append(3)
        ten_last_values.pop(0)
        ten_last_values.append(value)
    else:
        result = value - previous_value
        ten_last_values.append(value)
        counter += 1
        if result == -5:
            co2_var_list.append(0)
        elif result == 0:
            co2_var_list.append(2)
        elif result == 5:
            co2_var_list.append(3)

df['CO2Var'] = co2_var_list

s1_light_min = df['S1Light'].min()
s1_light_max = df['S1Light'].max()

s2_light_min = df['S2Light'].min()
s2_light_max = df['S2Light'].max()

s3_light_min = df['S3Light'].min()
s3_light_max = df['S3Light'].max()

time_min = df['Time'].min()
time_max = df['Time'].max()

co2_val_min = df['CO2'].min()
co2_val_max = df['CO2'].max()

co2_var_min = df['CO2Var'].min()
co2_var_max = df['CO2Var'].max()

time = ctrl.Antecedent(np.arange(time_min, time_max, 1), 'time')
time['day'] = fuzz.trimf(time.universe, [time_min, time_min, time_max])
time['night'] = fuzz.trimf(time.universe, [time_min, time_max, time_max])

s3_light = ctrl.Antecedent(np.arange(s3_light_min, s3_light_max, 1), 's3_light')
s3_light['low'] = fuzz.trimf(s3_light.universe, [s3_light_min, s3_light_min, (s3_light_max + s3_light_min) / 2])
s3_light['medium'] = fuzz.trimf(s3_light.universe, [s3_light_min, (s3_light_max + s3_light_min) / 2, s3_light_max])
s3_light['high'] = fuzz.trimf(s3_light.universe, [(s3_light_max + s3_light_min) / 2, s3_light_max, s3_light_max])

out_s3_light = ctrl.Consequent(np.arange(s3_light_min, s3_light_max, 1), 'out_s3_light')
out_s3_light['off'] = fuzz.trimf(out_s3_light.universe, [s3_light_min, s3_light_min, s3_light_max])
out_s3_light['on'] = fuzz.trimf(out_s3_light.universe, [s3_light_min, s3_light_max, s3_light_max])

s1_light = ctrl.Antecedent(np.arange(s1_light_min, s1_light_max, 1), 's1_light')
s1_light['off'] = fuzz.trimf(s1_light.universe, [s1_light_min, s1_light_min, s1_light_max])
s1_light['on'] = fuzz.trimf(s1_light.universe, [s1_light_min, s1_light_max, s1_light_max])

s2_light = ctrl.Antecedent(np.arange(s2_light_min, s2_light_max, 1), 's2_light')
s2_light['off'] = fuzz.trimf(s2_light.universe, [s2_light_min, s2_light_min, s2_light_max])
s2_light['on'] = fuzz.trimf(s2_light.universe, [s2_light_min, s2_light_max, s2_light_max])

out_light = ctrl.Consequent(np.arange(0, 2, 1), 'out_light')
out_light['empty'] = fuzz.trimf(out_light.universe, [0, 0, 1])
out_light['crowded'] = fuzz.trimf(out_light.universe, [0, 1, 1])

co2_var = ctrl.Antecedent(np.arange(0, 5, 1), 'co2_var')
co2_var['very slow'] = fuzz.trimf(co2_var.universe, [0, 0, 1])
co2_var['slow'] = fuzz.trimf(co2_var.universe, [0, 1, 2])
co2_var['medium'] = fuzz.trimf(co2_var.universe, [1, 2, 3])
co2_var['fast'] = fuzz.trimf(co2_var.universe, [2, 3, 4])
co2_var['very fast'] = fuzz.trimf(co2_var.universe, [3, 4, 4])

co2_val = ctrl.Antecedent(np.arange(co2_val_min, co2_val_max, 1), 'co2_val')
co2_val['low'] = fuzz.trimf(co2_val.universe, [co2_val_min, co2_val_min, (co2_val_max + co2_var_min) / 2])
co2_val['medium'] = fuzz.trimf(co2_val.universe, [co2_val_min, (co2_val_max + co2_var_min) / 2, co2_val_max])
co2_val['high'] = fuzz.trimf(co2_val.universe, [(co2_val_max + co2_var_min) / 2, co2_val_max, co2_val_max])

out_co2 = ctrl.Consequent(np.arange(0, 2, 1), 'out_co2')
out_co2['empty'] = fuzz.trimf(out_co2.universe, [0, 0, 1])
out_co2['crowded'] = fuzz.trimf(out_co2.universe, [0, 1, 1])

pir1 = ctrl.Antecedent(np.arange(0, 2, 1), 'pir1')
pir1['stopped'] = fuzz.trimf(pir1.universe, [0, 0, 1])
pir1['moved'] = fuzz.trimf(pir1.universe, [0, 1, 1])

pir2 = ctrl.Antecedent(np.arange(0, 2, 1), 'pir2')
pir2['stopped'] = fuzz.trimf(pir2.universe, [0, 0, 0])
pir2['moved'] = fuzz.trimf(pir2.universe, [1, 1, 1])

out_pir = ctrl.Consequent(np.arange(0, 2, 1), 'out_pir')
out_pir['empty'] = fuzz.trimf(out_pir.universe, [0, 0, 1])
out_pir['crowded'] = fuzz.trimf(out_pir.universe, [0, 1, 1])

final_out = ctrl.Consequent(np.arange(0, 2, 1), 'final_out')
final_out['empty'] = fuzz.trimf(final_out.universe, [0, 0, 1])
final_out['crowded'] = fuzz.trimf(final_out.universe, [0, 1, 1])

# Rules for the light sensor 3
rule1 = ctrl.Rule(time['day'] & s3_light['low'], out_s3_light['off'])
rule2 = ctrl.Rule(time['day'] & s3_light['medium'], out_s3_light['off'])
rule3 = ctrl.Rule(time['day'] & s3_light['high'], out_s3_light['on'])

rule4 = ctrl.Rule(time['night'] & s3_light['low'], out_s3_light['off'])
rule5 = ctrl.Rule(time['night'] & s3_light['medium'], out_s3_light['off'])
rule6 = ctrl.Rule(time['night'] & s3_light['high'], out_s3_light['on'])

s3_light_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
s3_light_ctrl_sim = ctrl.ControlSystemSimulation(s3_light_ctrl)
ctrl_out_s3_light = []
for row_index, row in df.iterrows():
    s3_light_ctrl_sim.input['time'] = df['Time'][row_index]
    s3_light_ctrl_sim.input['s3_light'] = df['S3Light'][row_index]

    # Crunch the numbers
    s3_light_ctrl_sim.compute()

    ctrl_out_s3_light.append(round(s3_light_ctrl_sim.output['out_s3_light']))

df['GoodS3Light'] = ctrl_out_s3_light
#print(ctrl_out_s3_light)
out_s3_light_min = min(ctrl_out_s3_light)
out_s3_light_max = max(ctrl_out_s3_light)

out_s3_light_ant = ctrl.Antecedent(np.arange(out_s3_light_min, out_s3_light_max, 1), 'out_s3_light_ant')
out_s3_light_ant['on'] = fuzz.trimf(out_s3_light_ant.universe, [out_s3_light_min, out_s3_light_min, out_s3_light_max])
out_s3_light_ant['off'] = fuzz.trimf(out_s3_light_ant.universe, [out_s3_light_min, out_s3_light_max, out_s3_light_max])


# Rules for output of all light sensors
rule7 = ctrl.Rule(s1_light['off'] & s2_light['off'] & out_s3_light_ant['off'], out_light['empty'])
rule8 = ctrl.Rule(s1_light['off'] & s2_light['off'] & out_s3_light_ant['on'], out_light['empty'])
rule9 = ctrl.Rule(s1_light['off'] & s2_light['on'] & out_s3_light_ant['off'], out_light['empty'])
rule10 = ctrl.Rule(s1_light['on'] & s2_light['off'] & out_s3_light_ant['off'], out_light['empty'])
rule11 = ctrl.Rule(s1_light['on'] & s2_light['on'] & out_s3_light_ant['off'], out_light['empty'])
rule12 = ctrl.Rule(s1_light['on'] & s2_light['off'] & out_s3_light_ant['on'], out_light['empty'])
rule13 = ctrl.Rule(s1_light['off'] & s2_light['on'] & out_s3_light_ant['on'], out_light['empty'])
rule14 = ctrl.Rule(s1_light['on'] & s2_light['on'] & out_s3_light_ant['on'], out_light['crowded'])

lights_ctrl = ctrl.ControlSystem([rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14])

lights_ctrl_sim = ctrl.ControlSystemSimulation(lights_ctrl)
ctrl_out_lights = []
for row_index, row in df.iterrows():
    s3_light_ctrl_sim.input['time'] = df['Time'][row_index]
    s3_light_ctrl_sim.input['s3_light'] = df['S3Light'][row_index]
    lights_ctrl_sim.input['s1_light'] = df['S1Light'][row_index]
    lights_ctrl_sim.input['s2_light'] = df['S2Light'][row_index]
    lights_ctrl_sim.input['out_s3_light_ant'] = df['GoodS3Light'][row_index]
    # Crunch the numbers
    lights_ctrl_sim.compute()

    ctrl_out_lights.append(round(lights_ctrl_sim.output['out_light']))

df['NrPersonLights'] = ctrl_out_lights
#print(ctrl_out_lights)
lights_min = min(ctrl_out_lights)
lights_max = max(ctrl_out_lights)

out_lights_ant = ctrl.Antecedent(np.arange(lights_min, lights_max, 1), 'out_lights_ant')
out_lights_ant['empty'] = fuzz.trimf(out_lights_ant.universe, [lights_min, lights_min, lights_max])
out_lights_ant['crowded'] = fuzz.trimf(out_lights_ant.universe, [lights_min, lights_max, lights_max])


# Rules for output of PIR sensors
rule15 = ctrl.Rule(pir1['stopped'] & pir2['stopped'], out_pir['empty'])
rule16 = ctrl.Rule(pir1['stopped'] & pir2['moved'], out_pir['empty'])
rule17 = ctrl.Rule(pir1['moved'] & pir2['stopped'], out_pir['empty'])
rule18 = ctrl.Rule(pir1['moved'] & pir2['moved'], out_pir['crowded'])

pir_ctrl = ctrl.ControlSystem([rule15, rule16, rule17, rule18])
pir_ctrl_sim = ctrl.ControlSystemSimulation(pir_ctrl)
ctrl_out_pir = []
for row_index, row in df.iterrows():
    pir_ctrl_sim.input['pir1'] = df['PIR1'][row_index]
    pir_ctrl_sim.input['pir2'] = df['PIR2'][row_index]

    # Crunch the numbers
    pir_ctrl_sim.compute()

    ctrl_out_pir.append(round(pir_ctrl_sim.output['out_pir']))

df['NrPersonPIR'] = ctrl_out_pir
#print(ctrl_out_pir)
out_pir_ant = ctrl.Antecedent(np.arange(0, 2, 1), 'out_pir_ant')
out_pir_ant['empty'] = fuzz.trimf(out_pir_ant.universe, [0, 0, 1])
out_pir_ant['crowded'] = fuzz.trimf(out_pir_ant.universe, [0, 1, 1])

# Rules for output of CO2 sensors
rule19 = ctrl.Rule(co2_val['low'] & co2_var['very slow'], out_co2['empty'])
rule20 = ctrl.Rule(co2_val['low'] & co2_var['slow'], out_co2['empty'])
rule21 = ctrl.Rule(co2_val['low'] & co2_var['medium'], out_co2['empty'])
rule22 = ctrl.Rule(co2_val['low'] & co2_var['fast'], out_co2['empty'])
rule23 = ctrl.Rule(co2_val['low'] & co2_var['very fast'], out_co2['crowded'])

rule24 = ctrl.Rule(co2_val['medium'] & co2_var['very slow'], out_co2['empty'])
rule25 = ctrl.Rule(co2_val['medium'] & co2_var['slow'], out_co2['empty'])
rule26 = ctrl.Rule(co2_val['medium'] & co2_var['medium'], out_co2['empty'])
rule27 = ctrl.Rule(co2_val['medium'] & co2_var['fast'], out_co2['crowded'])
rule28 = ctrl.Rule(co2_val['medium'] & co2_var['very fast'], out_co2['crowded'])

rule29 = ctrl.Rule(co2_val['high'] & co2_var['very slow'], out_co2['empty'])
rule30 = ctrl.Rule(co2_val['high'] & co2_var['slow'], out_co2['empty'])
rule31 = ctrl.Rule(co2_val['high'] & co2_var['medium'], out_co2['empty'])
rule32 = ctrl.Rule(co2_val['high'] & co2_var['fast'], out_co2['crowded'])
rule33 = ctrl.Rule(co2_val['high'] & co2_var['very fast'], out_co2['crowded'])

co2_ctrl = ctrl.ControlSystem([rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26,
                               rule27, rule28, rule29, rule30, rule31, rule32, rule33])
co2_ctrl_sim = ctrl.ControlSystemSimulation(co2_ctrl)
ctrl_out_co2 = []
for row_index, row in df.iterrows():
    co2_ctrl_sim.input['co2_var'] = df['CO2Var'][row_index]
    co2_ctrl_sim.input['co2_val'] = df['CO2'][row_index]

    # Crunch the numbers
    co2_ctrl_sim.compute()
    ctrl_out_co2.append(round(co2_ctrl_sim.output['out_co2']))
df['NrPersonCO2'] = ctrl_out_co2
#print(ctrl_out_co2)
out_co2_ant = ctrl.Antecedent(np.arange(0, 2, 1), 'out_co2_ant')
out_co2_ant['empty'] = fuzz.trimf(out_co2_ant.universe, [0, 0, 1])
out_co2_ant['crowded'] = fuzz.trimf(out_co2_ant.universe, [0, 1, 1])

# Rules for final output
rule34 = ctrl.Rule(out_lights_ant['empty'] & out_pir_ant['empty'] & out_co2_ant['empty'], final_out['empty'])
rule35 = ctrl.Rule(out_lights_ant['empty'] & out_pir_ant['empty'] & out_co2_ant['crowded'], final_out['empty'])
rule36 = ctrl.Rule(out_lights_ant['empty'] & out_pir_ant['crowded'] & out_co2_ant['empty'], final_out['empty'])
rule37 = ctrl.Rule(out_lights_ant['crowded'] & out_pir_ant['empty'] & out_co2_ant['empty'], final_out['empty'])
rule38 = ctrl.Rule(out_lights_ant['crowded'] & out_pir_ant['crowded'] & out_co2_ant['empty'], final_out['crowded'])
rule39 = ctrl.Rule(out_lights_ant['crowded'] & out_pir_ant['empty'] & out_co2_ant['crowded'], final_out['crowded'])
rule40 = ctrl.Rule(out_lights_ant['empty'] & out_pir_ant['crowded'] & out_co2_ant['crowded'], final_out['crowded'])
rule41 = ctrl.Rule(out_lights_ant['crowded'] & out_pir_ant['crowded'] & out_co2_ant['crowded'], final_out['crowded'])


room_ctrl = ctrl.ControlSystem([rule34, rule35, rule36, rule37, rule38, rule39, rule40, rule41])
room_ctrl_sim = ctrl.ControlSystemSimulation(room_ctrl)
ctrl_out_room = []
for row_index, row in df.iterrows():
    room_ctrl_sim.input['out_lights_ant'] = df['NrPersonLights'][row_index]
    room_ctrl_sim.input['out_pir_ant'] = df['NrPersonPIR'][row_index]
    room_ctrl_sim.input['out_co2_ant'] = df['NrPersonCO2'][row_index]
    # Crunch the numbers
    room_ctrl_sim.compute()
    ctrl_out_room.append(round(room_ctrl_sim.output['final_out']))

df['FinalOutput'] = ctrl_out_room
#print(ctrl_out_room)

print("Confusion Matrix", confusion_matrix(ctrl_out_room, df['Persons']))
print("Macro-Precision: ", precision_score(ctrl_out_room, df['Persons'], average='macro'))
print("Macro-Recall: ", recall_score(ctrl_out_room, df['Persons'], average='macro'))
print("Macro-F1: ", f1_score(ctrl_out_room, df['Persons'], average='macro'))