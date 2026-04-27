import json
import subprocess
import time

with open('demo.json', 'r') as file:
    data = json.load(file)

# 基本参数
processor = 30
processors = []
Nt_s = {40000, 80000}
dilation_radius_s = {0.3, 0.4, 0.5}
Nb_s = {80}

program = './test/fsi_real_MV_dilation'
command = 'taskset -c {processor} nohup {program} config/test{processor}.json > log/MV{processor}.out &'

for Nb in Nb_s:
    for Nt in Nt_s:
        for dilation_radius in dilation_radius_s:
            data['Nb'] = Nb
            data['Nt'] = Nt
            data['processor'] = processor
            data['dilation_radius'] = dilation_radius
            data['command'] = command.format(processor=processor, program=program)
            processors.append(processor)
            with open('config/test'+str(processor)+'.json', 'w') as file:
                json.dump(data, file,ensure_ascii=False, indent=4)
            processor += 1
        

for processor in processors:
    with open('config/test'+str(processor)+'.json', 'r') as file:
        data = json.load(file)
    print(data['command'])
