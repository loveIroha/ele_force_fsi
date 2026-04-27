
pressure_path = "/home/kokkos/geometry-tiny/MV-Gao/other/input_pressure.txt"
with open(pressure_path, 'r') as file:
    lines = file.readlines()

num_data = int(lines[0].strip())
time_step = float(lines[1].strip())

data = [-float(line.strip()) for line in lines[2:num_data+1]]
time = [i*time_step for i in range(len(data))]

time1 = [0.165]
data1 = [data[int(0.165/time_step)]]

time2 = [0.365]
data2 = [data[int(0.365/time_step)]]


time3 = [0.6]
data3 = [data[int(0.6/time_step)]]
time4 = [0.75]
data4 = [data[int(0.75/time_step)]]
time5 = [0.1]
data5 = [data[int(0.1/time_step)]]

from plot import plot_multiple_lines

types = ['lines','dots','dots','dots','dots','dots']
labels =[ 'pressure', '0.165', '0.365', '0.6', '0.75', '0.1']
plot_multiple_lines(
    [time,time1,time2,time3,time4,time5], 
    [data,data1,data2,data3,data4,data5],
    labels=labels,
    types = types)