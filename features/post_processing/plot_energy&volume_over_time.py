

# 根据 UUID4 创建一个文件夹
from matplotlib import pyplot as plt
import uuid
import os


def get_unique_path():
    output_path = "/tmp/"+str(uuid.uuid4())
    os.makedirs(output_path)
    return output_path

# 将面积数据导出到一个文件中
def extract_data_from_log(log_path, output_path):
    command_extract = f"""
    grep "The volume of the solid"  {log_path} >> {output_path}/volumes.txt
    grep "The potential_energy"     {log_path} >> {output_path}/potential_energy.txt
    grep "The kinematic_energy"     {log_path} >> {output_path}/kinematic_energy.txt
    """
    os.system(command_extract)

# 匹配科学计数法表示的数的正则表达式
def extract_data_by_line(filename):
    import re
    pattern = r"[-+]?\d\.\d+[eE][-+]\d+"
    potential_energy = []
    # 打开文件进行读取
    with open(filename, "r") as file:
        # 逐行读取文件内容，并将每行数据添加到列表中
        lines = file.readlines()
        for line in lines:
            matches = re.findall(pattern, line)
            print(matches)
            potential_energy.append(float(matches[0]))
    return potential_energy

# 打开文件进行读取
legends = [
    "BE-FE",
    "BE-BE",
    "BE-FE, $\Delta t = 0.0005$"
]
dt = 1e-1
total_time = 10.0

log_path = [
    # "/home/kokkos/ssh/npuheart/build_amend_noad/lid_driven_sphere_30_64_5000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_amend_noad/lid_driven_sphere_30_64_10000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    "/home/kokkos/ssh/npuheart/build_amend_noad/lid_driven_sphere_30_64_20000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    "/home/kokkos/ssh/npuheart/build_amend_noad/lid_driven_sphere_implicit_30_64_5000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_amend_1/lid_driven_sphere_30_64_5000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_amend_1/lid_driven_sphere_30_64_10000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_amend_1/lid_driven_sphere_30_64_20000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_sphere/lid_driven_sphere_implicit_30_64_5000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_sphere/lid_driven_sphere_implicit_30_64_10000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
    # "/home/kokkos/ssh/npuheart/build_sphere/lid_driven_sphere_implicit_30_64_20000_10.0_with_1.00e-02_1.00e-01_1.00e+02/everything.log",
]

num_demo = len(log_path)
output_path = [get_unique_path() for _ in range(num_demo)]

for i in range(num_demo):
    extract_data_from_log(log_path[i], output_path[i])


areas = [extract_data_by_line(f"{output_path[i]}/volumes.txt")
         for i in range(num_demo)]
potential_energy = [extract_data_by_line(
    f"{output_path[i]}/potential_energy.txt") for i in range(num_demo)]
kinematic_energy = [extract_data_by_line(
    f"{output_path[i]}/kinematic_energy.txt") for i in range(num_demo)]

times = [
    [i/2000 for i in range(len(kinematic_energy[0]))],
    # [i/1000 for i in range(len(kinematic_energy[1]))],
    # [i/2000 for i in range(len(kinematic_energy[2]))],
    # [i/500 for i in range(len(kinematic_energy[3]))],
    # [i/1000 for i in range(len(kinematic_energy[4]))],
    # [i/2000 for i in range(len(kinematic_energy[5]))],
    [i/500 for i in range(len(kinematic_energy[1]))],
    # [i*dt for i in range(len(kinematic_energy[2]))],
    # # [i*dt for i in range(len(kinematic_energy[2]))],
    # [i*dt for i in range(len(kinematic_energy[6]))],
    # [i*dt for i in range(len(kinematic_energy[7]))],
    # [i*dt for i in range(len(kinematic_energy[8]))],
]

# 数据处理
# areas = extract_data_by_line(f"{output_path[0]}/volumes.txt")
# areas_2 = extract_data_by_line(f"{output_path[1]}/volumes.txt")
# areas_3 = extract_data_by_line(f"{output_path[2]}/volumes.txt")
# 1. 体积变化率，将第一个数据近似成初始时刻的数据
# area_change = [abs(areas[i]-areas[0])/areas[0]*100 for i in range(len(areas))]
# area_change[0] = 1e-10 # avoid log(0)

# 2. 总能量
total_energy = [
    [potential_energy[j][i]+kinematic_energy[j][i] for i in range(len(potential_energy[j]))] for j in range(num_demo)
]

# 画图函数


def plot(h_list, e_list, x_axis='x', y_axis='y', title='title', markers=["D", "x", "+", 'v', '^', '1', "2", "3", "4"], colors=["red", "blue", "#34bf49"], legends=[]):
    plt.figure()
    fig, ax = plt.subplots()
    for i in range(len(h_list)):
        h, e = h_list[i], e_list[i]
        # if len(h) < 300:
        #     h = h[:len(h)//2]
        #     e = e[:len(e)//2]
        while len(h) > 50:
            h = h[::3]
            e = e[::3]
        # plt.plot(h,e,marker=markers[i%len(markers)], linestyle='dashed', linewidth=1, markersize=2)
        plt.scatter(h, e, marker=markers[i % len(
            markers)],color=colors[i % len(colors)], s=10)
    plt.legend(legends)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    plt.xlim(0, total_time)
    # plt.xlim(1.44, 1.46)
    plt.ylim(0, 0.1)
    # ax.set_xscale('linear')
    # ax.set_yscale('log')
    plt.savefig(title+'.jpg', dpi=1000)
    plt.close()


plot(times, total_energy, x_axis='$t(s)$',
     y_axis='', title='$\Delta t = 0.0005$', legends=legends)

print(output_path[0])


# import re
# pattern = r"[-+]?\d\.\d+[eE][-+]\d+"  # 匹配科学计数法表示的数的正则表达式


#   # 文件名


# sns = []
# # 打开文件进行读取
# with open(filename, "r") as file:
#     # 逐行读取文件内容，并将每行数据添加到列表中
#     lines = file.readlines()
#     for line in lines:
#         matches = re.findall(pattern, line)
#         sns.append(float(matches[0]))


# all = [sns[i]+dns[i] for i in range(len(dns))]

# plot([times[:100], times[:100], times[:100]],[dns[:100],sns[:100],all[:100]],x_axis='$t(s)$',y_axis='$\\frac{S_t-S_0}{S_0}\\times 100\%$',title='aaaaa',legends=['implicit', "explicit", "total"])
# plot([times[:100]],[dns[:100]],x_axis='$t(s)$',y_axis='$\\frac{S_t-S_0}{S_0}\\times 100\%$',title='bbbbb',legends=['implicit', "explicit", "total"])
