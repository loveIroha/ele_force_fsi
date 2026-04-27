# 根据 UUID4 创建一个文件夹
from matplotlib import pyplot as plt
import uuid
import os


def get_unique_path():
    output_path = "./"+str(uuid.uuid4())
    os.makedirs(output_path)
    return output_path

# 将面积数据导出到一个文件中
def extract_data_from_log(log_path, output_path):
    command_extract = f"""
    grep "Current tension "  {log_path} >> {output_path}/volumes.txt
    """
    os.system(command_extract)


# 匹配科学计数法表示的数的正则表达式
def extract_data_by_line(filename, num_entries=1):
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
            for i in range(num_entries):
                potential_energy.append(float(matches[i]))
    return potential_energy

log_path = [
    # "/home/kokkos/ssh/npuheart/build_systole_1/real_BV_systole_3D_explicit_40_64_30000_2.0_with_4.00e-02_1.00e+06_1.00e+06/WARNING.log",
    "/home/kokkos/ssh/npuheart/build_systole_2/real_LV_new_mesh_systole_3D_explicit_40_64_30000_2.0_with_4.00e-02_5.00e+06_5.00e+06/WARNING.log"
]

num_demo = len(log_path)
output_path = [get_unique_path() for _ in range(num_demo)]

for i in range(num_demo):
    extract_data_from_log(log_path[i], output_path[i])

data1 = extract_data_by_line(f"{output_path[0]}/volumes.txt",3)

a, b, c, d = [], [], [], []
for i in range(len(data1)//3):
    a.append(data1[i*3])
    b.append(data1[i*3+1])
    c.append(data1[i*3+2])
    # d.append(data1[i*4+3])

def plot(h_list, e_list, x_axis='x', y_axis='y', title='title', markers=["", "", "", 'v', '^', '1', "2", "3", "4"], colors=["red", "blue", "#34bf49"], legends=[]):
    plt.figure()
    fig, ax = plt.subplots()
    for i in range(len(h_list)):
        h, e = h_list[i], e_list[i]
        # if len(h) < 300:
        #     h = h[:len(h)//2]
        #     e = e[:len(e)//2]
        # while len(h) > 50:
        # h = h[::4]
        # e = e[::4]
        plt.plot(h,e,marker=markers[i%len(markers)], linestyle='dashed', linewidth=1, markersize=2)
        # plt.scatter(h, e, marker=markers[i % len(
        #     markers)],color=colors[i % len(colors)], s=10)
    plt.legend(legends)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.title(title)
    # plt.xlim(0, total_time)
    # plt.xlim(1.44, 1.46)
    # plt.ylim(0, 0.1)
    # ax.set_xscale('linear')
    # ax.set_yscale('log')
    plt.savefig(title + '.png', dpi=100)
    plt.close()

print(b)
plot([b],[a])
# plot([c],[b])
# plot([c],[d])
