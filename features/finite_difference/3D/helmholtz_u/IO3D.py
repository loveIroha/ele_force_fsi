import os


def check_directory_exists(file_path):
    # file_path为文件路径，包含文件名
    # 获取文件所在的目录
    directory = os.path.dirname(file_path)
    if directory == '':
        return
    # 检查目录是否存在，如果不存在则创建它
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在。")
        try:
            os.makedirs(directory)
            print(f"已创建目录：'{directory}'")
        except OSError as e:
            print(f"创建目录时发生错误：{str(e)}")
    else:
        print(f"目录 '{directory}' 已存在。")


def write_data(data, file_path, fun):
    check_directory_exists(file_path)
    # 打开文件
    try:
        with open(file_path, "w") as file:
            # 读取并打印文件内容
            # print(f"将数据写入文件 '{file_path}' ...")
            fun(data, file)
    except FileNotFoundError:
        print(f"不能打开 '{file_path}' 未找到。")
    except Exception as e:
        print(f"发生了错误：{str(e)}")


def write_vector_3D(data, file_path):
    def fun(data, file):
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                for k in range(data.shape[2]):
                    print(f"{data[i, j, k]:.10f}", end=" ", file=file)
                print(file=file)
            print(file=file)

    write_data(data, file_path, fun)



# def read_vector_3D(file_path):
#     def fun(file):
#         data = []
#         for line in file:
#             data.append([float(x) for x in line.split()])
#         return np.array(data)
#     return read_data(file_path, fun)

# write_vector_3D(p, "data1/p.txt")
# print(p)
# p = read_vector_3D('data1/p.txt')


# import numpy as np

# file_path = 'data1/p.txt'
# try:
#     with open(file_path, 'r') as file:
#         # 读取文件的每一行并将其分割成浮点数列表
#         lines = file.readlines()
#         print(lines[0])
#         data = []
#         data_1 = []
#         for i in range(len(lines)):
#             if lines[i][0] == '\n':
#                 data.append(data_1)
#                 data_1 = []
#                 continue
#             lines[i] = [float(x) for x in lines[i].split()]
#             data_1.append(lines[i])
#             # print(data_1)
#         # data = []
#         # for line in lines:
#         #     values = [float(x) for x in line.split()]
#         #     data.append(values)
#         # # 将列表转换为NumPy数组
#         numpy_array = np.array(data)
#         # 打印NumPy数组
#         print(numpy_array)
# except FileNotFoundError:
#     print(f"文件 '{file_path}' 未找到。")
# except Exception as e:
#     print(f"发生了错误：{str(e)}")