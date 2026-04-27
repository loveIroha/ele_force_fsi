import matplotlib.pyplot as plt

# plot_multiple_lines(time, data, labels,types, title='porous pressure', xlabel='time step', ylabel='p')
def plot_multiple_lines(time, data, labels=None, types=None, colors=None, linestyles=None, title=None, xlabel=None, ylabel=None, xlim=None, ylim=None):
    """
    Plot multiple lines from a list of lists.
    
    Parameters:
        data (list of lists): List of data points for each line.
        labels (list): List of labels for each line. Default is None.
        title (str): Title of the plot. Default is None.
        xlabel (str): Label for the x-axis. Default is None.
        ylabel (str): Label for the y-axis. Default is None.
    """
    basic_colors = ["#1663A9","#614099","#B9181A","#369F2D","#FC8002"]
    basic_linestyles = ["-","--",":","-.","-","-","-"]

    if not labels:
        labels = [f'Line {i+1}' for i in range(len(data))]

    if not types:
        types = ['lines' for i in range(len(data))]
    
    if not linestyles:
        linestyles = [basic_linestyles[i] for i in range(len(data))]
    
    if not colors:
        colors = [basic_colors[i%len(basic_colors)] for i in range(len(data))]

    if not title:
        title = "plot"
    
    # TODO: labels, types, 的列表长度要和data一样长
    for i in range(len(data)):
        if types[i] == 'lines':
            plt.plot(time[i], data[i], color=colors[i], linestyle = linestyles[i], label=labels[i])
        elif types[i] == 'dots':
            plt.scatter(time[i], data[i], color=colors[i], label=labels[i])
        else:
            print("画图的 types 参数错误")

    plt.legend()
    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # plt.ylim(0, 1000)
    plt.savefig(title+".png")
    plt.close()