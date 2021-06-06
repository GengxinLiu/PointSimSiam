import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, "=" * rate_num,
                                    " " * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


def accuracy(ypred: np.ndarray, ytrue: np.ndarray):
    """
    Compute accuracy
    :param ypred: (n, )
    :param ytrue: (n, )
    """
    correct_prediction = np.equal(ypred, ytrue)
    return np.mean(correct_prediction)


def get_cmap(n, name='jet'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.
    """
    return plt.cm.get_cmap(name, n)


def savePointCloudColorSeg(points, label, filename='tmp.ply'):
    num = len(np.unique(label))
    f = open(filename, "w")

    f.write("ply\n")
    f.write("format ascii 1.0\n")
    f.write("element vertex {}\n".format(len(points)))
    f.write("property float x\n")
    f.write("property float y\n")
    f.write("property float z\n")
    f.write("property uchar red\n")
    f.write("property uchar green\n")
    f.write("property uchar blue\n")
    f.write("end_header\n")

    cmap = get_cmap(num)
    for i in range(np.max(label) + 1):
        index = np.where(label == i)
        x = points[index, 0]
        y = points[index, 1]
        z = points[index, 2]
        p_num = x.shape[1]

        for j in range(p_num):
            f.write(str(x[0, j]) + " " + str(y[0, j]) + " " + str(z[0, j]) + " " + str(int(
                255 * cmap(i)[0])) + " " + str(int(255 * cmap(i)[1])) + " " + str(int(255 * cmap(i)[2])) + "\n")
    f.close()


def get_timestamp():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


def show_acc_loss(txt_path, save_path, title=None):
    f = open(txt_path, 'r')
    epoch, loss, acc = [], [], []
    for line in f.readlines():
        words = line.split()
        epoch.append(int(words[1]))
        loss.append(float(words[7]))
        acc.append(float(words[-1]))
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(epoch, loss, 'r', label="loss")
    ax1.set_ylabel('loss')
    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(epoch, acc, 'g', label="acc")
    fig.legend()
    ax2.set_ylabel('classification accuracy')
    ax2.set_xlabel('epoch')
    if title is not None:
        plt.title(title)
    plt.savefig(save_path)
    del fig


if __name__ == '__main__':
    show_acc_loss('log_simsiam/2021_01_23_18_02_47/train.txt', 'log_simsiam/2021_01_23_18_02_47/loss_acc.png',
                  title='log_simsiam/2021_01_23_18_02_47')
