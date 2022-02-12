from models import YOLO
import matplotlib.pyplot as plt
from numpy import double
import numpy as np
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

STANDARD_COLORS = [
    'Blue', 'Red', 'Green', 'AliceBlue', 'Yellow', 'YellowGreen'
]


def draw_map_loss(filename):
    '''绘制训练过程中对验证集的map、ap和学习率变化过程'''

    with open(filename, "r") as f:
        res = f.read().splitlines()
        map50 = [double(ln.split()[1]) for ln in res]
        ap = [double(ln.split()[2]) for ln in res]
        loss = [double(ln.split()[-2]) for ln in res]
        lr = [double(ln.split()[-1]) for ln in res]

    x_axis = np.linspace(0, len(res), len(res))
    plt.subplot(2, 2, 1)
    plt.plot(x_axis, map50)
    plt.plot(x_axis, ap)
    plt.grid(ls='--')
    plt.subplot(2, 2, 2)
    plt.plot(x_axis, loss)
    plt.grid(ls='--')
    plt.subplot(2, 2, 3)
    plt.plot(x_axis, lr)
    plt.grid(ls='--')
    plt.show()


def draw_map_loss_ls(filenames, model_names):
    plt.figure(figsize=(10, 6), dpi=100)
    for i, fn in enumerate(filenames):
        with open(fn, "r") as f:
            res = f.read().splitlines()
            map50 = [double(ln.split()[1]) for ln in res]
            ap = [double(ln.split()[2]) for ln in res]
            loss = [double(ln.split()[-2]) for ln in res]
            lr = [double(ln.split()[-1]) for ln in res]

        x_axis = np.linspace(0, len(res), len(res))
        plt.subplot(2, 2, 1)
        plt.plot(x_axis, map50, color=STANDARD_COLORS[i])
        plt.plot(x_axis, ap, color=STANDARD_COLORS[i])
        plt.grid(ls='--')
        # plt.legend(model_names, fontsize=8, loc='lower right')
        plt.subplot(2, 2, 2)
        plt.plot(x_axis, loss, color=STANDARD_COLORS[i])
        # plt.legend(model_names, fontsize=8, loc='upper right')
        plt.grid(ls='--')
        plt.subplot(2, 2, 3)
        plt.plot(x_axis, lr)
        plt.grid(ls='--')
    plt.show()


if __name__ == '__main__':
    draw_map_loss("results/Double-YOLOv4-Fshare-Global-Concat-SE3v-102/results20220211-143545.txt")
