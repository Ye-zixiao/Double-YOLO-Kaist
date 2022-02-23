from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from typing import List

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def draw_pr_fm(npy_path_list: List, model_name_list: List, save_path=''):
    '''
    将各个模型得到recall-precision、fppi-mr曲线聚合绘制到两个子图中
    :param npy_path_list: 各个模型测试过程中产生的npy文件路径列表
    :param model_name_list: 各个模型对应的名字
    :param save_path: 若非空则指向当前图像的保存路径
    :return:
    '''

    assert len(npy_path_list) == len(model_name_list)
    pr_model_names = model_name_list.copy()
    fm_model_names = model_name_list.copy()

    plt.figure(figsize=(13, 5), dpi=300)
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.98, wspace=0.15, hspace=0)
    for i, (npy_path, _) in enumerate(zip(npy_path_list, model_name_list)):
        # 从各个模型对应的npy文件中取出recall、precision、fppi和mr数据
        ld: dict = np.load(npy_path, allow_pickle=True).item()
        rec = ld['recall']
        prec = ld['precision']
        fppi = ld['fppi']
        lamr = ld['lamr']
        mr = ld['mr']
        ap = ld['ap']

        # 为各个模型的名字中添加AP和LAMR指标
        pr_model_names[i] += f" ({(ap * 100):.1f}%)"
        fm_model_names[i] += f" ({(lamr * 100):.1f}%)"
        # 绘制PR曲线
        plt.subplot(1, 2, 1)
        plt.plot(rec, prec)
        # 绘制FPPI-MR曲线
        plt.subplot(1, 2, 2)
        plt.plot(fppi, mr)

    # 设置子图图例、标题、坐标轴标签、网格等各项参数
    plt.subplot(1, 2, 1)
    plt.legend(pr_model_names, frameon=True, loc='lower left')
    plt.title("(a) P-R 曲线", y=-0.18)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(ls='--')
    plt.subplot(1, 2, 2)
    plt.legend(fm_model_names, frameon=True, loc='lower left')
    plt.title("(b) FPPI-MR 曲线", y=-0.18)
    plt.xlabel("False Positives Per Image")
    plt.ylabel("Miss Rate")
    # 将FPPI-MR图绘制成log-log坐标格式的图像
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    # 使FPPI-MR图的坐标轴以0.1为主刻度距离，并限制y轴刻度范围
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.ylim(0.1, 1)
    plt.grid(ls='--')

    # 保存或者展示P-R/FPPI-MR生成图
    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path)


def draw_pr(npy_path_list: List, model_name_list: List, save_path=''):
    '''
    绘制各个模型得到recall-precision曲线
    :param npy_path_list: 各个模型测试过程中产生的npy文件路径列表
    :param model_name_list: 各个模型对应的名字
    :param save_path: 若非空则指向当前图像的保存路径
    :return:
    '''

    assert len(npy_path_list) == len(model_name_list)
    pr_model_names = model_name_list.copy()

    plt.figure(figsize=(7, 6), dpi=300)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.15, hspace=0)
    for i, (npy_path, _) in enumerate(zip(npy_path_list, model_name_list)):
        # 从各个模型对应的npy文件中取出recall、precision、fppi和mr数据
        ld: dict = np.load(npy_path, allow_pickle=True).item()
        rec = ld['recall']
        prec = ld['precision']
        ap = ld['ap']

        pr_model_names[i] += f" ({(ap * 100):.1f}%)"
        plt.plot(rec, prec)

    plt.legend(pr_model_names, frameon=True, loc='lower left')
    # plt.title("P-R 曲线", y=-0.18)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(ls='--')
    if save_path == "":
        plt.show()
    else:
        plt.savefig(save_path)


def draw_fppi_mr(npy_path_list: List, model_name_list: List, save_path=''):
    '''
    绘制各个模型得到的fppi-mr曲线
    :param npy_path_list: 各个模型测试过程中产生的npy文件路径列表
    :param model_name_list: 各个模型对应的名字
    :param save_path: 若非空则指向当前图像的保存路径
    :return:
    '''

    assert len(npy_path_list) == len(model_name_list)
    fm_model_names = model_name_list.copy()

    plt.figure(figsize=(7, 6), dpi=300)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95, wspace=0.15, hspace=0)
    for i, (npy_path, _) in enumerate(zip(npy_path_list, model_name_list)):
        # 从各个模型对应的npy文件中取出recall、precision、fppi和mr数据
        ld: dict = np.load(npy_path, allow_pickle=True).item()
        fppi = ld['fppi']
        lamr = ld['lamr']
        mr = ld['mr']

        fm_model_names[i] += f" ({(lamr * 100):.1f}%)"
        # 绘制FPPI-MR曲线
        plt.plot(fppi, mr)

    plt.legend(fm_model_names, frameon=True, loc='lower left')
    # plt.title("(b) FPPI-MR 曲线", y=-0.18)
    plt.xlabel("False Positives Per Image")
    plt.ylabel("Miss Rate")
    # 将FPPI-MR图绘制成log-log坐标格式的图像
    plt.xscale('log')
    plt.yscale('log')
    ax = plt.gca()
    # 使FPPI-MR图的坐标轴以0.1为主刻度距离，并限制y轴刻度范围
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.ylim(0.1, 1)
    plt.grid(ls='--')

    if save_path == '':
        plt.show()
    else:
        plt.savefig(save_path)


if __name__ == '__main__':
    yolov3_npy_paths = [
        "results/Visible-YOLOv3-Normal102/rec-prec.fppi-mr.npy",
        "results/Double-YOLOv3-Add-SL102/rec-prec.fppi-mr.npy",
        "results/Double-YOLOv3-Concat-SE102/rec-prec.fppi-mr.npy",
        # "results/Double-YOLOv3-Fshare-Global-Add-SL102/rec-prec.fppi-mr.npy",
        # "results/Double-YOLOv3-Fshare-Global-Concat-SE102/rec-prec.fppi-mr.npy",
        # "results/Double-YOLOv3-Fshare-Global-Concat-SE3-102/rec-prec.fppi-mr.npy",
    ]
    yolov3_model_names = [
        "Visible-YOLOv3",
        "Double-YOLOv3-ASL",
        "Double-YOLOv3-CSE",
        # "Double-YOLOv3-FSHASL",
        # "Double-YOLOv3-FSHCSE5",
        # "Double-YOLOv3_FSHCSE"
    ]
    draw_pr(yolov3_npy_paths, yolov3_model_names, save_path="")
    draw_fppi_mr(yolov3_npy_paths, yolov3_model_names, save_path="")

    yolov4_npy_paths = [
        "results/Visible-YOLOv4-Normal102/rec-prec.fppi-mr.pny.npy",
        # "results/Visible-YOLOv4-MNv2-102/rec-prec.fppi-mr.npy",
        # "results/Visible-YOLOv4-MNv3-102/rec-prec.fppi-mr.npy",
        "results/Double-YOLOv4-Concat-SE102/rec-prec.fppi-mr.npy",
        "results/Double-YOLOv4-Add-SL102/rec-prec.fppi-mr.npy",
        # "results/Double-YOLOv4-Fshare-Global-Concat-SE3v-102/rec-prec.fppi-mr.npy",
        # "results/Double-YOLOv4-MNv2-Fshare-Global-CSE3-102/rec-prec.fppi-mr.npy",
        # "results/Double-YOLOv4-MNv3-Fshare-Global-CSE3-102/rec-prec.fppi-mr.npy"
    ]
    yolov4_model_names = [
        "Visible-YOLOv4",
        # "V-YOLOv4-MN2",
        # "V-YOLOv4-MN3",
        "Double-YOLOv4-ASL",
        "Double-YOLOv4-CSE",
        # "Double-YOLOv4-FSHCSE",
        # "D-YOLOv4-MNv2-FSHCSE",
        # "D-YOLOv4-MNv3-FSHCSE"
    ]
    draw_pr(yolov4_npy_paths, yolov4_model_names, save_path="")
    draw_fppi_mr(yolov4_npy_paths, yolov4_model_names, save_path="")

    draw_pr(yolov3_npy_paths + yolov4_npy_paths, yolov3_model_names + yolov4_model_names,
            save_path="docs/yolov3-4.pr6.png")
    draw_fppi_mr(yolov3_npy_paths + yolov4_npy_paths, yolov3_model_names + yolov4_model_names,
                 save_path="docs/yolov3-4.fm6.png")
