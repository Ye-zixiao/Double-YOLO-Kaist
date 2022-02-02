from build_utils.utils import xywh2xyxy
from cv2 import cv2

import numpy as np


def random_cut_box(w: int, h: int, n: int = 10, step: int = 32):
    '''
    随机的在图像中生成一个裁剪区域
    :param w: 所处理图像的宽度
    :param h: 所处理图像的高度
    :param n: 需要在图像中产生多少个裁剪区域
    :param step: 裁剪区域的大小（宽度==高度）
    :return:
    '''

    # 计算裁剪区域的xmin和ymin
    rand_box_xymin = np.random.rand(n, 2)
    rand_box_xymin *= np.array([w - 1, h - 1], dtype=np.float64).transpose()
    # 计算裁剪区域的xmax和ymax
    # TODO: 我不是确定图片会不会算上这个右下边界的像素点！
    rand_box_xmax = np.clip(rand_box_xymin[:, 0] + step - 1, 0, w - 1)
    rand_box_ymax = np.clip(rand_box_xymin[:, 1] + step - 1, 0, h - 1)
    rand_box_xymax = np.stack((rand_box_xmax, rand_box_ymax), 1)
    return np.floor(np.concatenate((rand_box_xymin, rand_box_xymax), 1))


def cover_iou(box1: np.ndarray, box2: np.ndarray):
    '''
    用于计算出裁剪区域与标注边界框之间的重叠面积相对于标注边界框的比例
    :param box1: labels [N, 4]
    :param box2: rand_box [M, 4]
    :return: [N, M]
    '''

    def box_area(box):
        return np.maximum((box[2] - box[0]) * (box[3] - box[1]),
                          np.ones(np.shape(box)[1]))

    label_area = box_area(box1.transpose())
    inter_area = np.prod(np.clip(np.minimum(box1[:, None, 2:], box2[:, 2:]) -  # clip用来避免重叠区域面积出现小于0的现象
                                 np.maximum(box1[:, None, :2], box2[:, :2]), 0, np.shape(box2)[0]), axis=2)
    return inter_area / label_area[:, None]


def clahe_image(v_img, l_img, clip_limit=1.0, tile_grid_size=(4, 4)):
    '''对可见光图像和红外光图像执行带对比度限制的直方图均衡CLAHE'''
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    v_img = cv2.merge([clahe.apply(x) for x in cv2.split(v_img)])
    l_img = cv2.merge([clahe.apply(x) for x in cv2.split(l_img)])
    return v_img, l_img


def snowflake_cutout(v_img: np.ndarray,
                     l_img: np.ndarray,
                     labels: np.ndarray,  # 标注边界框信息，数据结构一般传入的List[np.array]
                     xywh: bool = True,  # 标注信息是否以xywh的方式给出，否则就是以xyxy的方式给出
                     clahe: bool = True,  # 是否开启自适应直方图均衡处理
                     label_thr: float = 0.35,  # 裁剪标注边界框的重叠比阈值
                     n_thr: int = 6,  # 开启雪花cutout处理的阈值，图中有超过这个数的标注边界框就执行处理
                     step: int = 16,  # 裁剪区域大小
                     n_snow: int = 25,  # 裁剪区域数量
                     clip_limit=1.0,  # CLAHE对比度限制
                     grid_size=(4, 4)):  # CLAHE局部直方图均衡网格大小
    '''执行雪花明化变换处理'''

    if clahe:
        # 执行带对比度限制的直方图均衡处理
        v_img, l_img = clahe_image(v_img, l_img, clip_limit, grid_size)

    # 如果图像中的标注边界框足够多，那么我们就假设性认为已经存在一些物体遮挡了，此时我们就不需要做雪花裁剪
    if labels.shape[0] <= n_thr:
        w, h = np.shape(v_img)[:2]
        assert w == np.shape(l_img)[0] and h == np.shape(l_img)[1]
        rand_box = random_cut_box(w, h, n_snow, step)  # 随机生成指定数量的裁剪区域

        if xywh:
            labels_ = xywh2xyxy(labels[:, 1:])  # 将标注边界框信息转换成绝对xyxy形式
            labels_[:, [0, 2]] *= w
            labels_[:, [1, 3]] *= h
        else:
            labels_ = labels[:, 1:]

        # 计算裁剪区域与标注边界框的重叠覆盖比，放弃那些超过阈值的标注边界框
        iou = cover_iou(labels_, rand_box)
        labels = labels[np.sum(iou, axis=1) < label_thr]
        # 裁剪可见光图像和红外光图像中的指定部分区域
        for xmin, ymin, xmax, ymax in rand_box:
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            v_img[ymin:ymax, xmin:xmax] = 0
            l_img[ymin:ymax, xmin:xmax] = 0

    return v_img, l_img, labels
