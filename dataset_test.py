from build_utils.kaist_dataset import LoadKaistImagesAndLabels
from PIL import Image, ImageDraw
from cv2 import cv2

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import yaml
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def draw_box(image, labels):
    image_height, image_weight = image.shape[:2]
    labels_ = labels.copy()

    # 使标注中的相对坐标信息转换为绝对坐标信息
    labels_[:, [0, 2]] *= image_weight
    labels_[:, [1, 3]] *= image_height

    # 分别计算xmin、ymin、xmax、ymax
    boxes = labels.copy()
    boxes[:, 0] = labels_[:, 0] - labels_[:, 2] / 2
    boxes[:, 1] = labels_[:, 1] - labels_[:, 3] / 2
    boxes[:, 2] = labels_[:, 0] + labels_[:, 2] / 2
    boxes[:, 3] = labels_[:, 1] + labels_[:, 3] / 2

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for i in range(boxes.shape[0]):
        xmin, ymin, xmax, ymax = tuple(boxes[i].tolist())
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        # 在原图上绘制边界框
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=1, fill="Red")
    return image  # 返回绘制好之后的图像


def test_dataset():
    random.seed(time.time())
    # 创建数据集对象
    with open("config/hyp.yaml", "r") as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    # 测试训练集使用四合变换时效果正常，但使用测试集时出现了问题，经过排查应该是rect变换代码致使
    # 标注信息发生了错误的变换，因此暂时决定在验证的过程中不使用rect变换
    train_dataloader = LoadKaistImagesAndLabels(data_txt_path="data/kaist_val_data.txt",
                                                hyp=hyp, single_cls=True, augment=False,
                                                rect=False)

    for i in range(5):
        # assert False, "下面的代码需要在LoadKaistImagesAndLabels更改返回值类型才能正常运行"
        # 从数据集对象中随机地选取一张可见光图像和对应的红外光图像
        v_img, l_img, labels = train_dataloader.__getitem__(
            random.randint(0, train_dataloader.__len__() - 1))

        # 对可见光图像和红外光图像绘制边界框
        v_img = draw_box(v_img[:, :, ::-1], labels[:, 1:].copy())
        l_img = draw_box(l_img[:, :, ::-1], labels[:, 1:].copy())

        # 绘制标注好之后的可见光图像和红外光图像
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(v_img)
        plt.subplot(1, 2, 2)
        plt.imshow(l_img)
        plt.show()


if __name__ == '__main__':
    test_dataset()
