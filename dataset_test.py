from build_utils.kaist_dataset import LoadKaistImagesAndLabels
from PIL import Image, ImageDraw

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


def dataset_test():
    random.seed(time.time())
    # 创建数据集对象
    with open("config/hyp.scratch.yaml", "r", encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    # 测试训练集使用四合变换时效果正常，但使用测试集时出现了问题，经过排查应该是rect变换代码致使
    # 标注信息发生了错误的变换，因此暂时决定在验证的过程中不使用rect变换
    train_dataloader = LoadKaistImagesAndLabels(data_txt_path="data/kaist_train_data.txt",
                                                hyp=hyp, single_cls=True, augment=True,
                                                snowflake=True, rect=False)

    for i in range(1):
        # assert False, "下面的代码需要在LoadKaistImagesAndLabels更改返回值类型才能正常运行"
        # 从数据集对象中随机地选取一张可见光图像和对应的红外光图像
        v_img, l_img, labels = train_dataloader.__getitem__(
            random.randint(0, train_dataloader.__len__() - 1))

        # 对可见光图像和红外光图像绘制边界框
        v_img = draw_box(v_img[:, :, ::-1], labels[:, 1:].copy())
        l_img = draw_box(l_img[:, :, ::-1], labels[:, 1:].copy())

        img_list = [v_img, l_img]
        img_name = ['(a) visible', "(b) infrared"]

        # 绘制标注好之后的可见光图像和红外光图像
        plt.figure(figsize=(8, 4), dpi=150)
        plt.subplots_adjust(left=0, right=1, bottom=0.08, top=1, wspace=0.05)
        for j, img in enumerate(img_list):
            plt.subplot(1, 2, j + 1)
            plt.title(img_name[j], y=-0.08, fontdict=dict(fontsize=12))
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")
        plt.show()
        # plt.savefig("mosic_{}.png".format(i))


if __name__ == '__main__':
    dataset_test()
