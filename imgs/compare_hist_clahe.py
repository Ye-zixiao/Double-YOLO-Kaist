from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

j = 0


def compare_hist_clahe(img_path, clipLimit=2.0, tileGridSize=(8, 8)):
    '''将一张图像输入使用直方图均衡和自适应直方图均衡两种算法进行处理，对比结果'''

    # 读取路径指定的图片
    if img_path.endswith("_lwir.jpg"): return

    v_img_path = img_path
    l_img_path = img_path.replace("_visible.jpg", "_lwir.jpg")
    v_img = cv2.imread(v_img_path)
    l_img = cv2.imread(l_img_path)

    b1, g1, r1 = cv2.split(v_img)
    b2, g2, r2 = cv2.split(l_img)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    res11 = cv2.merge([cv2.equalizeHist(x) for x in [b1, g1, r1]])  # 使用普通直方图均衡处理图像
    res12 = cv2.merge([clahe.apply(x) for x in [b1, g1, r1]])
    res21 = cv2.merge([cv2.equalizeHist(x) for x in [b2, g2, r2]])  # 使用自适应直方图均衡处理图像
    res22 = cv2.merge([clahe.apply(x) for x in [b2, g2, r2]])
    img_list = [v_img, res11, res12, l_img, res21, res22]
    titles = ['(a) original visible', '(b) HE visible', '(c) CLAHE visible',
              '(d) original infrared', '(e) HE infrared', '(f) CLAHE infrared']

    # 绘制对比图
    plt.figure(figsize=(11, 6), dpi=150)
    plt.subplots_adjust(left=0, right=1, bottom=0.08, top=1, wspace=0.05)
    for i, im in enumerate(img_list):
        plt.subplot(2, 3, i + 1)
        plt.imshow(im[:, :, ::-1])
        plt.title(titles[i], y=-0.15, fontdict=dict(fontsize=10))
        plt.xticks([])
        plt.yticks([])
    plt.savefig("hist_{}.png".format(j))
    # plt.show()


if __name__ == '__main__':
    root_path = "ori"
    for img_path in os.listdir(root_path):
        compare_hist_clahe(os.path.join(root_path, img_path), 1.0, (4, 4))
        j += 1
    # compare_hist_clahe("imgs/ori/I01206_visible.jpg", 1.0, (4, 4))
    # compare_hist_clahe("imgs/ori/I01206_visible.jpg")
    #
    # compare_hist_clahe("imgs/ori/I01206_lwir.jpg", 1.0, (4, 4))
    # compare_hist_clahe("imgs/ori/I01206_lwir.jpg")
    #
    # compare_hist_clahe("imgs/ori/I00070_visible.jpg", 1.0, (4, 4))
    # compare_hist_clahe("imgs/ori/I00070_visible.jpg")
    #
    # compare_hist_clahe("imgs/ori/I00070_lwir.jpg", 1.0, (4, 4))
    # compare_hist_clahe("imgs/ori/I00070_lwir.jpg")
