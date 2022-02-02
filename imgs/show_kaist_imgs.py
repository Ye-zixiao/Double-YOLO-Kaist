from typing import List
from cv2 import cv2

import matplotlib.pyplot as plt
import os


def show_kaist(img_names: List, save=False):
    root_path = "ori"
    visible_img_names = [img_name + "_visible.jpg" for img_name in img_names]
    lwir_img_names = [img_name + "_lwir.jpg" for img_name in img_names]

    visible_imgs = []
    lwir_imgs = []
    for i in range(len(img_names)):
        visible_imgs.append(cv2.imread(os.path.join(root_path, visible_img_names[i])))
        lwir_imgs.append(cv2.imread(os.path.join(root_path, lwir_img_names[i])))

    plt.figure(figsize=(10, 4), dpi=150)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.01, wspace=0.01)
    for i in range(len(img_names)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(visible_imgs[i][:, :, ::-1])
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, 4, i + 5)
        plt.imshow(lwir_imgs[i][:, :, ::-1])
        plt.xticks([])
        plt.yticks([])
    if not save:
        plt.show()
    else:
        plt.savefig("kaist_example.png")


if __name__ == '__main__':
    img_names = ['I00070', 'I00200', 'I00737', 'I01206']
    show_kaist(img_names)
