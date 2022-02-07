from build_utils import img_utils, torch_utils, utils
from build_utils.draw_box_utils import draw_box
from build_utils.snowflake import clahe_image
from models import YOLO
from typing import List
from cv2 import cv2

import matplotlib.pyplot as plt
import numpy as np
import argparse
import warnings
import random
import torch
import json
import tqdm
import os

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_images(v_img_path, l_img_path, input_size, device, clahe=False):
    assert os.path.exists(v_img_path), "visible image '{}' not exist.".format(v_img_path)
    assert os.path.exists(l_img_path), "infrared image '{}' not exist.".format(l_img_path)

    # 读取可见光和红外光图像并将其进行一定程度的缩放
    v_img_o = cv2.imread(v_img_path)
    l_img_o = cv2.imread(l_img_path)
    v_img = img_utils.letterbox(v_img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
    l_img = img_utils.letterbox(l_img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]

    if clahe:
        v_img, l_img = clahe_image(v_img, l_img)

    # 将两张图像从BGR-HWC转换成RGB-CHW的组织形式
    v_img = v_img[:, :, ::-1].transpose(2, 0, 1)
    l_img = l_img[:, :, ::-1].transpose(2, 0, 1)
    v_img = np.ascontiguousarray(v_img)
    l_img = np.ascontiguousarray(l_img)

    # 将两张图像转换成Tensor形式并以浮点数记录
    v_img = torch.from_numpy(v_img).to(device).float() / 255.
    l_img = torch.from_numpy(l_img).to(device).float() / 255.
    v_img = v_img.unsqueeze(0)
    l_img = l_img.unsqueeze(0)

    return v_img_o, l_img_o, v_img, l_img


def get_image_paths(img_path):
    img_base_path = img_path.replace('.jpg', '').replace('_visible', '') \
        .replace('_lwir', '')
    v_img_path = img_base_path + "_visible.jpg"
    l_img_path = img_base_path + "_lwir.jpg"
    return v_img_path, l_img_path


def get_base_name(img_path):
    return img_path.split('/')[-1].replace('.jpg', '') \
        .replace('_visible', '').replace('_lwir', '')


def category_index(classes_json_path):
    assert os.path.exists(classes_json_path), "classes json file '{}' not exist.".format(classes_json_path)

    with open(classes_json_path, "r") as f:
        class_dict = json.load(f)
    category_idx = {v: k for k, v in class_dict.items()}
    return category_idx


def load_model(cfg_path, weight_path, img_size, device):
    model = YOLO(cfg=cfg_path, img_size=(img_size, img_size))
    state_dict = torch.load(weight_path, device)['model']
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def detect(img_path_list: List):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Use device '{}' detecting".format(device))

    # 创建预训练的检测模型，生成索引-类别表
    model = load_model(opt.cfg, opt.weight, opt.img_size, device)
    category_idx = category_index(opt.classes_json)

    # 记录性信息
    total_fps = 0.0
    num_miss = 0

    model.eval()
    with torch.no_grad():
        for img_path in tqdm.tqdm(img_path_list, desc="detecting..."):
            # 加载可见光图像、红外光图像，以及对应张量形式的数据
            v_img_o, l_img_o, v_img, l_img = load_images(*get_image_paths(img_path),
                                                         input_size=opt.img_size,
                                                         device=device,
                                                         clahe=opt.clahe)

            # 将图像输入网络进行推理
            t1 = torch_utils.time_synchronized()
            pred = model(v_img, l_img)[0]
            t2 = torch_utils.time_synchronized()
            total_fps += 1 / (t2 - t1)

            # 对预测得到的边界框使用NMS消除那些未达标的边界框，其中预测得到的每个向量内的数据为：
            # x  y  w  h  conf  classes_scores
            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.5, multi_label=True)[0]
            if pred is None:
                num_miss += 1
                continue

            # 将预测的坐标信息转换回原图尺度
            pred[:, :4] = utils.scale_coords(v_img.shape[2:], pred[:, :4], v_img_o.shape).round()
            # 获取预测得到的边界框xywh数据，置信度以及目标类别分数
            bboxes1 = pred[:, :4].detach().cpu().numpy()
            scores1 = pred[:, 4].detach().cpu().numpy()
            classes1 = pred[:, 5].detach().cpu().numpy().astype(np.int32) + 1

            # 在可见光图像中绘制预测得到的边界框，并将BGR-HWC转换成RGB-HWC的数据格式，并最终展示预测结果
            v_img_res = draw_box(v_img_o[:, :, ::-1].copy(), bboxes1, classes1, scores1, category_idx)

            img_list = [v_img_o[:, :, ::-1], l_img_o[:, :, ::-1], v_img_res]
            img_names = ['(a) 可见光图像', '(b) 红外光图像', '(c) {}'.format(opt.model_name)]

            plt.figure(figsize=(10, 3), dpi=200)
            plt.subplots_adjust(left=0, right=1, bottom=0.05, top=0.99, wspace=0.03)
            for i, img in enumerate(img_list):
                plt.subplot(1, 3, i + 1)
                plt.imshow(img)
                plt.title(img_names[i], fontdict={'weight': 'normal', 'size': 9}, y=-0.1)
                plt.xticks([])
                plt.yticks([])

            save_dir = opt.save
            if save_dir == '':
                plt.show()
            else:
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                plt.savefig(os.path.join(save_dir, get_base_name(img_path) + ".jpg"))

        average_fps = total_fps / len(img_path_list)
        miss_rate = num_miss / len(img_path_list) * 100
        print(f"average fps: {average_fps:0.3f}\nmiss rate: {miss_rate:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-name', type=str, default='Double-YOLOv3-CSPDarknet-Fshare-Global-CSE',
                        help='detect model name')
    parser.add_argument('--src', type=str, default='imgs/ori/I00200_lwir.jpg', help='detect image path or name')
    parser.add_argument('--save', type=str,
                        default='results/Double-YOLOv3-CSPDarknet-Fshare-Global-CSE3-Snow-102/imgs',
                        help='result saved dir')
    parser.add_argument('--cfg', type=str, default='config/kaist_dyolov3_cspdarknet_fshare_global_concat_se3.cfg',
                        help='model config file path')
    parser.add_argument('--weight', type=str,
                        default='results/Double-YOLOv3-CSPDarknet-Fshare-Global-CSE3-Snow-102/kaist_dyolov3_cspdarknet_snowflake_best.pt',
                        help='initial weights path')
    parser.add_argument('--clahe', action='store_true', help="use clahe to process images")
    parser.add_argument('--classes-json', type=str, default='data/kaist_voc_classes.json',
                        help='classes json file path')
    parser.add_argument('--img-size', type=int, default=512, help='detect image size')
    opt = parser.parse_args()

    root = "imgs/ori/"
    # root = "Kaist_YOLO/test/images/"
    img_path_list = [os.path.join(root, x) for x in os.listdir(root) if x.endswith('_visible.jpg')]
    random.shuffle(img_path_list)
    detect(img_path_list[:40])
