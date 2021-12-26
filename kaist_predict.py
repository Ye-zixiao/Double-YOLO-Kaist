from build_utils import img_utils, torch_utils, utils
from matplotlib import pyplot as plt
from draw_box_utils import draw_box
from models import YOLOv3
from cv2 import cv2

import numpy as np
import torch
import json
import time
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

normal_cfg_path = "config/kaist_yolov3.cfg"
double_cfg_path = "config/kaist_dyolov3_1.cfg"
json_path = "data/kaist_voc_classes.json"
normal_weight_path = "results/experiment 2/kaist_yolov3_best.pt"
double_weight_path = "results/experiment 3/kaist_dyolov3_best.pt"
assert os.path.exists(normal_cfg_path), "normal cfg file {} does not exist".format(normal_cfg_path)
assert os.path.exists(double_cfg_path), "double cfg file {} does not exist".format(double_cfg_path)
assert os.path.exists(normal_weight_path), "normal weights file {} does not exist".format(normal_weight_path)
assert os.path.exists(double_weight_path), "double weights file {} does not exist".format(double_weight_path)
assert os.path.exists(json_path), "json file {} does not exist".format(json_path)

img_size = 512
input_size = (img_size, img_size)

# 加载记录类别编号信息的json文件
json_file = open(json_path, "r")
class_dict = json.load(json_file)
category_index = {v: k for k, v in class_dict.items()}
json_file.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using {} device predicting".format(device))

# 创建网络模型并使用训练得到的权重参数初始化之
normal_model = YOLOv3(normal_cfg_path, input_size)
normal_state_dict = torch.load(normal_weight_path, device)['model']
normal_model.load_state_dict(normal_state_dict)
normal_model.to(device)

double_model = YOLOv3(double_cfg_path, input_size)
double_state_dict = torch.load(double_weight_path, device)['model']
double_model.load_state_dict(double_state_dict)
double_model.to(device)


def load_image(img_path: str, input_size, device):
    # 获取可见光和红外光图像的路径
    if img_path.endswith(".jpg"):
        v_img_path = img_path.replace(".jpg", "_visible.jpg")
        l_img_path = img_path.replace(".jpg", "_lwir.jpg")
    else:
        v_img_path = img_path + "_visible.jpg"
        l_img_path = img_path + "_lwir.jpg"
    assert os.path.exists(v_img_path), "visible image {} does not exist".format(v_img_path)
    assert os.path.exists(l_img_path), "lwir image {} does not exist".format(l_img_path)

    # 读取可见光和红外光图像并将其进行一定程度的缩放
    v_img_o = cv2.imread(v_img_path)
    l_img_o = cv2.imread(l_img_path)
    v_img = img_utils.letterbox(v_img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
    l_img = img_utils.letterbox(l_img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]

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


def main(img_path: str, display=True):
    # 网络模型开启验证模式
    normal_model.eval()
    double_model.eval()
    with torch.no_grad():
        # 加载可见光和红外光图像，并返回它们预处理之后的Tensor形式
        v_img_o, l_img_o, v_img, l_img = load_image(img_path, input_size, device)

        # 将图像送入网络模型中预测
        t1 = torch_utils.time_synchronized()
        pred1 = normal_model(v_img, l_img)[0]
        pred2 = double_model(v_img, l_img)[0]
        t2 = torch_utils.time_synchronized()
        print("normal predict time: {}".format(t2 - t1))

        # 对预测得到的边界框使用NMS消除那些未达标的边界框，其中预测得到的每个向量内的数据为：
        # x  y  w  h  conf  classes_scores
        pred1 = utils.non_max_suppression(pred1, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        pred2 = utils.non_max_suppression(pred2, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        print("nms calculate time: {}".format(t3 - t1))

        if pred1 is None or pred2 is None:
            print("No target detected")
            exit(0)

        # 将预测的坐标信息转换回原图尺度
        pred1[:, :4] = utils.scale_coords(v_img.shape[2:], pred1[:, :4], v_img_o.shape).round()
        pred2[:, :4] = utils.scale_coords(v_img.shape[2:], pred2[:, :4], v_img_o.shape).round()
        print(pred1.shape)
        print(pred2.shape)

        # 获取预测得到的边界框xywh数据，置信度以及目标类别分数
        bboxes1 = pred1[:, :4].detach().cpu().numpy()
        scores1 = pred1[:, 4].detach().cpu().numpy()
        classes1 = pred1[:, 5].detach().cpu().numpy().astype(np.int32) + 1
        bboxes2 = pred2[:, :4].detach().cpu().numpy()
        scores2 = pred2[:, 4].detach().cpu().numpy()
        classes2 = pred2[:, 5].detach().cpu().numpy().astype(np.int32) + 1

        # 在可见光图像中绘制预测得到的边界框，并将BGR-HWC转换成RGB-HWC的数据格式，并最终展示预测结果
        v_img_r1 = draw_box(v_img_o[:, :, ::-1].copy(), bboxes1, classes1, scores1, category_index)
        v_img_r2 = draw_box(v_img_o[:, :, ::-1].copy(), bboxes2, classes2, scores2, category_index)
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(v_img_o[:, :, ::-1])  # 将可见光原图从BGR-HWC转换成RGB-HWC
        plt.subplot(2, 2, 2)
        plt.imshow(l_img_o[:, :, ::-1])
        plt.subplot(2, 2, 3)
        plt.imshow(v_img_r1)
        plt.subplot(2, 2, 4)
        plt.imshow(v_img_r2)
        if display:
            plt.show()
        else:
            plt.savefig(os.path.join("imgs/res", img_path.split('/')[-1]),
                        bbox_inches='tight')


if __name__ == '__main__':
    main("imgs/I00200.jpg", False)
    main("imgs/I00304.jpg", False)
    main("imgs/I00933.jpg", False)
    main("imgs/I00972.jpg", False)
    main("imgs/I01020.jpg", False)
    main("imgs/I01039.jpg", False)
    main("imgs/I01206.jpg", False)
