from train_utils import get_coco_api_from_dataset, CocoEvaluator
from build_utils.kaist_dataset import LoadKaistImagesAndLabels
from build_utils.parse_config import parse_data_cfg
from build_utils.torch_utils import select_device
from torch.utils.data import DataLoader
from build_utils.utils import *
from other_utils.metrics import compute_ap_lamr
from models import YOLO

import numpy as np
import argparse
import warnings
import tqdm
import json
import yaml
import os

warnings.filterwarnings("ignore")


def category_index(json_file_path):
    assert os.path.exists(json_file_path), "class dict json file '{}' not exist!" \
        .format(json_file_path)

    with open(json_file_path, "r") as f:
        class_name_indices = json.load(f)
    category_idx = {v: k for k, v in class_name_indices.items()}
    return category_idx


def evaluate(opt, hyp):
    device = select_device(opt.device)

    batch_size = opt.batch_size
    img_size = opt.img_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))

    # 读取指示文件kaist_data.data，使用测试数据集作为当前性能的度量
    data_dict = parse_data_cfg(opt.data)
    test_path = data_dict[opt.test_type]

    val_dataset = LoadKaistImagesAndLabels(test_path, img_size, batch_size,
                                           hyp=hyp, rect=True, snowflake=False)
    val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=False, num_workers=nw, pin_memory=True,
                                    collate_fn=val_dataset.collate_fn)

    # 创建网络模型，并使用预训练权重初始化之
    model = YOLO(opt.cfg, img_size)
    model.load_state_dict(torch.load(opt.weights, map_location=device)['model'])
    model.to(device)

    # 创建Coco类对象和CocoEvaluator评估器对象
    coco = get_coco_api_from_dataset(val_dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    cpu_device = torch.device("cpu")
    total_fps = 0.0
    preds = []

    model.eval()
    with torch.no_grad():
        for v_imgs, l_imgs, targets, paths, shapes, img_index \
                in tqdm.tqdm(val_dataset_loader, desc="validation..."):
            # uint8 to float32, 0 - 255 to 0.0 - 1.0
            v_imgs = v_imgs.to(device).float() / 255.0
            l_imgs = l_imgs.to(device).float() / 255.0

            t1 = torch_utils.time_synchronized()
            pred = model(v_imgs, l_imgs)[0]  # only get inference result
            t2 = torch_utils.time_synchronized()
            pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)
            total_fps += t2 - t1

            outputs = []
            for index, p in enumerate(pred):
                if p is None:
                    p = torch.empty((0, 6), device=cpu_device)
                    boxes = torch.empty((0, 4), device=cpu_device)
                else:
                    # xmin, ymin, xmax, ymax
                    boxes = p[:, :4]
                    # shapes: (h0, w0), ((h / h0, w / w0), pad)
                    # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                    boxes = scale_coords(v_imgs[index].shape[1:], boxes, shapes[index][0], shapes[index][1])
                    # boxes = scale_coords(v_imgs[index].shape[1:], boxes, shapes[index][0])

                # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
                info = {"boxes": boxes.to(cpu_device),
                        "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                        "scores": p[:, 4].to(cpu_device)}
                outputs.append(info)

            # 计算COCO性能评估器所需要的字典数据
            res = {img_id: output for img_id, output in zip(img_index, outputs)}
            coco_evaluator.update(res)

            # 计算get_ap()函数所需要的预测边界框字典数据，并将其加入到列表preds中
            for img_id, output in zip(img_index, outputs):
                for i in range(output['labels'].shape[0]):
                    info = dict()
                    info['img_id'] = img_id
                    info['conf'] = output['scores'][i].item()
                    info['bbox'] = output['boxes'][i].numpy()
                    preds.append(info)

    # 使用COCO性能评估器计算当前网络模型的性能指标
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # 使用get_ap()计算VOC数据集AP性能指标和行人检测模型常用的FPPI-MR性能指标
    preds.sort(key=lambda x: float(x['conf']), reverse=True)
    saved_dict = compute_ap_lamr(preds, val_dataset.labels, val_dataset.shapes)
    print(f"VOC Average Precision (VOC-AP)@[IoU = 0.5] = {(saved_dict['ap'] * 100):.2f}%")
    print(f"Log Average Miss Rate (LAMR)@[IoU = 0.5] = {(saved_dict['lamr'] * 100):.2f}%")
    # 根据程序指令给出的路径保存recall-precision、fppi-mr等数据
    if opt.npy_path != '':
        np.save(opt.npy_path, saved_dict)

    total_fps = 1. / (total_fps / len(val_dataset.labels))
    print(f"average detecting fps: {total_fps:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data', type=str, default='data/kaist_data.data', help='*.data path')
    parser.add_argument('--test-type', type=str, default='test', help="test dataset type")
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.4.yaml', help='hyperparameters path')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='batch size when validation.')
    parser.add_argument('--cfg', type=str, default='config/kaist_dyolov4_mobilenetv2_fshare_global_cse3.cfg', help="*.cfg path")
    parser.add_argument('--weights', type=str,
                        default='results/Double-YOLOv4-MNv2-Fshare-Global-CSE3-102/kaist_dyolov4_mobilenetv2_fshare_global_cse3_best.pt',
                        help='detecting weights')
    parser.add_argument('--npy-path', type=str, help="save recall, precision, fppi, mr into this npy file",
                        default='')
    opt = parser.parse_args()

    with open(opt.hyp, encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(opt, hyp)
