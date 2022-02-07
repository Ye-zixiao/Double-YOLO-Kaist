from train_utils import get_coco_api_from_dataset, CocoEvaluator
from build_utils.kaist_dataset import LoadKaistImagesAndLabels
from build_utils.parse_config import parse_data_cfg
from torch.utils.data import DataLoader
from build_utils.utils import *
from models import YOLO

import numpy as np
import argparse
import warnings
import tqdm
import json
import math
import yaml
import os

warnings.filterwarnings("ignore")

IOU_THRESHOLD = 0.5


def category_index(json_file_path):
    assert os.path.exists(json_file_path), "class dict json file '{}' not exist!" \
        .format(json_file_path)

    with open(json_file_path, "r") as f:
        class_name_indices = json.load(f)
    category_idx = {v: k for k, v in class_name_indices.items()}
    return category_idx


def voc_ap(recall: np.ndarray, precision: np.ndarray):
    '''
    由于实际的PR散点图常为锯齿形，故采用插值法计算VOC平均准确率AP值，默认是
    工作在IoU阈值0.5的情况下的，如果想修改可以修改最上面的IOU_THRESHOLD
    :param recall: 召回率
    :param precision: 准确率
    :return: 平均准确率
    '''

    # 在前面插入一个(0, 0)点，在后面插入一个(1, 0)点
    mrec = np.concatenate(([0.], recall, [1.]))
    mprec = np.concatenate(([0.], precision, [0.]))

    # 从后往前遍历，使得当前点的准确率precision值为后向最大值，使得其单调递减
    for i in range(mprec.size - 1, 0, -1):
        mprec[i - 1] = np.maximum(mprec[i - 1], mprec[i])
    # 计算出横坐标召回率recall第一个不同值的下标索引
    indices = np.where(mrec[1:] != mrec[:-1])[0] + 1
    # 通过累加矩形面积的方式计算PR曲线下的面积，记为AP值
    ap = np.sum((mrec[indices] - mrec[indices - 1]) * mprec[indices])
    return ap


def log_average_miss_rate(recall: np.ndarray, fp_cumsum: np.ndarray, num_imgs: int):
    '''
    计算对数平均漏检率Log Average Miss Rate
    :param recall: 召回率
    :param fp_cumsum: 假正例FP数累加值
    :param num_imgs: 图像总数
    :return: 对数平均漏检率lamr、假正例数/每图像FPPI和漏检率
    '''

    # 计算FPPI和MR
    fppi = fp_cumsum / float(num_imgs)
    mr = 1 - recall

    # 在FPPI前面插一个特殊值-1.0，在MR前面插一个1.0，避免越界情况发生
    fppi_tmp = np.concatenate(([-1.0], fppi))
    mr_tmp = np.concatenate(([1.0], mr))

    # 在长度为9的等比数列[10^-2.0, ..., 10^0.0]下标位置上取MR，
    # 如果该位置上没有与之对应的FPPI，就取小于它的最大点位置
    refs = np.logspace(-2.0, 0.0, num=9)
    for i, ref_p in enumerate(refs):
        j = np.where(fppi_tmp <= ref_p)[0][-1]
        refs[i] = mr_tmp[j]

    # 计算对数平均漏检率
    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, refs))))

    return lamr, fppi, mr


def box_iou(box1: np.ndarray, box2: np.ndarray):
    '''
    计算两个边界框box矩阵之间的IoU
    :param box1: 一个预测边界框的大小，大小必须为[1, 4]
    :param box2: 该预测边界框所属图像上的所有真实边界框，大小为[n, 4]
    :return: 预测边界框与真实边界框的IoU值矩阵
    '''

    def box_area(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)

    area1 = box_area(box1.transpose())
    area2 = box_area(box2.transpose())
    # TODO：clip用来避免重叠区域面积出现小于0的现象，但必须设置上限，我不知道numpy中还有没有别的方法
    inter = np.prod(np.clip(np.minimum(box1[:, None, 2:], box2[:, 2:])
                            - np.maximum(box1[:, None, :2], box2[:, :2]) + 1,
                            0, 100000), axis=2)
    return inter / (area1[:, None] + area2 - inter)


def get_ap(preds, labels: np.ndarray, shapes: np.ndarray):
    '''
    计算行人目标的平均准确率AP值和对数平均漏检率LAMR值
    :param preds: 类型List[dict]。为预测边界框列表，每一个元素都是一个字典，记录着预测边界框的所属图像编号，
                 预测边界框置信度，预测边界框的xyxy坐标信息
    :param labels: 类型np.ndarray。为真实边界框矩阵，矩阵大小为[img_num x ni x 5]，其中第一维正是img_id，
                 ni表示每张图片中有多少个真实的标注边界框，5表示真实边界框conf+xywh相对坐标的信息
    :return: 返回recall、precision、fppi和mr四个数据
    '''

    def xywh2xyxy_(l, s):
        '''将labels中的xywh相对边界框数据转换成xyxy绝对坐标格式'''
        assert len(l) == len(s), "label's len != shape's len"

        for i in range(len(labels)):
            l[i][:, [1, 3]] *= shapes[i][0]
            l[i][:, [2, 4]] *= shapes[i][1]
            l[i][:, 1] -= l[i][:, 3] / 2
            l[i][:, 2] -= l[i][:, 4] / 2
            l[i][:, 3] = l[i][:, 1] + l[i][:, 3]
            l[i][:, 4] = l[i][:, 2] + l[i][:, 4]
        return l

    def ground_truth_num(l):
        '''计算labels中所有真实标注边界框的总数，因为labels并不是矩阵，故采用此法'''
        n_gt = 0
        for i in range(len(labels)):
            n_gt += l[i].shape[0]
        return n_gt

    # 真实边界框数据，列表的索引即是img_id，每一个元素为大小[n x 5]的numpy矩阵。矩阵
    # 内每一行向量就是一个真实边界框数据，分别为class_idx、xmin、ymin、xmax、ymax。
    # 由于我们只做行人检测，所以类别一定是行人，故将第一个class_idx用来做该真实边界框
    # 是否与某一个预测边界框相映射的标志位，0表示没有与之对应的预测边界框；否则有。
    labels = xywh2xyxy_(labels, shapes)

    nd = len(preds)  # 预测边界框总数
    nt = ground_truth_num(labels)  # 真实边界框总数
    tp = np.zeros((nd,), dtype=np.int32)  # 真正例矩阵
    fp = np.zeros((nd,), dtype=np.int32)  # 假正例矩阵
    for idx, pred in enumerate(preds):
        img_id = pred['img_id']  # 预测边界框所属的图像编号img_id
        bbox = pred['bbox']  # 该预测边界框的xyxy信息
        gt_bboxes = labels[img_id][:, 1:].astype(np.int32)  # 该图像上所有的真实边界框数据
        iou = box_iou(bbox.reshape(-1, 4), gt_bboxes)[0]
        max_iou_idx = np.argmax(iou)

        if iou[max_iou_idx] >= IOU_THRESHOLD:
            if labels[img_id][max_iou_idx][0] == 0:
                # 该预测边界框与对应图像上的真正边界框的IoU>=0.5，且该真实边界框
                # 未与其他预测边界框所映射，那么就判断这个预测边界框是真正例
                labels[img_id][max_iou_idx][0] = 1
                tp[idx] = 1
            else:
                # 发生了多重映射（重复检测）
                fp[idx] = 1
        else:
            fp[idx] = 1

    # 计算FP和TP的累加矩阵
    fp_cumsum = np.cumsum(fp)
    tp_cumsum = np.cumsum(tp)
    # 计算召回率Recall和准确率Precision
    recall = tp_cumsum / nt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    # 上面实际就是在计算不同置信度阈值下的tp_cumsum、fp_cumsum、recall、precision，
    # 从这些数据的变化来反映出网络模型整体的性能水平。
    #           |
    #           | 如果阈值设置在这个位置使得预测边界框做出了如下的分类，使得前面的框被认为是正例，
    #           | 后面的是反例，那么tp有一个，fp也有一个。所以此时的准确率precision=1/2，回归
    #           v 率recall为1/nums_this_class_obj
    # tp: [1, 1, 2, 3, 3]
    # fp: [0, 1, 1, 1, 2]

    # 计算VOC数据集常用的VOC AP指标，该数值反映了算法模型对目标捕获的准确程度，越高越好
    ap = voc_ap(recall, precision)
    print(f"VOC Average Precision (VOC-AP)@[IoU = 0.5] = {(ap * 100):.2f}%")

    # 计算对数平均漏检率LAMR，该数值反映了算法模型对目标的捕获能力，越低越好
    lamr, fppi, mr = log_average_miss_rate(recall, fp_cumsum, len(labels))
    print(f"Log Average Miss Rate (LAMR)@[IoU = 0.5] = {(lamr * 100):.2f}%")

    return {'recall': recall,
            'precision': precision,
            'fppi': fppi,
            'mr': mr,
            'ap': ap,
            'lamr': lamr}


def validate(opt, hyp):
    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    print("Using device '{}' detecting...".format(device))

    batch_size = opt.batch_size
    img_size = opt.img_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(nw))

    # 读取指示文件kaist_data.data，使用测试数据集作为当前性能的度量
    data_dict = parse_data_cfg(opt.data)
    test_path = data_dict[opt.test_type]

    val_dataset = LoadKaistImagesAndLabels(test_path, img_size, batch_size,
                                           hyp=hyp, rect=True, snowflake=False, clahe=opt.clahe)
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
                    boxes = scale_coords(v_imgs[index].shape[1:], boxes, shapes[index][0]).round()

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
    saved_dict = get_ap(preds, val_dataset.labels, val_dataset.shapes)
    if opt.npy_path != '':
        # 根据程序指令给出的路径保存recall-precision、fppi-mr等数据
        np.save(opt.npy_path, saved_dict)

    total_fps = 1. / (total_fps / len(val_dataset.labels))
    print(f"average detecting fps: {total_fps:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('--data', type=str, default='data/kaist_data.data', help='*.data path')
    parser.add_argument('--test-type', type=str, default='test', help="test dataset type")
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N', help='batch size when validation.')
    parser.add_argument('--cfg', type=str, help="*.cfg path",
                        default='config/kaist_dyolov3_cspdarknet_fshare_global_concat_se3.cfg')
    parser.add_argument('--weights', type=str, help='detecting weights',
                        default='results/Double-YOLOv3-CSPDarknet-Fshare-Global-Concat-SE3-102/kaist_dyolov3_cspdarknet_fshare_global_concat_se3_best.pt')
    parser.add_argument('--npy-path', type=str, help="save recall, precision, fppi, mr into this npy file",
                        default='')
    parser.add_argument('--clahe', action='store_true', help="use clahe to process images")
    opt = parser.parse_args()

    with open(opt.hyp, encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    validate(opt, hyp)
