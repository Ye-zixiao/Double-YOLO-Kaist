import numpy as np
import math

IOU_THRESHOLD = 0.5


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
                            0, 1E5), axis=2)
    return inter / (area1[:, None] + area2 - inter)


def compute_ap_lamr(preds, labels: np.ndarray, shapes: np.ndarray):
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
    # TODO: 在实际代码使用的过程中，下面copy的举措还是不能解决compute_ap_lamr()仅能重复使用一次的问题
    labels = xywh2xyxy_(labels, shapes).copy() # 因为下面的操作会修改labels的class_idx，所以拷贝下

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
    # 计算对数平均漏检率LAMR，该数值反映了算法模型对目标的捕获能力，越低越好
    lamr, fppi, mr = log_average_miss_rate(recall, fp_cumsum, len(labels))

    return {'recall': recall,
            'precision': precision,
            'fppi': fppi,
            'mr': mr,
            'ap': ap,
            'lamr': lamr}
