from PIL import Image, ExifTags
from lxml import etree
from tqdm import tqdm

import numpy as np
import random
import os

# 找到图像exif信息中对应旋转信息的key
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == "Orientation":
        break


def exif_size(img):
    '''
    获取由Image.open()方法打开的图像的大小
    :param img: PIL图像
    :return: 图像宽高
    '''
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:
            s = s(s[1], s[0])
    except:
        pass
    return s


class VOCDatasetLabels(object):
    '''负责从指定的数据集中读取所有图像的大小、标注边界框大小信息'''

    def __init__(self, root_path, txt_name="test.txt"):
        # 检查指向的VOC数据集是否存在
        assert os.path.exists(root_path), "VOC Dataset not exist at {}".format(root_path)

        # 获取标注xml文件所在目录以及记录训练集图片文件名的txt文件所在路径
        self.annotations_root_path = os.path.join(root_path, "Annotations")
        txt_file_path = os.path.join(root_path, "ImageSets", "Main", txt_name)
        assert os.path.exists(txt_file_path), "not found txt file {}".format(txt_name)

        # 读取txt文件获取所有相关图片文件对应的标注xml文件路径
        with open(txt_file_path, 'r') as f:
            self.xml_list = [os.path.join(self.annotations_root_path, line.strip() + ".xml")
                             for line in f.readlines() if len(line.strip()) > 0]

        # 确定xml_list记录的标注xml文件是否存在
        assert len(self.xml_list) > 0, "in '{}' file does not find any information".format(txt_name)
        for xml_path in self.xml_list:
            assert os.path.exists(xml_path), "not found '{}' file".format(xml_path)

    def __len__(self):
        return len(self.xml_list)

    def parse_xml_to_dict(self, xml):
        '''
        将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
        :param xml: xml tree obtained by parsing XML file contents using lxml.etree
        :return: Python dictionary holding XML contents
        '''

        if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
            return {xml.tag: xml.text}

        result = {}
        for child in xml:
            child_result = self.parse_xml_to_dict(child)  # 递归遍历标签信息
            if child.tag != 'object':
                result[child.tag] = child_result[child.tag]
            else:
                if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                    result[child.tag] = []
                result[child.tag].append(child_result[child.tag])
        return {xml.tag: result}

    def get_info(self):
        '''获取所有xml文件记录的图片文件高宽信息，以及图片中的标注边界框高宽信息'''
        im_wh_list = []
        boxes_wh_list = []
        for xml_path in tqdm(self.xml_list, desc="read voc wh info."):
            # read xml
            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = self.parse_xml_to_dict(xml)["annotation"]
            im_height = int(data["size"]["height"])
            im_width = int(data["size"]["width"])

            wh = []
            for obj in data["object"]:
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                wh.append([(xmax - xmin) / im_width, (ymax - ymin) / im_height])

            if len(wh) == 0:
                continue

            im_wh_list.append([im_width, im_height])
            boxes_wh_list.append(wh)

        return im_wh_list, boxes_wh_list


class YOLODatasetLabels(object):
    def __init__(self, train_data_txt="data/kaist_train_data.txt"):
        try:
            with open(train_data_txt, "r") as f:
                img_files = f.read().splitlines()
                self.img_files = [imf.replace(".jpg", "_visible.jpg") for imf in img_files]
                self.label_files = [imf.replace("images", "labels").replace("jpg", "txt") for imf in img_files]
                n = len(img_files)
        except Exception as e:
            raise FileNotFoundError("Error loading data from '{}': {}".format(train_data_txt, e))

        train_data_shape_path = train_data_txt.replace(".txt", ".shapes")
        try:
            # 尝试从指定的.shape文件中加载每张图的尺寸信息
            with open(train_data_shape_path, "r") as f:
                sp = [x.split() for x in f.read().splitlines()]
        except Exception as e:
            # 否则就挨个读取图像文件获取它们的尺寸信息
            visible_img_files = tqdm(self.img_files, desc="reading image shapes")
            sp = [exif_size(Image.open(f)) for f in visible_img_files]
            np.savetxt(train_data_shape_path, sp, fmt="%g")
        # shapes记录每一张图片的宽高wh
        self.img_shapes = np.array(sp, dtype=np.float64)

        # 遍历所有的标签文件，将其加载到对象内存中
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n
        found_num, miss_num = 0, 0
        pbar = tqdm(self.label_files)
        for i, fl in enumerate(pbar):
            try:
                with open(fl, "r") as f:
                    l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
            except Exception as e:
                print("An error occurred while loading the file {}: {}".format(fl, e))
                continue

            self.labels[i] = l
            pbar.desc = "Caching labels ({} found, {} missing, for {} images)".format(found_num, miss_num, n)

    def __len__(self):
        return len(self.labels)

    def get_info(self):
        return self.img_shapes, [boxes[:, 3:] for boxes in self.labels]


def wh_iou(wh1, wh2):
    '''
    计算所有边界框和聚簇中心的9个典型边界框之间的IoU数值
    :param wh1: 所有输入的边界框数据集合
    :param wh2: 聚簇中心的9个典型边界框
    :return:
    '''

    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2] 在中间加一个维度
    wh2 = wh2[None]  # [1,M,2] 在前面加一个维度
    # np.prod()的作用是返回给定轴上的数组元素的乘积，这里的功能是将w、h相乘
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def k_means(boxes, k, dist=np.median):
    '''
    yolo k-means methods, refer: https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
    :param boxes: 需要聚类的bboxes
    :param k: 簇数(聚成几类)
    :param dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
    :return:
    '''

    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # np.random.seed(0)  # 固定随机数种子

    # init k clusters
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
        distances = 1 - wh_iou(boxes, clusters)
        current_nearest = np.argmin(distances, axis=1)
        if (last_nearest == current_nearest).all():
            break  # clusters won't change
        for cluster in range(k):
            # update clusters
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0)

        last_nearest = current_nearest

    return clusters


def anchor_fitness(whs: np.ndarray, anchors: np.ndarray, thr: float):
    '''
    以所有标注边界框与锚定框组合之间的最大最小边界比MMBR作为适应度函数值
    :param whs: 所有训练集上标注边界框的高宽信息
    :param anchors: 当前代产生9个锚定框组合
    :param thr: mmbr阈值
    :return: 将最终的边界框聚类结果返回
    '''

    ratio = whs[:, None] / anchors[None]  # 计算标注边界框与锚定框组合之间的边界比
    ratio_gamma = np.minimum(ratio, 1. / ratio).min(2)  # 计算最小边界比
    mmbr = ratio_gamma.max(1)  # 取最大最小边界比

    # 计算适应度值
    fitness_val = (mmbr * (mmbr > thr).astype(np.float32)).mean()
    # 计算最佳召回率，即匹配上anchors的边界框占总体标注边界框的比例
    best_recall = (mmbr > thr).astype(np.float32).mean()
    return fitness_val, best_recall


def anchor_cluster(img_size=512, n=9, thr=0.25, gen=1000):
    # 从数据集中读取所有图片的wh以及对应bboxes的wh
    # dataset_labels = VOCDatasetLabels(root_path="Kaist_VOC", txt_name='train.txt')
    dataset_labels = YOLODatasetLabels(train_data_txt="data/kaist_train_data.txt")
    im_wh, boxes_wh = dataset_labels.get_info()

    # 最大边缩放到img_size
    im_wh = np.array(im_wh, dtype=np.float32)
    shapes = img_size * im_wh / im_wh.max(1, keepdims=True)
    wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])  # wh

    # Filter 过滤掉小目标
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # 只保留wh都大于等于2个像素的box

    # 使用IoU度量方式对锚定框进行k-means聚类
    k = k_means(wh, n)

    # 按面积排序
    k = k[np.argsort(k.prod(1))]  # sort small to large
    f, br = anchor_fitness(k, wh, thr)
    print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best recall: {br:.5f}")

    # 使用遗传算法在kmeans的结果基础上变异（mutation）
    npr = np.random
    # f表示当前的先验框anchors的适应度，sh表示anchor的shape（9，2），
    # mpi表示变异概率（默认设置为90%），s表示变异系数sigma，控制每个基因的变化程度
    f, sh, mp, s = anchor_fitness(k, wh, thr)[0], k.shape, 0.9, 0.1
    pbar = tqdm(range(gen), desc='Evolving anchors with Genetic Algorithm:')
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # 只要有一个变异因子不为1，则跳出循环，确保至少有一个锚定框发生变异
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)

        # 进化：当前代的anchors ＊　每一个基因的变异因子（在1的附近）
        kg = (k.copy() * v).clip(min=2.0)

        # 计算适应度函数，选择最佳anchors
        fg, br = anchor_fitness(kg, wh, thr)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'

    # 按面积排序，并打印最后生成的锚定框组合结果
    k = k[np.argsort(k.prod(1))]
    print("genetic: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best recall: {br:.5f}")

    return k


def change_cfg_file_anchors(cfg_path):
    # 对训练数据集中的标注边界框宽高进行聚类
    k = anchor_cluster(img_size=512, n=9, thr=0.25, gen=1000)
    # 读取原先的模型配置文件
    with open(cfg_path, "r") as f:
        cfg_lines = f.read().splitlines()

    # 找到对应配置文件中的anchors行进行修改
    for i, cl in enumerate(cfg_lines):
        if cl.startswith("anchors"):
            cfg_lines[i] = "anchors = " + ", ".join([f"{int(i[0])}, {int(i[1])}" for i in k])

    # 将修改后的模型配置文件回写
    with open(cfg_path, "w") as f:
        f.write('\n'.join(cfg_lines))


if __name__ == "__main__":
    anchor_cluster()  # 一定要放到项目根目录运行
