from build_utils.snowflake import snowflake_cutout
from build_utils.img_utils import letterbox
from build_utils.utils import xyxy2xywh
from torch.utils.data import Dataset
from PIL import Image, ExifTags
from pathlib import Path
from tqdm import tqdm
from cv2 import cv2

import numpy as np
import random
import torch
import math
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


class LoadKaistImagesAndLabels(Dataset):
    def __init__(self,
                 data_txt_path,  # 数据集文件名列表指引文件，例如"./data/kaist_train_data.txt"
                 img_size=416,  # 默认加载成的图像大小，若原图像大小不为此，则缩放之
                 batch_size=4,
                 augment=False,  # 是否启动随机图像增强
                 hyp=None,  # 超参数字典
                 rect=False,  # 是否启动矩形框方式训练使得不同批次的图像具有不同的大小
                 single_cls=False,  # 数据集是否只有一种目标类型
                 snowflake=True,  # 是否开启雪花变化
                 pad=0.0):
        # 1、检查数据指定文件是否存在，若存在获取相应可见光和红外光图像的文件名
        #   注意data_txt_path路径指向的文件中没有加入visible和lwir单词
        try:
            data_txt_path = str(Path(data_txt_path))
            if os.path.exists(data_txt_path) and os.path.isfile(data_txt_path):
                with open(data_txt_path, "r") as f:
                    f = f.read().splitlines()
            else:
                raise Exception("'{}' doesn't exist".format(data_txt_path))

            # 获取图像路径列表，包括无类型名、可见光图像路径名、红外光图像路径名
            self.img_files = [x for x in f if os.path.splitext(x)[-1].lower() == '.jpg']  # 仅支持jpg图像文件
            self.visible_img_files = [x.replace(".jpg", "_visible.jpg") for x in self.img_files]
            self.lwir_img_files = [x.replace(".jpg", "_lwir.jpg") for x in self.img_files]

        except Exception as e:
            raise FileNotFoundError("Error loading data from '{}': {}".format(data_txt_path, e))

        # 2、使用__init__方法参数初始化加载对象中的成员变量
        n = len(self.img_files)
        assert n > 0, "No images found in '{}'".format(data_txt_path)

        # 计算batch_index和batch_num对于rect变换是有用的
        batch_index = np.floor(np.arange(n) / batch_size).astype(np.int32)  # 为每一张图像分配一个指定的batch编号
        batch_num = batch_index[-1] + 1  # 批次总数

        self.img_num = n
        self.batch_index = batch_index
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.rect = rect
        self.snowflake = snowflake
        self.quadra_trans = self.augment and not self.rect  # 是否开启4合变换

        # 3、使用img_files获得图像的标注文件路径到label_files中
        self.label_files = [x.replace("images", "labels").replace(os.path.splitext(x)[-1], ".txt")
                            for x in self.img_files]

        # 4、根据img_files读取每张图片的大小（宽高）信息写入到shapes中
        shape_path = data_txt_path.replace(".txt", ".shapes")
        try:
            # 尝试从指定的.shape文件中加载每张图的尺寸信息
            with open(shape_path, "r") as f:
                sp = [x.split() for x in f.read().splitlines()]
                assert len(sp) == n, "shape file out of sync"
        except Exception as e:
            # 否则就挨个读取图像文件获取它们的尺寸信息
            visible_img_files = tqdm(self.visible_img_files, desc="reading image shapes")
            sp = [exif_size(Image.open(f)) for f in visible_img_files]
            np.savetxt(shape_path, sp, fmt="%g")
        # shapes记录每一张图片的宽高wh
        self.shapes = np.array(sp, dtype=np.float64)

        # 5、如果开了rect模式，那么每一批次中的所有图片都会缩放到一个类似原图像比例的矩形（让最大边为img_size,
        #   另一条边做缩放操作），否则所有的图片都会默认放缩到(img_size x img_size)的图像大小。
        if self.rect:
            s = self.shapes
            aspect_ratio = s[:, 1] / s[:, 0]  # 计算高宽比
            rect_index = aspect_ratio.argsort()  # 根据高宽比对下标进行排序，这对于后续图像文件顺序调整非常有用

            # 根据高宽比，调整对象中各个图像文件名、标注文件名列表的顺序
            self.img_files = [self.img_files[i] for i in rect_index]
            self.visible_img_files = [self.visible_img_files[i] for i in rect_index]
            self.lwir_img_files = [self.lwir_img_files[i] for i in rect_index]
            self.label_files = [self.label_files[i] for i in rect_index]
            self.shapes = s[rect_index]
            aspect_ratio = aspect_ratio[rect_index]

            # 计算每个批次所会使用的图像大小
            shapes = [[1, 1]] * batch_num
            for i in range(batch_num):
                # 获取当前批次中的所有图像的高宽比
                aspect_ratio_i = aspect_ratio[i == batch_index]
                mini, maxi = aspect_ratio_i.min(), aspect_ratio_i.max()

                # 若高宽比不等于1，则调整尺寸使得其中最长边等于img_size
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]
            # 计算每一个批次需要放缩到的图像大小（同一批中的所有图像的目标放缩大小一样，但不同批图像的目标放缩大小不一定相同）
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / 32. + pad).astype(np.int32) * 32

        # 6、根据情况从labels.xxx.npy文件中或者遍历从每个原始标签文件中读取到所有的标签数据
        self.labels = [np.zeros((0, 5), dtype=np.float32)] * n  # 每个图像对应的标签以[x, 5]的shape矩阵记录
        miss_num, found_num, empty_num, duplicate_num = 0, 0, 0, 0
        labels_loaded = False

        # 命名记录所有标签信息矩阵的汇总文件路径名
        np_labels_path = str(Path(self.label_files[0]).parent) + str(".rect.npy" if rect else ".norect.npy")
        if os.path.exists(np_labels_path) and os.path.isfile(np_labels_path):
            # 如果存在标注汇总文件，则直接加载之，避免重复到标注文件目录下遍历读取所有标注文件
            x = np.load(np_labels_path, allow_pickle=True)
            if len(x) == n:
                self.labels = x
                labels_loaded = True

        # 遍历所有的标签文件，将其加载到对象内存中
        pbar = tqdm(self.label_files)
        for i, file in enumerate(pbar):
            if labels_loaded:
                l = self.labels[i]
            else:
                try:
                    with open(file, "r") as f:
                        l = np.array([x.split() for x in f.read().splitlines()], dtype=np.float32)
                except Exception as e:
                    print("An error occurred while loading the file {}: {}".format(file, e))
                    miss_num += 1
                    continue

            # 当前标签文件中有大于0个的标注信息
            if l.shape[0] > 0:
                assert l.shape[1] == 5, "> 5 label columns: {}".format(file)
                assert (l >= 0).all(), "negative labels: {}".format(file)
                assert (l[:, 1:] <= 1).all(), "non-normalized or out of bounds coordinate labels: {}".format(file)

                if np.unique(l, axis=0).shape[0] < l.shape[0]:
                    duplicate_num += 1
                if single_cls:
                    l[:, 0] = 0

                self.labels[i] = l
                found_num += 1
            else:
                empty_num += 1

            pbar.desc = "Caching labels ({} found, {} missing, {} empty, {} duplicate, for {} images)" \
                .format(found_num, miss_num, empty_num, duplicate_num, n)
        assert found_num > 0, "No labels found in {}".format(os.path.dirname(self.label_files[0]) + os.sep)

        # 7、如果是首次读取标签数据且图像数较大，那么将这些数据写入到labels.xxx.npy文件中
        if not labels_loaded and n > 1000:
            print("Saving labels to '{}' for faster future loading".format(np_labels_path))
            np.save(np_labels_path, np.array(self.labels, dtype=np.object))

    def load_image(self, index):
        '''根据索引号加载可见光和红外光图像，以及对应的原图宽高信息、缩放调整后的图像宽高'''

        # 获取可见光和红外光图像的路径
        visible_img_path = self.visible_img_files[index]
        lwir_img_path = self.lwir_img_files[index]
        assert os.path.exists(visible_img_path), "file {} not exist!".format(visible_img_path)
        assert os.path.exists(lwir_img_path), "file {} not exist!".format(lwir_img_path)

        # 读取可见光和红外光图像
        visible_img = cv2.imread(visible_img_path)
        lwir_img = cv2.imread(lwir_img_path)
        assert visible_img.shape[:2] == lwir_img.shape[:2], "visible image size != lwir image size"
        h, w = visible_img.shape[:2]

        # 将可见光和红外光图像进行等比例缩放或放大
        r = self.img_size / max(h, w)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            visible_img = cv2.resize(visible_img, (int(w * r), int(h * r)), interpolation=interp)
            lwir_img = cv2.resize(lwir_img, (int(w * r), int(h * r)), interpolation=interp)

        # 返回可见光图像、红外光图像、原图像尺寸、当前图像尺寸
        return visible_img, lwir_img, (h, w), visible_img.shape[:2]

    def load_quadra_images(self, index):
        '''
        四合交叉旋转放射畸变：将四张图像按照一个随机中心放置到(2 * img_size, 2 * img_size)
        的大图像中，然后旋转放缩平移错切这样的仿射变换
        :param index: 需加载的图像索引编号
        :return: 返回四合变换之后的可见光图像、红外光图像、调整之后的标注信息（xyxy）
        '''

        labels4 = []
        s = self.img_size
        # 随机确定一个4合图像中心点坐标
        xr, yr = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]
        # 在图像文件列表中随机选择3张其他的图像索引
        indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]

        for i, index in enumerate(indices):
            # 加载可见光图像和红外光图像
            v_img, l_img, _, (h, w) = self.load_image(index)

            if i == 0:
                # 创建4合图像（可见光背景和红外光背景）
                v_img4 = np.full((s * 2, s * 2, v_img.shape[2]), 0, dtype=np.uint8)
                l_img4 = v_img4.copy()

                # 注意在图像中的x轴指的就是横轴，y轴指的是竖轴，与数学坐标系相同
                # 计算背景将被覆盖的区域a: x1y1、x2y2，和将填充进的图像区域b: x1y1、x2y2
                x1a, y1a, x2a, y2a = max(xr - w, 0), max(yr - h, 0), xr, yr
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xr, max(yr - h, 0), min(xr + w, s * 2), yr
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xr - w, 0), yr, xr, min(s * 2, yr + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xr, w), min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xr, yr, min(xr + w, s * 2), min(s * 2, yr + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # 将当前可见光图像和红外光图像按照随机中心的位置对齐某一顶点，然后填充进大的四合图像中
            v_img4[y1a:y2a, x1a:x2a] = v_img[y1b:y2b, x1b:x2b]
            l_img4[y1a:y2a, x1a:x2a] = l_img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # 根据四合图像的与原填充进图像的相对坐标，调整四合图像上所有的标注信息
            x = self.labels[index]
            labels = x.copy()
            if x.size > 0:
                labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw  # 标注边界框在4合相框中的xmin
                labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh  # ymin
                labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw  # xmax
                labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh  # ymax
                # 注意这里其实有些标注边界框没有用了！！！这些无用的标注边界框都是放到仿射变换中处理
            labels4.append(labels)

        if len(labels4) > 0:
            # 将所有的标注信息调整汇总/对齐到同一个维度中
            labels4 = np.concatenate(labels4, axis=0)
            # 并对于越界的标注信息进行裁切
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        if self.snowflake:
            # 雪花cutout变换，注意这里传入的标注信息都是绝对xyxy形式
            v_img4, l_img4, labels4 = snowflake_cutout(v_img4, l_img4, labels4, xywh=False, n_thr=24, n_snow=80)

        # 对4合处理之后的可见光大图像和红外光大图像执行随机放射变换，并将两张图像的大小缩放到s * s
        v_img4, l_img4, labels4 = random_affine(v_img4, l_img4, labels4,
                                                degrees=self.hyp['degrees'],
                                                translate=self.hyp['translate'],
                                                scale=self.hyp['scale'],
                                                shear=self.hyp['shear'],
                                                border=-s // 2)

        return v_img4, l_img4, labels4

    def load_normal_images(self, index):
        v_img, l_img, (h0, w0), (h, w) = self.load_image(index)

        # 对上述两张图像进行一定的放缩
        shape = self.batch_shapes[self.batch_index[index]] if self.rect else self.img_size
        v_img, ratio, pad = letterbox(v_img, shape, auto=False, scale_up=self.augment)
        l_img, _1, _2 = letterbox(l_img, shape, auto=False, scale_up=self.augment)
        assert (ratio == _1 and pad == _2), "visible lwir ratio/pad is different!"
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # 对于cocotools指标计算有用

        labels = []
        x = self.labels[index]
        if x.size > 0:
            # 由于上述图像经过一定的放缩，所以图像上的标注信息也需要按照同比例进行放缩，格式为xyxy
            labels = x.copy()  # label: class, x, y, w, h
            labels[:, 1] = ratio[0] * w * (x[:, 1] - x[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (x[:, 2] - x[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (x[:, 1] + x[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (x[:, 2] + x[:, 4] / 2) + pad[1]

        if self.snowflake:
            # 雪花cutout变换，注意这里的标注信息是绝对xyxy形式
            v_img, l_img, labels = snowflake_cutout(v_img, l_img, labels, xywh=False)

        return v_img, l_img, labels, shapes

    def __len__(self):
        '''获取数据集中图像总数（对应的可见光和红外光记为一张）'''
        return len(self.img_files)

    def __getitem__(self, index):
        '''
        取数据，它根据索引读取指定图像（一张可见光和一张红外光图像）并做预处理。
        1、首先它会根据是否开启4合变换的情况，使用四合交叉旋转仿射畸变来对图像、对应标注信息做处理，
           否则执行rect变换到当前批次要求的尺寸；
        2、然后根据设置情况执行随机图像增强和随机水平翻转
        如果开启了rect标志，那么取得图像可以继续保持矩形状态；否则得到的图像一定是正方形大小，augment
        只是决定是否开启对图像的增益操作，仅在rect=False的时候执行四合雪花明化变换
        '''

        hyp = self.hyp
        if self.quadra_trans:
            # 若开启四合变换，则对按照相同的随机性对可见光和红外光图像进行4合变换
            v_img, l_img, labels = self.load_quadra_images(index)
            shapes = None
        else:
            # 若不开启四合变换，则直接加载可见光图像和红外光图像，并根据rect标志自动使用矩形加载
            v_img, l_img, labels, shapes = self.load_normal_images(index)

        if self.augment:
            # 进行随机增强图像
            if not self.quadra_trans:
                v_img, l_img, labels = random_affine(v_img, l_img, labels,
                                                     degrees=hyp['degrees'],
                                                     translate=hyp['translate'],
                                                     scale=hyp['scale'],
                                                     shear=hyp['shear'])
            augment_hsv(v_img, l_img, h_gain=hyp['hsv_h'], s_gain=hyp["hsv_s"], v_gain=hyp['hsv_v'])

        # 如果这两张对齐的可见光图像和红外光图像中具有标注的边界框，那么对其记录的xyxy标注信息转换成xywh的格式
        nl = len(labels)
        if nl:
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0-1
            labels[:, [2, 4]] /= v_img.shape[0]  # height
            labels[:, [1, 3]] /= v_img.shape[1]  # width

        # 如果执行了随机图像增强，那么就对图像进行水平随机翻转（同样的标注信息也需要执行相同的转换）
        # （注意：水平翻转的过程必须后于标注信息从xyxy形式转换为xywh）
        if self.augment:
            # random left-right flip
            lr_flip = True  # 随机水平翻转
            if lr_flip and random.random() < 0.5:
                v_img = np.fliplr(v_img)
                l_img = np.fliplr(l_img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]  # 1 - x_center

        # 计算最终的标注矩阵
        labels_out = torch.zeros((nl, 6))
        if nl:
            # 其中labels_out[:,0]对于collate_fun函数有用
            labels_out[:, 1:] = torch.from_numpy(labels)

        # 将图像从BGR HWC转换到RGB CHW
        v_img = v_img[:, :, ::-1].transpose(2, 0, 1)
        l_img = l_img[:, :, ::-1].transpose(2, 0, 1)
        v_img = np.ascontiguousarray(v_img)
        l_img = np.ascontiguousarray(l_img)

        # 返回可见光、红外光图像最终输出的Tensor元组，标注矩阵，图像文件路径
        # （去除visible等标识），变换前后的形状信息，图像文件索引编号
        return torch.from_numpy(v_img), torch.from_numpy(l_img), \
               labels_out, self.img_files[index], shapes, index

        # return v_img, l_img, labels  # 测试时使用，展示四合变换之后的效果

    def coco_index(self, index):
        """该方法专门为cocotools统计标签信息做准备，不对图像和标签做任何处理"""
        o_shapes = self.shapes[index][::-1]  # 将wh转换成hw
        # 加载标注信息
        x = self.labels[index]
        labels = x.copy()
        return torch.from_numpy(labels), o_shapes

    @staticmethod
    def collate_fn(batch):
        """该函数是用来对多个批次输入的可见光图像和红外光图像进行打包处理"""
        v_imgs, l_imgs, labels, paths, shapes, indexes = zip(*batch)
        for i, l in enumerate(labels):
            l[:, 0] = i  # 对同一批次中的不同图片的标注进行标识，方便确定标注边界框的归属
        return torch.stack(v_imgs, 0), torch.stack(l_imgs, 0), \
               torch.cat(labels, 0), paths, shapes, indexes


def random_affine(v_img, l_img, labels=(), degrees=10, translate=0.1, scale=0.1, shear=0.1, border=0):
    '''
    随机仿射变换
    :param v_img:
    :param l_img:
    :param labels:
    :param degrees: 旋转角度
    :param translate: 平移比例
    :param scale: 缩放比例
    :param shear: 错切角度
    :param border: 4合相框减小宽高度
    :return: 返回执行随机仿射变换之后的可见光图像、红外光图像以及对应的标签数据
    '''

    # 计算随机仿射变换后大的四合图像需要变换到的最终图像尺寸高度和宽度
    assert v_img.shape[:2] == l_img.shape[:2], "visible images4's size != lwir images4's size"
    target_height = v_img.shape[0] + border * 2
    target_weight = v_img.shape[1] + border * 2

    # 计算仿射矩阵
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(v_img.shape[1] / 2, v_img.shape[0] / 2), scale=s)

    # 计算平移矩阵
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * v_img.shape[0] + border
    T[1, 2] = random.uniform(-translate, translate) * v_img.shape[1] + border

    # 计算错切矩阵
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)

    # 计算仿射变换矩阵
    M = S @ T @ R
    if border != 0 or (M != np.eye(3)).any():
        # 对两张四合图像执行仿射变换，dsize指出了最终需要变换到的最终图像尺寸大小
        v_img = cv2.warpAffine(v_img, M[:2], dsize=(target_weight, target_height),
                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        l_img = cv2.warpAffine(l_img, M[:2], dsize=(target_weight, target_height),
                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    n = len(labels)
    if n > 0:
        # 对标注边界框坐标也进行仿射变换，获得映射之后的坐标位置
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
        xy = (xy @ M.T)[:, :2].reshape(n, 8)

        # 计算标注边界框在仿射变换之后的xmin，ymin，xmax，ymax
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(axis=1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # 对上述坐标进行限制，防止越界
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, target_weight)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, target_height)
        w = xy[:, 2] - xy[:, 0]
        h = xy[:, 3] - xy[:, 1]

        # 按照变换之后的高宽比以及放缩程度，过滤掉一些过小或者变形过大的标注信息
        new_area = w * h
        original_area = (labels[:, 3] - labels[:, 1]) * (labels[:, 4] - labels[:, 2])
        aspect_ratio = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
        i = (w > 4) & (h > 4) & (new_area / (original_area * s + 1e-16) > 0.2) & (aspect_ratio < 10)

        labels = labels[i]  # 删除掉那些不再有效的边界框
        labels[:, 1:5] = xy[i]  # 更新有效边界框的xmin，ymin，xmax，ymax

    return v_img, l_img, labels


def augment_hsv(v_img, l_img, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    '''
    对可见光和红外光图像执行随机图像增强
    :param v_img:
    :param l_img:
    :param h_gain:
    :param s_gain:
    :param v_gain:
    :return:
    '''

    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    hue1, sat1, val1 = cv2.split(cv2.cvtColor(v_img, cv2.COLOR_BGR2HSV))
    hue2, sat2, val2 = cv2.split(cv2.cvtColor(l_img, cv2.COLOR_BGR2HSV))
    dtype = v_img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    v_img_hsv = cv2.merge((cv2.LUT(hue1, lut_hue), cv2.LUT(sat1, lut_sat), cv2.LUT(val1, lut_val))).astype(dtype)
    l_img_hsv = cv2.merge((cv2.LUT(hue2, lut_hue), cv2.LUT(sat2, lut_sat), cv2.LUT(val2, lut_val))).astype(dtype)
    cv2.cvtColor(v_img_hsv, cv2.COLOR_HSV2BGR, dst=v_img)
    cv2.cvtColor(l_img_hsv, cv2.COLOR_HSV2BGR, dst=l_img)
