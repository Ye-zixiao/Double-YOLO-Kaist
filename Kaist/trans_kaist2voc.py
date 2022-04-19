from scipy.io import loadmat
from collections import defaultdict
from lxml import etree, objectify
import shutil
import random
import tqdm
import glob
import os

IMAGE_SIZE = (640, 512)  # KAIST Multispectral Benchmark


def vbb_anno2dict(vbb_file, sub_dir):
    '''
    将vbb文件转换成dict形式的字典
    :param vbb_file:
    :param sub_dir:
    :return: 返回记录有标注信息的字典dict，注意这是一个字典的字典
            每一个内部的字典元素都是一个对图片的标注信息
    '''

    vid_name = os.path.splitext(os.path.basename(vbb_file))[0]
    annos = defaultdict(dict)
    vbb = loadmat(vbb_file)
    # object info in each frame: id, pos, occlusion, lock, posv

    objLists = vbb['A'][0][0][1][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]

    nFrame = int(vbb['A'][0][0][0][0][0])
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]

    for frame_id, obj in enumerate(objLists):

        frame_name = '/'.join([sub_dir, vid_name, 'I{:05d}'.format(frame_id)])
        annos[frame_name] = defaultdict(list)
        annos[frame_name]["id"] = frame_name

        if len(obj[0]) > 0:
            for id, pos, occl, lock, posv in zip(
                    obj['id'][0], obj['pos'][0], obj['occl'][0],
                    obj['lock'][0], obj['posv'][0]):
                id = int(id[0][0]) - 1  # for matlab start from 1 not 0
                pos = pos[0].tolist()
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                annos[frame_name]["label"].append(objLbl[id])
                annos[frame_name]["occlusion"].append(occl)
                annos[frame_name]["bbox"].append(pos)

    return annos


def instance2xml_base(anno, img_size, bbox_type='xyxy'):
    '''
    根据给定带有标注信息的字典生成xml树
    :param anno: 一个带有标注信息的字典
    :param img_size: 图像大小
    :param bbox_type:bbox类型
    :return: 返回xml树
    '''

    """bbox_type: xyxy (xmin, ymin, xmax, ymax); xywh (xmin, ymin, width, height)"""
    assert bbox_type in ['xyxy', 'xywh']

    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('KAIST Multispectral Ped Benchmark'),
        E.filename(anno['id']),
        E.source(
            E.database('KAIST pedestrian'),
            E.annotation('KAIST pedestrian'),
            E.image('KAIST pedestrian'),
            E.url('https://soonminhwang.github.io/rgbt-ped-detection/')
        ),
        E.size(
            E.width(img_size[0]),
            E.height(img_size[1]),
            E.depth(4)
        ),
        E.segmented(0),
    )
    for index, bbox in enumerate(anno['bbox']):
        bbox = [float(x) for x in bbox]
        if bbox_type == 'xyxy':
            xmin, ymin, w, h = bbox
            xmax = xmin + w
            ymax = ymin + h
        else:
            xmin, ymin, xmax, ymax = bbox

        E = objectify.ElementMaker(annotate=False)

        anno_tree.append(
            E.object(
                E.name(anno['label'][index]),
                E.bndbox(
                    E.xmin(xmin),
                    E.ymin(ymin),
                    E.xmax(xmax),
                    E.ymax(ymax)
                ),
                E.pose('unknown'),
                E.truncated(0),
                E.difficult(0),
                E.occlusion(anno["occlusion"][index])
            )
        )
    return anno_tree


def parse_anno_files(kaist_dir, sample_step=3, label_threshold=2):
    '''
    将KAIST自定义格式vbb格式下的数据集转换为VOC格式下的数据集
    :param kaist_dir: KAIST数据集根目录
    :param sample_step: 抽样距离，防止相邻帧之间重复内容图片的出现
    :param label_threshold: 如果图像中有超过或等于该数目的标注边界框，则允许抽取之
    :return:
    '''

    assert os.path.exists(kaist_dir), "kaist dataset not exist"
    annotation_dir = os.path.join(kaist_dir, "annotations-vbb")
    annotation_sub_dirs = os.listdir(annotation_dir)

    # 创建目标转移VOC根目录
    voc_dir = os.path.join(kaist_dir, "../Kaist_VOC")
    if not os.path.exists(voc_dir):
        os.makedirs(voc_dir)

    # 创建目标转移VOC根目录下的存放标注信息的Annotations目录
    voc_anno_dir = os.path.join(voc_dir, "Annotations")
    if not os.path.exists(voc_anno_dir):
        os.makedirs(voc_anno_dir)

    # 创建目标转移VOC根目录下的存放图片的JPEGImages目录及其子目录
    voc_image_dir = os.path.join(voc_dir, "JPEGImages")
    if not os.path.exists(voc_image_dir):
        os.makedirs(voc_image_dir)
    voc_visible_image_dir = os.path.join(voc_image_dir, "visible")
    voc_lwir_image_dir = os.path.join(voc_image_dir, "lwir")
    if not os.path.exists(voc_visible_image_dir):
        os.makedirs(voc_visible_image_dir)
    if not os.path.exists(voc_lwir_image_dir):
        os.makedirs(voc_lwir_image_dir)

    # 遍历所有的vbb注释文件
    for sub_dir in annotation_sub_dirs:
        vbb_files = glob.glob(os.path.join(annotation_dir, sub_dir, "*.vbb"))

        for vbb_file in tqdm.tqdm(vbb_files, desc="translate {}...".format(sub_dir.split('\\')[-1])):
            # 将KAIST数据集使用的vbb注释格式文件转换成字典表数据，其中每一个key代表一个图片文件
            annos = vbb_anno2dict(vbb_file, sub_dir)
            if annos:
                step = 0
                # 遍历字典表数据中的所有图片
                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    # 如果当前图片中有标注边界框存在
                    if "bbox" in anno and step == 0:
                        # 我们使用图像中标注为person和cyclist的标注边界框，而不使用people！！
                        indexes = [i for i, l in enumerate(anno['label']) if l in ['person', 'cyclist']]
                        anno['label'] = [anno['label'][i] for i in indexes]
                        anno['occlusion'] = [anno['occlusion'][i] for i in indexes]
                        anno['bbox'] = [anno['bbox'][i] for i in indexes]

                        # 若这样图片中的标注边界框数量未达到指定阈值，则不选择之
                        if len(anno['label']) < label_threshold:
                            continue

                        new_filename = filename.replace('/', '_')
                        anno_outfile = os.path.join(voc_anno_dir, new_filename + ".xml")

                        # 修改标注字典中的部分信息
                        anno['id'] = new_filename
                        anno['label'] = ['person'] * len(anno['label'])

                        # 将标注信息转换为xml格式写入到Kaist_VOC下的Annotations目录中
                        anno_tree = instance2xml_base(anno, IMAGE_SIZE)
                        etree.ElementTree(anno_tree).write(anno_outfile, pretty_print=True)

                        # 将标注指向的可见光和红外光图像文件拷贝到Kaist_VOC下的JPEGImages/xxx目录中
                        filename_split = filename.split('/')
                        filename_split.insert(-1, "visible")

                        # 拷贝可将光版本的图像文件
                        old_visible_image_path = os.path.join(kaist_dir, "/".join(filename_split) + ".jpg")
                        if not os.path.exists(old_visible_image_path):
                            os.remove(anno_outfile)
                            continue
                        visible_image_target_path = os.path.join(voc_visible_image_dir, new_filename + ".jpg")
                        shutil.copyfile(old_visible_image_path, visible_image_target_path)

                        # 拷贝红外光版本的图像文件
                        filename_split[-2] = "lwir"
                        old_lwir_image_path = os.path.join(kaist_dir, "/".join(filename_split) + ".jpg")
                        if not os.path.exists(old_lwir_image_path):
                            os.remove(anno_outfile)
                            os.remove(visible_image_target_path)
                            continue
                        lwir_image_target_path = os.path.join(voc_lwir_image_dir, new_filename + ".jpg")
                        shutil.copyfile(old_lwir_image_path, lwir_image_target_path)

                    assert sample_step > 0, "sample step can't be greater or equal to 0"
                    step = (step + 1) % sample_step


def dataset_split(set_dict, kaist_voc_root="../Kaist_VOC"):
    '''
    将数据集进行划分，划分出训练集、验证集、白天测试集、夜间测试集
    :param set_dict: 各个数据集划分比例字典，形如{'set00':{'train': 0.8, 'valid': 0.1,'day_test':0.1},}
    :param kaist_voc_root: KAIST_VOC数据集根目录
    :return: 返回训练集文件名，验证集文件名，全天候、日间、夜间测试集文件名
    '''

    # 创建相关的目录
    assert os.path.exists(kaist_voc_root), "kaist voc root dir '{}' not exist!".format(kaist_voc_root)
    kaist_voc_anno_dir = os.path.join(kaist_voc_root, "Annotations")
    assert os.path.exists(kaist_voc_anno_dir), "kaist voc anno dir '{}' not exist!".format(kaist_voc_anno_dir)

    train_files = []
    valid_files = []
    day_test_files = []
    night_test_files = []

    # 将相关的图片放入到数据集中
    for k, v in set_dict.items():
        anno_files = glob.glob(os.path.join(kaist_voc_anno_dir, "*{}*.xml".format(k)))
        anno_files = [fn.split('\\')[-1].strip(".xml") for fn in anno_files]  # 在Linux上应该使用'/'
        random.shuffle(anno_files)

        total_num = len(anno_files)
        train_num = int(total_num * v['train'])
        valid_num = int(total_num * v['valid'])
        day_test_num = int(total_num * v['day_test'])
        night_test_num = int(total_num * v['night_test'])
        print("extracted from the dataset '{}'({}): train {}, valid {}, day_test {}, night_test {}".
              format(k, total_num, train_num, valid_num, day_test_num, night_test_num))

        train_files += anno_files[:train_num]
        valid_files += anno_files[train_num:train_num + valid_num]
        day_test_files += anno_files[train_num + valid_num:train_num + valid_num + day_test_num]
        night_test_files += anno_files[train_num + valid_num + day_test_num:train_num + valid_num
                                                                            + day_test_num + night_test_num]

    return {'train': train_files,
            'val': valid_files,
            'test': day_test_files + night_test_files,
            'day_test': day_test_files,
            'night_test': night_test_files}


def dict_dump(file_dict, kaist_voc_root="../Kaist_VOC"):
    '''将file_dict字典中的各个划分数据集文件名写入到不同的txt文件中'''

    # 创建相关的目录
    assert os.path.exists(kaist_voc_root), "kaist voc root dir '{}' not exist!".format(kaist_voc_root)
    kaist_voc_imgsets_dir = os.path.join(kaist_voc_root, "ImageSets")
    kaist_voc_imagesets_main_dir = os.path.join(kaist_voc_imgsets_dir, "Main")
    if not os.path.exists(kaist_voc_imgsets_dir):
        os.mkdir(kaist_voc_imgsets_dir)
    if not os.path.exists(kaist_voc_imagesets_main_dir):
        os.mkdir(kaist_voc_imagesets_main_dir)

    # 将划分到不同数据集的图片文件名写入到各自的txt文件中
    for k, fns in file_dict.items():
        txt_file_path = os.path.join(kaist_voc_imagesets_main_dir, f"{k}.txt")
        with open(txt_file_path, "w") as f:
            for fn in fns:
                f.write(fn + '\n')
        print(f"{k}.txt done")


if __name__ == '__main__':
    # 将Kaist数据集转换为VOC格式数据集
    parse_anno_files(os.getcwd(), sample_step=3, label_threshold=1)

    # 按照下面的指示对数据集进行随机划分
    set_dict = {
        'set00': {'train': 0.8, 'valid': 0.1, 'day_test': 0.1, 'night_test': 0.0},
        'set01': {'train': 0.8, 'valid': 0.2, 'day_test': 0.0, 'night_test': 0.0},
        'set02': {'train': 0.7, 'valid': 0.2, 'day_test': 0.1, 'night_test': 0.0},
        'set03': {'train': 0.8, 'valid': 0.1, 'day_test': 0.0, 'night_test': 0.1},
        'set04': {'train': 0.8, 'valid': 0.1, 'day_test': 0.0, 'night_test': 0.1},
        'set05': {'train': 0.8, 'valid': 0.2, 'day_test': 0.0, 'night_test': 0.0},
        'set06': {'train': 0.0, 'valid': 0.0, 'day_test': 1.0, 'night_test': 0.0},
        'set07': {'train': 0.8, 'valid': 0.1, 'day_test': 0.1, 'night_test': 0.0},
        'set08': {'train': 0.7, 'valid': 0.2, 'day_test': 0.1, 'night_test': 0.0},
        'set09': {'train': 0.0, 'valid': 0.0, 'day_test': 0.0, 'night_test': 1.0},
        'set10': {'train': 0.0, 'valid': 0.0, 'day_test': 0.0, 'night_test': 1.0},
        'set11': {'train': 0.3, 'valid': 0.2, 'day_test': 0.0, 'night_test': 0.5},
    }

    dt = dataset_split(set_dict, kaist_voc_root="../Kaist_VOC")
    print("in summary, train: {}, valid: {}, test: {}, day_test: {}, night_test: {}".format(
        len(dt['train']), len(dt['val']), len(dt['test']), len(dt['day_test']), len(dt['night_test'])))

    dict_dump(dt, kaist_voc_root="../Kaist_VOC")
