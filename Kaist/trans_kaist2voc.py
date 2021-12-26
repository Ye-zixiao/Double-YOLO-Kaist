from scipy.io import loadmat
from collections import defaultdict
from lxml import etree, objectify
from tqdm import tqdm
import shutil
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

    # objStr = vbb['A'][0][0][5][0]
    # objEnd = vbb['A'][0][0][6][0]
    # objHide = vbb['A'][0][0][7][0]
    # altered = int(vbb['A'][0][0][8][0][0])
    # log = vbb['A'][0][0][9][0]
    # logLen = int(vbb['A'][0][0][10][0][0])

    # objLists = vbb['A'][0][0][1][0]
    # objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    # person index
    # person_index_list = np.where(np.array(objLbl) == "person")[0]

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
    :return:
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


def parse_anno_files(kaist_dir, extract_step=3, label_threshold=2):
    assert os.path.exists(kaist_dir), "kaist dataset not exist"
    annotation_dir = os.path.join(kaist_dir, "annotations-vbb")
    annotation_sub_dirs = os.listdir(annotation_dir)

    # 创建目标转移VOC目录、存放图片目录以及标准目录
    voc_dir = os.path.join(kaist_dir, "../Kaist_VOC")
    if not os.path.exists(voc_dir):
        os.makedirs(voc_dir)

    voc_anno_dir = os.path.join(voc_dir, "Annotations")
    if not os.path.exists(voc_anno_dir):
        os.makedirs(voc_anno_dir)

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

        for vbb_file in tqdm(vbb_files, desc="translate {} ...".format(sub_dir)):
            annos = vbb_anno2dict(vbb_file, sub_dir)
            if annos:
                step = 0
                for filename, anno in sorted(annos.items(), key=lambda x: x[0]):
                    if "bbox" in anno:
                        indexes = [i for i, l in enumerate(anno['label']) if l == 'person']
                        anno['label'] = [anno['label'][i] for i in indexes]
                        anno['occlusion'] = [anno['occlusion'][i] for i in indexes]
                        anno['bbox'] = [anno['bbox'][i] for i in indexes]

                        if len(anno['label']) < label_threshold:
                            continue

                        # 提供简单的跨步抽样功能
                        if step == extract_step:
                            step = 0
                            continue
                        step += 1

                        new_filename = filename.replace('/', '_')
                        anno_outfile = os.path.join(voc_anno_dir, new_filename + ".xml")

                        # 修改标注字典中的部分信息
                        anno['id'] = anno['filename'] = new_filename
                        anno['label'] = ['person'] * len(anno['label'])

                        # 将标准信息写入到Kaist_VOC下的Annotations目录中
                        anno_tree = instance2xml_base(anno, IMAGE_SIZE)
                        etree.ElementTree(anno_tree).write(anno_outfile, pretty_print=True)

                        # 将标注指向的可见光和红外光图像文件拷贝到Kaist_VOC下的JPEGImages/xxx目录中
                        filename_split = filename.split('/')
                        filename_split.append(filename_split[-1])
                        filename_split[-2] = "visible"

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


if __name__ == '__main__':
    parse_anno_files(os.getcwd())
