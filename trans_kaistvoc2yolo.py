from tqdm import tqdm
from lxml import etree
import shutil
import json
import os

label_json_path = "data/kaist_voc_classes.json"
kaist_voc_root = "Kaist_VOC"
yolo_root = "Kaist_YOLO"

if not os.path.exists(yolo_root):
    os.makedirs(yolo_root)

voc_images_path = os.path.join(kaist_voc_root, "JPEGImages")
voc_xml_path = os.path.join(kaist_voc_root, "Annotations")


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def translate_info(file_names: list, save_root: str, class_dict: dict, type_str='train'):
    """
    将对应xml文件信息转为yolo中使用的txt文件信息
    :param file_names:
    :param save_root:
    :param class_dict:
    :param type_str:
    :return:
    """
    # 在Kaist_YOLO/train或Kaist_YOLO/val目录下创建/打开lables和images目录
    save_txt_path = os.path.join(save_root, type_str, "labels")
    if os.path.exists(save_txt_path) is False:
        os.makedirs(save_txt_path)
    save_images_path = os.path.join(save_root, type_str, "images")
    if os.path.exists(save_images_path) is False:
        os.makedirs(save_images_path)

    for file in tqdm(file_names, desc="translate {} files...".format(type_str)):
        # 检查下图像文件是否存在
        visible_img_path = os.path.join(voc_images_path, "visible", file + ".jpg")
        assert os.path.exists(visible_img_path), "file: '{}' not exist...".format(visible_img_path)

        lwir_img_path = os.path.join(voc_images_path, "lwir", file + ".jpg")
        assert os.path.exists(lwir_img_path), "file: '{}' not exist...".format(lwir_img_path)

        # 检查xml文件是否存在
        xml_path = os.path.join(voc_xml_path, file + ".xml")
        assert os.path.exists(xml_path), "file: '{}' not exist...".format(xml_path)

        # read xml
        with open(xml_path) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = parse_xml_to_dict(xml)["annotation"]
        img_height = int(data["size"]["height"])
        img_width = int(data["size"]["width"])

        # write object info into txt
        assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
        if len(data["object"]) == 0:
            # 如果xml文件中没有目标就直接忽略该样本
            print("Warning: in '{}' xml, there are no objects.".format(xml_path))
            continue

        with open(os.path.join(save_txt_path, file + ".txt"), "w") as f:
            for index, obj in enumerate(data["object"]):
                # 获取每个object的box信息
                xmin = float(obj["bndbox"]["xmin"])
                xmax = float(obj["bndbox"]["xmax"])
                ymin = float(obj["bndbox"]["ymin"])
                ymax = float(obj["bndbox"]["ymax"])
                class_name = obj["name"]
                class_index = class_dict[class_name] - 1  # 目标id从0开始

                # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
                if xmax <= xmin or ymax <= ymin:
                    print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                    continue

                # 将box信息转换到yolo格式
                xcenter = xmin + (xmax - xmin) / 2
                ycenter = ymin + (ymax - ymin) / 2
                w = xmax - xmin
                h = ymax - ymin

                # 绝对坐标转相对坐标，保存6位小数
                xcenter = round(xcenter / img_width, 6)
                ycenter = round(ycenter / img_height, 6)
                w = round(w / img_width, 6)
                h = round(h / img_height, 6)

                info = [str(i) for i in [class_index, xcenter, ycenter, w, h]]

                if index == 0:
                    f.write(" ".join(info))
                else:
                    f.write("\n" + " ".join(info))

        # copy image into save_images_path
        # 复制可见光图像到指定目录
        visible_path_copy_to = os.path.join(save_images_path, file + "_visible.jpg")
        if not os.path.exists(visible_path_copy_to):
            shutil.copyfile(visible_img_path, visible_path_copy_to)
        # 复制红外光图像到指定目录
        lwir_path_copy_to = os.path.join(save_images_path, file + "_lwir.jpg")
        if not os.path.exists(lwir_path_copy_to):
            shutil.copyfile(lwir_img_path, lwir_path_copy_to)


def create_class_names(class_dict: dict):
    keys = class_dict.keys()
    with open("data/kaist_data_label.names", "w") as w:
        for index, k in enumerate(keys):
            if index + 1 == len(keys):
                w.write(k)
            else:
                w.write(k + "\n")


def main():
    # 读取类别索引映射文件
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)

    # 指定VOC数据集txt文件路径
    voc_txt_names = ['train', 'val', 'test', 'day_test', 'night_test']
    voc_txt_files = [tn + '.txt' for tn in voc_txt_names]
    txt_paths = [os.path.join(kaist_voc_root, "ImageSets/Main", x) for x in voc_txt_files]
    for tp in txt_paths:
        assert os.path.exists(tp), f"VOC image set txt file '{tp}' not exist!"

    # 将VOC数据集转换成YOLO数据集格式
    for i, (tp, tn) in zip(txt_paths, voc_txt_names):
        # 读取.txt中的所有行信息，删除空行
        with open(tp, "r") as r:
            train_file_names = [i for i in r.read().splitlines() if len(i.strip()) > 0]
        # 将voc信息转yolo，并将图像文件复制到相应文件夹
        translate_info(train_file_names, yolo_root, class_dict, tn)

    # 创建kaist_data_label.names文件
    create_class_names(class_dict)


if __name__ == '__main__':
    main()
