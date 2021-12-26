from tqdm import tqdm
import random
import glob
import os


def dataset_split(kaist_dir, train_scale=0.7):
    '''
    :param kaist_dir:
    :param train_scale:
    :param val_scale:
    :return:
    '''
    # 创建存放训练集、测试集名的两个文件
    kaist_voc_imagesets_dir = os.path.join(kaist_dir, "ImageSets")
    kaist_voc_imagesets_main_dir = os.path.join(kaist_voc_imagesets_dir, "Main")
    if not os.path.exists(kaist_voc_imagesets_dir):
        os.mkdir(kaist_voc_imagesets_dir)
    if not os.path.exists(kaist_voc_imagesets_main_dir):
        os.mkdir(kaist_voc_imagesets_main_dir)

    # 读取所有文件名，并放入到列表当中
    kaist_voc_anno_dir = os.path.join(kaist_dir, "Annotations")
    filenames = glob.glob(os.path.join(kaist_voc_anno_dir, "*.xml"))
    filenames_len = len(filenames)
    filenames_index_list = list(range(filenames_len))
    random.shuffle(filenames_index_list)  # 打乱所有图片的文件名

    # 分配数据集中训练集和测试集的比例
    train_num = filenames_len * train_scale
    val_num = filenames_len - train_num
    curr_idx = 0

    # 打开train.txt和val.txt
    train_file = open(os.path.join(kaist_voc_imagesets_main_dir, "train.txt"), "w")
    val_file = open(os.path.join(kaist_voc_imagesets_main_dir, "val.txt"), "w")

    # 分配图片同时各自写入到对应txt文件
    for i in tqdm(filenames_index_list, desc="split dataset ..."):
        fn = filenames[i].split('\\')[-1].strip(".xml") # Linux上使用‘/’来进行分割
        if curr_idx < train_num:
            train_file.write(fn + "\n")
        else:
            val_file.write(fn + "\n")

        curr_idx += 1

    print("训练集{}数量: {}张".format(train_file.name, train_num))
    print("测试集{}数量: {}张".format(val_file.name, val_num))

    train_file.close()
    val_file.close()


if __name__ == "__main__":
    dataset_split("Kaist_VOC")
