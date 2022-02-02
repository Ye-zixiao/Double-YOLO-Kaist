import random
import glob
import os


def dataset_split(set_dict, kaist_voc_root="Kaist_VOC"):
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


def dict_dump(file_dict, kaist_voc_root="Kaist_VOC"):
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


if __name__ == "__main__":
    set_dict = {
        'set00': {'train': 0.8, 'valid': 0.1, 'day_test': 0.1, 'night_test': 0},
        'set01': {'train': 0.8, 'valid': 0.2, 'day_test': 0, 'night_test': 0},
        'set02': {'train': 0.7, 'valid': 0.2, 'day_test': 0.1, 'night_test': 0},
        'set03': {'train': 0.8, 'valid': 0.1, 'day_test': 0, 'night_test': 0.1},
        'set04': {'train': 0.8, 'valid': 0.1, 'day_test': 0, 'night_test': 0.1},
        'set05': {'train': 0.8, 'valid': 0.2, 'day_test': 0, 'night_test': 0},
        'set06': {'train': 0.2, 'valid': 0.2, 'day_test': 0.6, 'night_test': 0},
        'set07': {'train': 0.8, 'valid': 0.1, 'day_test': 0.1, 'night_test': 0},
        'set08': {'train': 0.7, 'valid': 0.2, 'day_test': 0.1, 'night_test': 0},
        'set09': {'train': 0.1, 'valid': 0.1, 'day_test': 0, 'night_test': 0.8},
        'set10': {'train': 0.1, 'valid': 0.1, 'day_test': 0, 'night_test': 0.8},
        'set11': {'train': 0.3, 'valid': 0.2, 'day_test': 0, 'night_test': 0.5},
    }

    dt = dataset_split(set_dict, kaist_voc_root="Kaist_VOC")
    print("in summary, train: {}, valid: {}, test: {}, day_test: {}, night_test: {}".format(
        len(dt['train']), len(dt['val']), len(dt['test']), len(dt['day_test']), len(dt['night_test'])))

    dict_dump(dt, kaist_voc_root="Kaist_VOC")
