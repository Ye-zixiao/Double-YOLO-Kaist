# 该脚本文件的目的是为了能够使用yolov3-spp网络的预训练权重参数文件来生成一个
# 适合于double-yolov3-kaist网络模型的权重参数文件，方便后续的网络训练

from models import YOLOv3
from collections import OrderedDict
import torch

model = YOLOv3(cfg="config/kaist_dyolov3_1.cfg")
model_state_dict = model.state_dict()
saved_state_dict = OrderedDict()
pretrained_state_dict = torch.load("./weights/pretrained.pt")['model']

num_pre, num_back, num_different = 0, 0, 0

for k, v in pretrained_state_dict.items():
    k_split = k.split('.')
    index = int(k_split[1])
    # 复制前端特征提取网络层参数
    if index < 75:
        k1, k2 = k_split.copy(), k_split.copy()
        k1[1], k2[1] = str(index), str(index + 75)
        k1, k2 = ".".join(k1), ".".join(k2)
        if model_state_dict[k1].numel() == v.numel():
            saved_state_dict[k1] = v
            num_pre += 1
        else:
            num_different += 1
        if model_state_dict[k2].numel() == v.numel():
            saved_state_dict[k2] = v
            num_pre += 1
        else:
            num_different += 1
    # # 提取后端预测层参数
    # elif index < 113:
    #     k1 = k_split.copy()
    #     k1[1] = str(index + 81)
    #     k1 = ".".join(k1)
    #     if model_state_dict[k1].numel() == v.numel():
    #         saved_state_dict[k1] = v
    #         num_back += 1
    #     else:
    #         num_different += 1
    # else:
    #     raise ValueError("k({}) out of boundary".format(k))
    else:
        break

# 展示下复制的网络模型参数
print("saved state dict's keys:")
for k in saved_state_dict.keys():
    print(k)

# 由于pretrained.pt是一个在coco数据上训练得到的YOLOv3网络的权重参数字典，所有最后一个预测层中的
# 参数个数一定是不同于我们这里使用的Double-YOLOv3-Kaist网络的预测层权重参数的
print("num_pre: {}, num_back: {}, num_different: {}".format(num_pre, num_back, num_different))

# 保存权重参数
saved_dict = {
    'model': saved_state_dict,
    'optimizer': None,
    'training_results': None,
    "epoch": -1
}
torch.save(saved_dict, "weights/pretrained_dyolov3.pt")
