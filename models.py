from build_utils.parse_config import *
from build_utils.layers import *


def create_modules(modules_defs: list, img_size):
    '''
    根据网络模块定义列表创建网络模型对象，并返回该对象、后续仍要使用的层索引以及网络信息
    :param modules_defs: 记录网络配置信息的字典列表
    :param img_size: 图像大小
    :return: 模型对象、后续需要使用的层输出索引以及网络元信息字典组成的元组
    '''
    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    net_infos = modules_defs[0]  # modules_defs中第一个字典记录网络的纵览信息
    modules_defs.pop(0)

    pre_out_filters = [3]  # 记录前一个网络层的输出通道数in_channels，即深度
    module_list = nn.ModuleList()  # 网络层列表
    routs = []  # 有部分网络层的输出在后续需要被再次使用，所以记录它们的索引
    yolo_index = -1  # 当前yolo层索引

    for i, mdef in enumerate(modules_defs):
        modules = nn.Sequential()

        if mdef['type'] == "convolutional":
            bn = mdef['batch_normalize']  # 是否加入BN层，0或1
            filters = mdef['filters']  # out_channels
            k = mdef['size']  # 卷积核
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(
                    in_channels=pre_out_filters[-1] if "second_index" not in net_infos or
                                                       i != net_infos["second_index"] else 3,
                    out_channels=filters,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2 if mdef['pad'] else 0,
                    bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type")

            if bn:
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                routs.append(i)  # 实际上这一步我感觉没有必要

            if mdef['activation'] == 'leaky':
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))

        elif mdef['type'] == "maxpool":
            k = mdef['size']
            stride = mdef['stride']
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef['type'] == "upsample":
            modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == "route":
            layers = mdef['layers']  # route结构所指向的浅层网络索引
            # 计算这些输入层在深度channels上叠加后的总深度
            filters = sum([pre_out_filters[l + 1 if l > 0 else l] for l in layers])
            # route的layers字段所指向的网络层输出必须保存
            layers = [i + l if l < 0 else l for l in layers]
            routs.extend(layers)
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == "shortcut":
            layers = mdef['from']
            filters = pre_out_filters[-1]
            # shortcut的from字段所指向的网络层输出必须保存
            routs.append(i + layers[0])
            modules = WeightedFeatureFusion(layers=[i + l if l < 0 else l for l in layers],
                                            weight="weights_type" in mdef)

        elif mdef['type'] == "yolo":
            yolo_index += 1
            stride = [32, 16, 8]

            modules = YOLOLayer(anchors=mdef['anchors'][mdef['mask']],
                                nc=mdef['classes'],
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # 根据上述论文初始化预测层的Conv2d网络层中的参数
            try:
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        module_list.append(modules)
        pre_out_filters.append(filters)
        # # 因为我们使用的YOLOv3模型可能有两个图像数据输入，在这种情况下第75个网络层就有可能
        # # 是第二张图像数据输入后第一个需要处理的卷积层，因此需要注意下这个层的输入通道数的设置
        # if "second_index" in net_infos:
        #     if net_infos['second_index'] - 1 == i:
        #         saved_filter = pre_out_filters[-1]
        #         pre_out_filters[-1] = 3
        #     elif net_infos['second_index'] == i:
        #         pre_out_filters[-2] = saved_filter

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary, net_infos


class YOLOLayer(nn.Module):
    '''
    YOLOLayer指的是每一个yolo预测分支最后一个卷积Conv2d预测器之后的网络层。
    当预测层predictor（也就是每一个分支上孤零零的Conv2d）输出的矩阵通过YOLO层
    会得到一个维度为[1, nx * ny * anchor_num, classes_num + 1 + 4]的矩阵
    '''

    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)  # anchors预设边界框
        self.stride = stride  # YOLO层输出的特征图对应原图上的步距[32, 16, 8]
        self.na = len(anchors)  # 一个网格上能够生成的预设框数量
        self.nc = nc  # 目标类别总数
        self.no = nc + 5  # 一个边界框bbox对应的参数个数: classes_num + confidence + xywh
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # nx、ny表示预测特征层的宽度和高度，ng表示网格grid cell的大小
        self.anchor_vec = self.anchors / self.stride  # 将anchors缩放到gride cell级别的尺度
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)  # 将每一个anchor的wh两两成对到同一个维度
        self.grid = None

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng  # ng表示网格grid cell的大小
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        # 只有在推理阶段我们才要将网络预测得到的相对边界框中心xy，相对边界框宽高wh映射回原图对应的比例，
        # 这也就意味着它需要程序先计算出这个预测边界框在输出特征图上的中心相对位置，这个过程需要meshgrid
        # 函数先生成每一个网格的相对坐标，然后进一步计算
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            # torch.meshgrid()的功能是生成网格，可以用于生成坐标。我们可以直接Ctrl+鼠标左键直接进入查看示例。
            # 而torch.stack()函数的功能就是将两个矩阵在指定维度（由dim参数给出）上进行堆叠相接
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # 计算出每一个网格的相对坐标，然后放到一个大小为(batch_size, na, grid_h, grid_w, 2)
            # 的矩阵中（最底层的维度正好放wh）
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()
        else:
            # 但在训练的过程中由于图片标注信息本身就是相对比例下的xywh，所以直接拿去计算定位损失函数就可以了
            pass

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    def forward(self, p):
        bs, _, ny, nx = p.shape  # batch_size, predict_param(255 = (80 + 1 + 4) * 3), grid(13), grid(13)
        if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
            # 在必要时创建网格，因为每一张输入图像的大小是不同的，且第一次的时候并没有生成网格
            self.create_grids((nx, ny), p.device)

        # 使用view和permute调整尺寸，使得其大小变换成(batch_size, anchor_num, grid_w, grid_h, classes + 1 + 4)
        # 而原先输入的矩阵其实还算是属于传统特征图的形式，与我们最终想要得到的结果只差一步，所以需要转换
        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor_num, grid, grid, xywh + obj + classes]
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p  # 如果是训练阶段，直接返回上面的p即可，剩下的直接交给后续的损失函数即可
        else:
            # inference 将推理得到的边界框中心点、宽和高回归，然后映射到原图大小，其实就是按照目标边界框公式计算
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output

            # 从下面的代码中我们也可以矩阵p中每一个bbox的参数的放置顺序是：
            #   0  1  2  3    4      5...N-1
            #   x  y  w  h  conf  classes_scores
            # io[...,:2]和self.grid这两个矩阵的大小都是(batch_size, anchor_num, grid_w, grid_h, 2)
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            torch.sigmoid_(io[..., 4:])  # 后面的置信度和类别分数通过激活函数处理
            # 计算得到最终映射回原图尺寸后，我们就不需要矩阵中的grid_w和grid_h两个维度，我们直接将所有的
            # 边界框信息直接返回即可。这里在预测的时候还会将原来YOLO层的输入矩阵p一同返回
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class YOLOv3(nn.Module):
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(YOLOv3, self).__init__()
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs, self.net_info = create_modules(self.module_defs, img_size)
        self.yolo_layers = get_yolo_layers(self)

        self.info(verbose)

    def get_yolo_layers(self):
        return [i for i, module in enumerate(self.module_list)
                if module.__class__.__name__ == 'YOLOLayer']

    def info(self, verbose=False):
        torch_utils.model_info(self, verbose)

    def forward(self, x, y=None):
        '''
        正向传播函数，既支持普通单通道图像数据的输入，也支持双通道图像数据的输入，
        这完全取决于网络模型的配置文件的结果
        :param x: 一张图片时类型为torch.Tensor，两张图片输入时类型为tuple[torch.Tensor,torch.Tensor]
        :return: 1、训练时，输出预测器predictor(Conv2d)生成的矩阵组成的列表p，长度为3
                 2、测试时，输出YOLO层生成的记录多个bbox参数的矩阵，以及上述预测器的输出p
        '''
        di = "second_index" in self.net_info and y is not None
        yolo_out, out = [], []

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__

            if name in ['WeightedFeatureFusion', 'FeatureConcat']:
                x = module(x, out)
            elif name == 'YOLOLayer':
                yolo_out.append(module(x))
            else:
                if di and i == self.net_info['second_index'] == i:
                    # 模型有两个图像输入，且当前正好是该处理第二个图像数据的卷积层
                    x = module(y)
                else:
                    x = module(x)

            out.append(x if self.routs[i] else [])

        if self.training:
            # 如果是训练模式，则直接得到的是分支predictor上输出的list(torch.Tensor)，内部
            # 张量的维度为：(bs, anchor_num, grid, grid, xywh + obj + classes)
            return yolo_out
        else:
            # 否则输出的是一个记录含有多个bbox边界框实际中心点宽高+置信度+类别分数的矩阵以及输入p
            x, p = zip(*yolo_out)
            return torch.cat(x, 1), p
