import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math
import sys


def make_divisible(v, divisor):
    # 对输入值向上补齐，使其变成divisor的倍数
    return math.ceil(v / divisor) * divisor


class Flatten(nn.Module):
    # Use after nn.AdaptiveAvgPool2d(1) to remove last 2 dimensions
    def forward(self, x):
        # 使张量的第一维不变，后面的维度全展平到第二个维度中
        return x.view(x.size(0), -1)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # 输入一个张量元组，将它们在深度上进行拼接
        return torch.cat(x, self.d)


class FeatureConcat(nn.Module):
    """
    将多个特征矩阵在channel维度进行concatenate拼接
    """

    def __init__(self, layers):
        super(FeatureConcat, self).__init__()
        self.layers = layers  # layer indices
        self.multiple = len(layers) > 1  # multiple layers flag

    def forward(self, x, outputs):
        # 将前面指定的某一层输出当作当前层的输出，或是将前面多个层的输出张量在深度上进行拼接再输出
        return torch.cat([outputs[i] for i in self.layers], 1) if self.multiple else outputs[self.layers[0]]


class WeightedFeatureFusion(nn.Module):  # weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070
    """
    将多个特征矩阵的值进行融合(add操作，加权特征融合)。注意这是对应维度上的操作，不会增加深度
    （默认情况下不做任何加权处理）
    """

    def __init__(self, layers, weight=False):
        super(WeightedFeatureFusion, self).__init__()
        self.layers = layers  # layer indices
        self.weight = weight  # apply weights boolean
        self.n = len(layers) + 1  # number of layers 融合的特征矩阵个数
        if weight:
            # nn.Parameter含义是将一个固定不可训练的tensor转换成可以训练的类型parameter，
            # 这样加权使用的权值都可训练、可学习的
            self.w = nn.Parameter(torch.zeros(self.n), requires_grad=True)  # layer weights

    def forward(self, x, outputs):
        # Weights，默认不加权，所以跳过这一步
        if self.weight:
            w = torch.sigmoid(self.w) * (2 / self.n)  # sigmoid weights (0-1)
            x = x * w[0]

        # Fusion，将输出层和前面某些层的输出矩阵进行特征融合（对应维度相加）
        nx = x.shape[1]  # input channels
        for i in range(self.n - 1):
            # 提取前面某一层的输出矩阵并根据是否加权布尔变量加权之
            a = outputs[self.layers[i]] * w[i + 1] if self.weight else outputs[self.layers[i]]  # feature to add
            na = a.shape[1]  # feature channels 提取当前需要融合的张量的深度

            # Adjust channels
            # 根据相加的两个特征矩阵的channel选择相加方式
            if nx == na:  # same shape 如果channel相同，直接相加
                x = x + a
            elif nx > na:  # slice input 如果channel不同，将channel多的特征矩阵砍掉部分channel保证相加的channel一致
                x[:, :na] = x[:, :na] + a  # or a = nn.ZeroPad2d((0, 0, 0, 0, 0, dc))(a); x = x + a
            else:  # slice feature
                x = x + a[:, :nx]

        return x


class ConvBnActivation(nn.Module):
    '''Conv2d层+BN层+激活层组成的模块，可用于实现CBL、CBM等结构'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, pad=0, activation="leaky", bn=True):
        super(ConvBnActivation, self).__init__()

        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding=kernel_size // 2 if pad else 0,
                                   bias=not bn))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == 'mish':
            self.conv.append(nn.Mish(inplace=True))
        elif activation == 'relu':
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == 'leaky':
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'linear':
            pass
        else:
            print("activate error: {} {} {}".format(sys._getframe().f_code.co_filename,
                                                    sys._getframe().f_code.co_name,
                                                    sys._getframe().f_lineno))

    def forward(self, x):
        for m in self.conv:
            x = m(x)
        return x


class ResBlock(nn.Module):
    '''残差模块Residual Block x n，默认使用Mish激活函数'''

    def __init__(self, in_channels, filter_1, out_channels, block_nums=1, activation='mish', shortcut=True):
        super(ResBlock, self).__init__()

        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(block_nums):
            resblock_one = nn.ModuleList()
            resblock_one.append(ConvBnActivation(in_channels, filter_1, 1, 1, 1, activation, True))
            resblock_one.append(ConvBnActivation(filter_1, out_channels, 3, 1, 1, activation, True))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for m in self.module_list:
            front = x
            for res in m:
                x = res(x)
            x = front + x if self.shortcut else x
        return x


class Inception(nn.Module):
    '''Inception结构，共有4条分支，分别为1x1卷积、3x3卷积、5x5卷积和一个池化分支'''

    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(Inception, self).__init__()

        self.branch1 = nn.Sequential(
            ConvBnActivation(in_channels, n1x1, kernel_size=1)
        )
        self.branch2 = nn.Sequential(
            ConvBnActivation(in_channels, n3x3_reduce, kernel_size=1),
            ConvBnActivation(n3x3_reduce, n3x3, kernel_size=3, pad=1)
        )
        self.branch3 = nn.Sequential(
            ConvBnActivation(in_channels, n5x5_reduce, kernel_size=1),
            ConvBnActivation(n5x5_reduce, n5x5, kernel_size=3, pad=1),
            ConvBnActivation(n5x5, n5x5, kernel_size=3, pad=1)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnActivation(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class SqueezeExcitation(nn.Module):
    '''通道注意力机制'''

    def __init__(self, in_channels: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channel = make_divisible(in_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channel, 1)
        self.fc2 = nn.Conv2d(squeeze_channel, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class SEInceptionFusion(nn.Module):
    '''SE-Inception Fusion混合模态特征融合模块'''

    def __init__(self, in_channels, out_channels, layers,
                 inception=False, icp_param_list=(),
                 tmse=False, squeeze_factor=4):
        super(SEInceptionFusion, self).__init__()

        self.concat = FeatureConcat(layers)
        self.enhance = nn.ModuleList()
        self.enhance.append(ConvBnActivation(in_channels, out_channels, kernel_size=1))
        # Inception结构增强网络宽度，执行特征稀疏化，提高特征融合效果
        if inception:
            self.enhance.append(Inception(out_channels, *icp_param_list))
        # SE模块，通道注意力机制
        if tmse:
            self.enhance.append(SqueezeExcitation(out_channels, squeeze_factor))

    def forward(self, x, outputs):
        y = self.concat(x, outputs)
        for m in self.enhance:
            y = m(y)
        return y


class MixConv2d(nn.Module):  # MixConv: Mixed Depthwise Convolutional Kernels https://arxiv.org/abs/1907.09595
    '''
    混合深度卷积（MixConv），即在一个卷积中自然的混合多个不同大小的卷积核。用这种简单的嵌入式替换普通的深度卷积，
    MixConv可以提高了MobileNets在ImageNet的分类任务和COCO的目标检测任务的准确率和效率。
    '''

    def __init__(self, in_ch, out_ch, k=(3, 5, 7), stride=1, dilation=1, bias=True, method='equal_params'):
        super(MixConv2d, self).__init__()

        groups = len(k)
        if method == 'equal_ch':  # equal channels per group
            i = torch.linspace(0, groups - 1E-6, out_ch).floor()  # out_ch indices
            ch = [(i == g).sum() for g in range(groups)]
        else:  # 'equal_params': equal parameter count per group
            b = [out_ch] + [0] * groups
            a = np.eye(groups + 1, groups, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            ch = np.linalg.lstsq(a, b, rcond=None)[0].round().astype(int)  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList([nn.Conv2d(in_channels=in_ch,
                                          out_channels=ch[g],
                                          kernel_size=k[g],
                                          stride=stride,
                                          padding=k[g] // 2,  # 'same' pad
                                          dilation=dilation,
                                          bias=bias) for g in range(groups)])

    def forward(self, x):
        return torch.cat([m(x) for m in self.m], 1)


# Activation functions below -------------------------------------------------------------------------------------------
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)  # sigmoid(ctx)
        return grad_output * (sx * (1 + x * (1 - sx)))


class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.mul(torch.tanh(F.softplus(x)))  # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        sx = torch.sigmoid(x)
        fx = F.softplus(x).tanh()
        return grad_output * (fx + x * sx * (1 - fx * fx))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class MemoryEfficientMish(nn.Module):
    def forward(self, x):
        return MishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HardSwish(nn.Module):  # https://arxiv.org/pdf/1905.02244.pdf
    def forward(self, x):
        return x * F.hardtanh(x + 3, 0., 6., True) / 6.


class Mish(nn.Module):  # https://github.com/digantamisra98/Mish
    def forward(self, x):
        return x * F.softplus(x).tanh()
