from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
from other_utils.metrics import compute_ap_lamr
from build_utils.utils import *
from torch.cuda import amp

import train_utils.distributed_utils as utils
import torch.nn.functional as F
import sys


def train_one_epoch(model,  # 网络模型对象
                    optimizer,  # 优化器
                    dataloader,  # 数据加载器
                    device,  # 所用计算设备
                    epoch,  # 训练轮次数
                    print_freq,  # 日志信息输出频率
                    accumulate,  # 每隔多少批次通过优化器更新一下网络权重
                    img_size,  # 图像大小
                    grid_min,  # 网格最小长度
                    grid_max,  # 网格最大长度
                    gs,  # 图像缩放比例，默认32
                    multi_scale=False,  # 是否开启多尺度训练
                    warmup=False):  # 是否预热学习率自动调整器
    # 1、开启模型的训练模式，并设置性能指标记录器
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = 'Epoch: [{}]'.format(epoch)  # 当前为第几轮

    # 2、设置学习率自动调整器
    lr_scheduler = None
    if epoch == 0 and warmup:  # 第一次训练也称为热身训练
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        accumulate = 1  # 第一次训练的时候每隔一批次就更新下网络权重参数

    # 3、检查是否开启混合精度训练，并设置一些训练时需要使用到的参数。
    # 其中混合精度预示着有不止一种精度的Tensor，而自动预示着Tensor的dtype类型会自动变化，也就是框架
    # 按需自动调整tensor的dtype。具体可以参考如下这篇文章：https://zhuanlan.zhihu.com/p/165152789
    enable_amp = "cuda" in device.type
    scaler = amp.GradScaler(enabled=enable_amp)

    mloss = torch.zeros(4).to(device)  # 平均损失：边界框损失，置信度损失、类别损失以及总损失？
    nb = len(dataloader)  # 批次总数
    now_lr = 0.  # 当前学习率，不过这里的0不是最初的学习率，最初的学习率由超参数决定

    # 4、通过数据加载器遍历所有训练集进行训练反向传播优化，并在测试集上进行性能测试
    for i, (v_imgs, l_imgs, targets, paths, _, _) in \
            enumerate(metric_logger.log_every(dataloader, print_freq, header)):
        ni = i + nb * epoch
        v_imgs = v_imgs.to(device).float() / 255.0
        l_imgs = l_imgs.to(device).float() / 255.0
        targets = targets.to(device)

        # 5、根据是否开启多尺度训练的情况，在开启的情况下执行一次随机缩放
        if multi_scale:
            # 每隔accumulate批次就调整下图像的缩放尺度，提高训练效果
            if ni % accumulate == 0:
                img_size = random.randrange(grid_min, grid_max + 1) * gs
            assert v_imgs.shape[:2] == l_imgs.shape[:2]
            sf = img_size / max(v_imgs.shape[2:])

            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in v_imgs.shape[2:]]
                # 利用插值方法，对输入的张量数组进行上\下采样操作，使得科学合理地改变数组的尺寸大小，
                # 尽量保持数据完整。显然这里是为了实现图像的缩放（下采样）
                v_imgs = F.interpolate(v_imgs, size=ns, mode='bilinear', align_corners=False)
                l_imgs = F.interpolate(l_imgs, size=ns, mode='bilinear', align_corners=False)

        # 6、执行正向传播，并计算相应的损失函数
        with amp.autocast(enabled=enable_amp):
            pred = model(v_imgs, l_imgs)  # 正向传播

            # 计算损失函数
            loss_dict = compute_loss(pred, targets, model)
            losses = sum(loss for loss in loss_dict.values())

            # 这里的reduce实际上就是map-reduce，显然这是用多GPU场景下才会使用。正常情况下就是直接返回当前输入的字典
            loss_dict_reduced = utils.reduce_dict(loss_dict)  # 收集其他进程计算的结果
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            loss_items = torch.cat((loss_dict_reduced["box_loss"],
                                    loss_dict_reduced["obj_loss"],
                                    loss_dict_reduced["class_loss"],
                                    losses_reduced)).detach()
            # loss_items =torch.cat((loss_dict['box_loss'], # 其实可以写成如下的形式（在单GPU的情况下替换）
            #                        loss_dict['obj_loss'],
            #                        loss_dict['class_loss'],
            #                        losses))

            mloss = (mloss * i + loss_items) / (i + 1)  # 计算到目前位置的总平均损失值

            if not torch.isfinite(losses_reduced):  # 检查当前总损失值是否无穷大
                print('WARNING: non-finite loss, ending training ', loss_dict_reduced)
                print("training image path: {}".format(",".join(paths)))
                sys.exit(1)

            losses *= 1. / accumulate  # scale loss

        # 7、进行反向传播，并更新相应的权重参数
        scaler.scale(losses).backward()
        # 每训练64张图片更新一次权重
        if ni % accumulate == 0:
            scaler.step(optimizer)  # 通过优化器更新网络权重参数
            scaler.update()
            optimizer.zero_grad()  # 将梯度置零

        # 8、使用性能指标记录器记录最新一次的测试性能指标
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

        if ni % accumulate == 0 and lr_scheduler is not None:
            lr_scheduler.step()

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, dataloader, coco=None, device=None):
    # 1、将模型设置为验证模式，并设置相应的性能指标记录器
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    # 2、设置cocotools工具
    if coco is None:  # coco还是静态对象？
        coco = get_coco_api_from_dataset(dataloader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    preds = []

    # 3、通过数据加载器将验证集数据送入模型进行测试
    for v_imgs, l_imgs, targets, paths, shapes, img_index in metric_logger.log_every(dataloader, 100, header):
        v_imgs = v_imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        l_imgs = l_imgs.to(device).float() / 255.0
        # targets = targets.to(device)

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        # 4、进行正向传播计算
        model_time = time.time()
        pred = model(v_imgs, l_imgs)[0]  # only get inference result
        pred = non_max_suppression(pred, conf_thres=0.01, iou_thres=0.6, multi_label=False)
        model_time = time.time() - model_time

        # 5、计算相应测试数据？
        outputs = []
        for index, p in enumerate(pred):
            if p is None:
                p = torch.empty((0, 6), device=cpu_device)
                boxes = torch.empty((0, 4), device=cpu_device)
            else:
                # xmin, ymin, xmax, ymax
                boxes = p[:, :4]
                # shapes: (h0, w0), ((h / h0, w / w0), pad)
                # 将boxes信息还原回原图尺度，这样计算的mAP才是准确的
                boxes = scale_coords(v_imgs[index].shape[1:], boxes, shapes[index][0]).round()

            # 注意这里传入的boxes格式必须是xmin, ymin, xmax, ymax，且为绝对坐标
            info = {"boxes": boxes.to(cpu_device),
                    "labels": p[:, 5].to(device=cpu_device, dtype=torch.int64),
                    "scores": p[:, 4].to(cpu_device)}
            outputs.append(info)

        res = {img_id: output for img_id, output in zip(img_index, outputs)}

        # 6、向性能指标记录器记录最新测试数据
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # 7、计算compute_ap_lamr()函数所需要的预测边界框字典数据，并将其加入到列表preds中
        for img_id, output in zip(img_index, outputs):
            for i in range(output['labels'].shape[0]):
                info = dict()
                info['img_id'] = img_id
                info['conf'] = output['scores'][i].item()
                info['bbox'] = output['boxes'][i].numpy()
                preds.append(info)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    result_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    # compute_ap_lamr()计算VOC数据集AP性能指标和行人检测模型常用的FPPI-MR性能指标
    preds.sort(key=lambda x: float(x['conf']), reverse=True)
    saved_dict = compute_ap_lamr(preds, dataloader.dataset.labels, dataloader.dataset.shapes)
    print(f"VOC Average Precision (VOC-AP)@[IoU = 0.5] = {(saved_dict['ap'] * 100):.2f}%")
    print(f"Log Average Miss Rate (LAMR)@[IoU = 0.5] = {(saved_dict['lamr'] * 100):.2f}%")

    return result_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
