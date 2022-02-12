from torch.utils.tensorboard import SummaryWriter
from build_utils.utils import check_file
from train_utils import kaist_train_eval_utils as train_util
from train_utils import get_coco_api_from_dataset
from build_utils.torch_utils import select_device
from build_utils.kaist_dataset import *
from models import *

import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
import datetime
import argparse
import yaml
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def train(hyp):
    # 1、选择训练设备信息
    device = select_device(opt.device)

    # 2、设置训练结果文件的相关路径
    weight_best_file = "weights/{}_best.pt".format(opt.name)
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Remove previous results 移除先前的训练结果记录指标文件
    # glob的主要作用是用来查找符合特定规则的文件路径名
    # for f in glob.glob(results_file):
    #     os.remove(f)

    # 3、初始化训练设置信息
    cfg = opt.cfg
    data = opt.data
    epochs = opt.epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    weights = opt.weights  # initial training weights
    imgsz_train = opt.img_size  # 训练时输入图像大小
    imgsz_test = opt.img_size  # test image sizes
    multi_scale = opt.multi_scale

    # 4、设置多尺度训练相关的参数
    # 图像要设置成32的倍数，我们默认的输出图像大小512是32的倍数
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_test, gs) == 0, "--img-size {} must be a {}-multiple".format(imgsz_test, gs)
    grid_min, grid_max = imgsz_test // gs, imgsz_test // gs  # 计算网格数
    if multi_scale:  # 若使用多尺度训练（即输入图像大小不一）
        imgsz_min = opt.img_size // 1.5
        imgsz_max = opt.img_size // 0.667

        # 将给定的最大，最小输入尺寸向下调整到32的整数倍
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs  # 网格最小、最大尺寸
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)  # 图像最小、最大尺寸
        imgsz_train = imgsz_max  # initialize with max size
        print("Using multi_scale training, image range({}, {})".format(imgsz_min, imgsz_max))

    # 5、设置训练相关参数
    # init_seeds()  # 初始化随机种子，保证结果可复现
    # 解析xx.data配置文件，从中提取训练图像文件路径、验证图像文件路径和数据集类别数
    data_dict = parse_data_cfg(data)
    train_path = data_dict["train"]
    test_path = data_dict["valid"]
    nc = 1 if opt.single_cls else int(data_dict["classes"])  # number of classes

    # 下面几个损失函数权重系数的调参挺有用的
    hyp["cls"] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset
    hyp["obj"] *= imgsz_test / 320 * opt.obj_gain  # TODO: 置信度损失权重可能越大越好？
    print(f"hyp['box']: {hyp['box']:0.3f}, hyp['obj']: {hyp['obj']:0.3f}. hyp['cls']: {hyp['cls']:0.3f},"
          f" {('CIoU Loss' if 'ciou' in hyp else 'GIoU Loss')}")

    # 6、创建网络模型对象，冻结部分网络结构的权重参数
    model = YOLO(cfg).to(device)
    if opt.freeze_layers >= 0:
        # 将Double-YOLO-Kaist的前149层（即Darknet特征提取网络部分）的参数冻结
        darknet_end_layers = opt.freeze_layers  # 对于dyolov3而言是默认是149
        for idx in range(darknet_end_layers + 1):
            for parameter in model.module_list[idx].parameters():
                parameter.requires_grad_(False)

    # 7、创建优化器
    pg = [p for p in model.parameters() if p.requires_grad]
    if opt.sgd:
        optimizer = optim.SGD(pg, lr=hyp["lr0"], momentum=hyp["momentum"],
                              weight_decay=hyp["weight_decay"], nesterov=True)
    else:
        optimizer = optim.Adam(pg, lr=hyp["lr0"], betas=(hyp['momentum'], 0.999),
                               weight_decay=hyp['weight_decay'])

    # 8、加载网络权重，使用权重文件中记录的数据初始化相关变量
    start_epoch = 0
    best_map = 0.0
    if weights.endswith(".pt") or weights.endswith(".pth"):
        print("load dict model weights from '{}'".format(weights))
        ckpt = torch.load(weights, map_location=device)

        # 尝试加载网络模型权重参数
        try:
            ckpt["model"] = {k: v for k, v in ckpt["model"].items()
                             if k in model.state_dict() and model.state_dict()[k].numel() == v.numel()}
            miss, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        except KeyError as e:
            s = "{} is not compatible with {}. Specify --weights '' or specify a --cfg compatible with {}. " \
                .format(opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # 尝试加载训练时所有的优化器参数
        if ckpt["optimizer"] is not None:
            # optimizer.load_state_dict(ckpt["optimizer"])
            if "best_map" in ckpt.keys():
                best_map = ckpt["best_map"]

        # 尝试加载先前训练得到的结果，并将其写入到结果文本文件中
        if ckpt.get("training_results") is not None:
            with open(results_file, "w") as file:
                file.write(ckpt["training_results"])  # write results.txt

        # 获取先前训练到的轮次数，方便后续继续训练
        start_epoch = ckpt["epoch"] + 1
        if epochs < start_epoch:
            print('{} has been trained for {} epochs. Fine-tuning for {} additional epochs.'
                  .format(opt.weights, ckpt['epoch'], epochs))
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt
    elif weights.endswith(".weights"):
        print("load binary model weights from '{}'".format(weights))
        load_darknet_weights(model, weights, cutoff=opt.cutoff)

    # 9、创建学习率自动调整器并作出相关初始化
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - hyp["lrf"]) + hyp["lrf"]  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch  # 指定从哪个epoch开始

    # 10、加载训练数据集和验证数据集，并初始化加载器
    # 训练集的图像尺寸指定为multi_scale_range中最大的尺寸
    train_dataset = LoadKaistImagesAndLabels(train_path, imgsz_train, batch_size,
                                             augment=True,
                                             hyp=hyp,  # augmentation hyperparameters
                                             rect=opt.rect,  # rectangular training 默认为False
                                             snowflake=opt.snow,
                                             single_cls=opt.single_cls)
    # 验证集的图像尺寸指定为img_size(512)
    val_dataset = LoadKaistImagesAndLabels(test_path, imgsz_test, batch_size,
                                           hyp=hyp,
                                           rect=True,
                                           snowflake=False,
                                           single_cls=opt.single_cls)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=not opt.rect,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    val_datasetloader = torch.utils.data.DataLoader(val_dataset,
                                                    batch_size=batch_size,
                                                    num_workers=nw,
                                                    pin_memory=True,
                                                    collate_fn=val_dataset.collate_fn)

    # 11、设置模型对象的其他剩余成员变量
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)
    coco = get_coco_api_from_dataset(val_dataset)

    # 12、根据剩余训练轮次数继续或者开始网络模型对象的训练
    left_epoches = epochs - start_epoch
    print("starting training for {} epochs, left {} epoches...".format(epochs, left_epoches))
    print('Using {} dataloader workers'.format(nc))
    for epoch in range(start_epoch, epochs):
        # 训练一个轮次，并从中获取训练过程中计算得到的平均损失值，和当前学习率
        mloss, lr = train_util.train_one_epoch(model, optimizer, train_dataloader,
                                               device, epoch,
                                               accumulate=accumulate,  # 迭代多少batch才训练完64张图片
                                               img_size=imgsz_train,  # 输入图像的大小
                                               multi_scale=multi_scale,
                                               grid_min=grid_min,  # grid的最小尺寸
                                               grid_max=grid_max,  # grid的最大尺寸
                                               gs=gs,  # grid step: 32
                                               print_freq=50,  # 每训练多少个step打印一次信息
                                               warmup=True)
        # update scheduler
        scheduler.step()

        # 13、对网络进行验证测试，并记录coco性能指标
        if opt.notest is False or epoch == epochs - 1:
            # evaluate on the test dataset
            result_info = train_util.evaluate(model, val_datasetloader,
                                              coco=coco, device=device)

            coco_mAP = result_info[0]  # (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]
            voc_mAP = result_info[1]  # (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]
            coco_mAR = result_info[8]  # (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]

            # 将测试得到的性能指标数据记录到tensorboard中
            if tb_writer:
                tags = ['train/box_loss', 'train/obj_loss', 'train/cls_loss', 'train/loss', "learning_rate",
                        "mAP@[IoU=0.50:0.95]", "mAP@[IoU=0.5]", "mAR@[IoU=0.50:0.95]"]

                for x, tag in zip(mloss.tolist() + [lr, coco_mAP, voc_mAP, coco_mAR], tags):
                    tb_writer.add_scalar(tag, x, epoch)

            # 将训练得到的性能指标数据记录到结果文本文件中
            with open(results_file, "a") as f:
                # 记录coco的12个指标加上训练总损失和lr
                result_info = [str(round(i, 4)) for i in result_info + [mloss.tolist()[-1]]] + [str(round(lr, 6))]
                txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
                f.write(txt + "\n")

            # 14、如果当前训练得到的网络模型性能最佳，那么就将网络模型的权重参数记录到pt文件中
            # update best mAP(IoU=0.50:0.95)
            if coco_mAP > best_map:
                best_map = coco_mAP

            if not opt.save_best:  # 每轮次都记录一下网络模型权重参数
                with open(results_file, 'r') as f:
                    save_files = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'training_results': f.read(),
                        'epoch': epoch,
                        'best_map': best_map}
                    torch.save(save_files, "./weights/{}-{}.pt".format(opt.name, epoch))
            else:  # 只记录最佳性能指标时的网络模型权重参数
                if best_map == coco_mAP:
                    with open(results_file, 'r') as f:
                        save_files = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'training_results': f.read(),
                            'epoch': epoch,
                            'best_map': best_map}
                        torch.save(save_files, weight_best_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 下面几个参数是我们重点需要配置的参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--hyp', type=str, default='config/hyp.scratch.4.yaml', help='hyperparameters path')
    parser.add_argument('--cfg', type=str, default='config/kaist_dyolov4_add_sl.cfg', help="*.cfg path")
    parser.add_argument('--weights', type=str, default='weights/pretrained_dyolov4.pt', help='initial weights path')
    parser.add_argument('--name', default='kaist_dyolov4_add_sl', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--freeze-layers', type=int, default=209,
                        help='Freeze feature extract layers, -1 means no layers will be froze')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    # 临时启用的程序参数
    parser.add_argument('--cutoff', type=int, default=104, help="model weights cutoff")
    parser.add_argument('--obj-gain', type=int, default=1, help="object loss gain")
    parser.add_argument('--snow', action='store_true', help='use snowflake change to process images')

    # 下面几个参数几乎不需要改动
    parser.add_argument('--sgd', action='store_true', help='use torch.optim.SGD() optimizer')
    parser.add_argument('--single-cls', type=bool, default=True, help='train as single-class dataset')
    parser.add_argument('--data', type=str, default='data/kaist_data.data', help='*.data path')
    parser.add_argument('--multi-scale', type=bool, default=True, help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', type=int, default=512, help='test size')
    parser.add_argument('--rect', action='store_true', help='rectangular training')  # 不要开启矩形变换，因为矩形变换的代码有错误
    parser.add_argument('--save-best', type=bool, default=True, help='only save best checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    opt = parser.parse_args()

    # 检查文件是否存在
    opt.cfg = check_file(opt.cfg)
    opt.data = check_file(opt.data)
    opt.hyp = check_file(opt.hyp)
    print(opt)

    with open(opt.hyp, encoding='utf-8') as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter(comment=opt.name)
    train(hyp)
