import math
import sys
import time

import torch


from .coco_eval import EvalCOCOMetric
from .coco_utils import get_coco_api_from_dataset, _get_iou_types
from .distributed_utils import MetricLogger, SmoothedValue, warmup_lr_scheduler, reduce_dict, is_main_process
from .loss import KpLoss
from ..dataset import transforms
from ..metric.metric import MeanAbsoluteError, F1Score


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)


    mloss = torch.zeros(1).to(device)  # mean losses
    for i, [instances, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        instances = {k: v.to(device) for k, v in instances.items()}

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(instances,targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced,**loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device, flip=False, flip_pairs=None):
    if flip:
        assert flip_pairs is not None, "enable flip must provide flip_pairs."

    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    mae_metric = MeanAbsoluteError()
    f1_metric = F1Score()
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    key_metric = EvalCOCOMetric(coco,iou_types , "key_results.json")

    for instances, targets in metric_logger.log_every(data_loader, 100, header):
        instances = {k: v.to(device) for k, v in instances.items()}

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        pred_images, outputs = model(instances,targets)
        model_time = time.time() - model_time


        mae_metric.update(pred_images, instances['masked_image'])
        f1_metric.update(pred_images, instances['masked_image'])


        key_metric.update(targets, outputs)
        metric_logger.update(model_time=model_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)


    # 从所以的线程中汇总结果
    mae_metric.gather_from_all_processes()
    f1_metric.reduce_from_all_processes()
    mae_info, f1_info = mae_metric.compute(), f1_metric.compute()

    # 同步所有进程中的数据
    key_metric.synchronize_results()

    if is_main_process():
        coco_info = key_metric.evaluate()
    else:
        coco_info = None
    return coco_info,mae_info, f1_info


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-4):
    params_group = [{"params": [], "weight_decay": 0.},  # no decay
                    {"params": [], "weight_decay": weight_decay}]  # with decay

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            # bn:(weight,bias)  conv2d:(bias)  linear:(bias)
            params_group[0]["params"].append(param)  # no decay
        else:
            params_group[1]["params"].append(param)  # with decay

    return params_group

