import json
import os
import datetime
import time
import torch
from torch.utils import data
import numpy as np

from core.dataset import transforms
from core.dataset.seal import SealDataset
from core.models.sealnet import u2_hr_backbone
from core.train_utils.train_eval_utils import train_one_epoch, evaluate, get_params_groups, create_lr_scheduler


def create_model(key_joints):
    model = u2_hr_backbone(key_joints=key_joints)
    # if load_pretrain_weights:
    #     weights_dict = torch.load("./pretrained/model_best2.pth", map_location='cpu')["model"]
    #
    #
    #     # 获取你当前模型的 state_dict
    #     model_dict = model.state_dict()
    #     # 过滤出只有 layer1 和 layer2 的权重的预训练字典
    #     pretrained_dict = {k: v for k, v in weights_dict.items() if k in model_dict and "u2net." in k}
    #
    #     model_dict.update(pretrained_dict)
    #     missing_keys, unexpected_keys = model.load_state_dict(model_dict)
    #
    #     if len(missing_keys) != 0:
    #         print(len(missing_keys))
    #         print("missing_keys: ", missing_keys)
    #

    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    with open(args.keypoints_path, "r") as f:
        person_kps_info = json.load(f)

    fixed_size = args.fixed_size
    heatmap_hw = (args.fixed_size[0], args.fixed_size[1])

    kps_weights = np.array(person_kps_info["kps_weights"],
                           dtype=np.float32).reshape((args.key_joints,))
    data_transform = {
        "train": transforms.Compose([
            transforms.ColorAugmentation(),
            transforms.BlurAugmentation(),
            transforms.AffineTransform(scale=(1.25, 1.25), rotation=(-179, 180), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=fixed_size),
            transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=2, keypoints_weights=kps_weights),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

    data_root = args.data_path



    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)



    train_dataset = SealDataset(data_root, True, transforms=data_transform["train"])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        pin_memory=True,
                                        num_workers=nw,
                                        collate_fn=train_dataset.collate_fn)


    val_dataset = SealDataset(data_root, False,transforms=data_transform["val"])
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      pin_memory=True,
                                      num_workers=1,
                                      collate_fn=val_dataset.collate_fn)

    # create model
    model = create_model(key_joints=args.key_joints)
    # model = u2_hr_backbone()
    # print(model)
    for param in model.u2net.parameters():
        param.requires_grad = False
        
    model.to(device)


    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    #

    # define optimizer
    # params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(params,
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    #
    # scaler = torch.cuda.amp.GradScaler() if args.amp else None
    #
    # # learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
    #                                                step_size=3,
    #                                                gamma=0.33)


    # params_group = get_params_groups(model, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(params_group, lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
    #                                    warmup=True, warmup_epochs=2)

    # scaler = torch.cuda.amp.GradScaler() if args.amp else None




    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    current_mae, current_f1 = 1.0, 0.0
    current_5map = 0.0
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 50 iterations
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True,
                                              scaler=scaler)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # evaluate on the test dataset
            coco_info,mae_info, f1_info = evaluate(model, val_data_loader, device=device,
                                       flip=True, flip_pairs=person_kps_info["flip_pairs"])

            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            # # write into txt
            # with open(results_file, "a") as f:
            #     # 写入的数据包括coco指标还有loss和learning rate
            #     result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            #     txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            #     f.write(txt + "\n")
            #
            val_map.append(coco_info[1])  # @0.5 mAP

            # save_best
            if current_mae >= mae_info and current_f1 < f1_info and current_5map < coco_info[1]:
                current_f1 = f1_info
                current_5map = coco_info[1]
                torch.save(save_files, "save_weights/model_best.pth")

            # # 0.5 map
            # if current_5map <= coco_info[1]:
            #     current_5map = coco_info[1]
            #     torch.save(save_files, "save_weights/model_5_best.pth")


        # only save latest 20 epoch weights
        if os.path.exists(f"save_weights/model_{epoch-20}.pth"):
            os.remove(f"save_weights/model_{epoch-20}.pth")

        torch.save(save_files, f"save_weights/model_{epoch}.pth")

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # parser.add_argument('--device', default='cpu', help='device')
    # 训练数据集的根目录(coco2017)
    # parser.add_argument('--data-path', default='../data_set/coco2017', help='dataset')
    # parser.add_argument('--data-path', default='coco_seal_mini', help='dataset')
    # parser.add_argument('--data-path', default='single_seal', help='dataset')
    parser.add_argument('--data-path', default='seal_dataset', help='dataset')
    # parser.add_argument('--data-path', default='mini_dataset', help='dataset')
    # COCO数据集人体关键点信息
    # parser.add_argument('--keypoints-path', default="./seal_keypoints.json", type=str,
    #                     help='person_keypoints.json path')
    parser.add_argument('--keypoints-path', default="./seal_keypoints2.json", type=str,
                        help='person_keypoints.json path')
    # parser.add_argument('--keypoints-path', default="./person_keypoints.json", type=str,
    #                     help='person_keypoints.json path')
    # 原项目提供的验证集person检测信息，如果要使用GT信息，直接将该参数置为None，建议设置成None
    parser.add_argument('--person-det', type=str, default=None)
    parser.add_argument('--fixed-size', default=[256, 256], nargs='+', type=int, help='input size')
    # keypoints点数
    parser.add_argument('--key-joints', default=10, type=int, help='key_joints')
    # parser.add_argument('--key-joints', default=17, type=int, help='key_joints')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # parser.add_argument('--resume', default='./save_weights/model_42.pth', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-steps', default=[90, 150, 200], nargs='+', type=int,
                        help='decrease lr every step-size epochs')
    # 针对torch.optim.lr_scheduler.MultiStepLR的参数
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    # 学习率
    parser.add_argument('--lr', default=0.001, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # AdamW的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                        help='batch size when training.')
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 10 Epochs")
    parser.add_argument("--eval-interval", default=1, type=int, help="validation interval default 10 Epochs")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
