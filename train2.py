from __future__ import division

import os
import random
import argparse
import time
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc0712 import VOCDetection
from data.coco2017 import COCODataset
from data.custom import FileDetection
from data import config
from data import BaseTransform, detection_collate
from PIL import Image, ImageDraw
import tools

from utils import distributed_utils
from utils.com_paras_flops import FLOPs_and_Params
from utils.augmentations import SSDAugmentation, SDAugmentation, ColorAugmentation, SSDAugmentationTest
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA
from torchvision import transforms
from glob import glob
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize target.')

    # model
    parser.add_argument('-v', '--version', default='yolo_v2',
                        help='yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')
    
    # dataset
    parser.add_argument('-root', '--data_root', default='/media/stamas01/Data/datasets/VOC/',
                        help='dataset root')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    
    # train trick
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=False,
                        help='use hi-res pre-trained backbone.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()

def denormalize(img, mean, std):
    #for ImageNet the mean and std are:
    #mean = np.asarray([ 0.485, 0.456, 0.406 ])
    #std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    return (res)

def get_directory(negative_dir, test_split, val_split):
    result = [y for x in os.walk(negative_dir) for y in glob(os.path.join(x[0], '*.jpg'))]
    df = pd.DataFrame({"images": result})
    train_df, test_df, val_df = slit_data(df, test_split, val_split)
    return train_df["images"].tolist(), test_df["images"].tolist(), val_df["images"].tolist()


def get_isic(isic2019csv, test_split, val_split):
    df = pd.read_csv(os.path.join(isic2019csv))
    train_df, test_df, val_df = slit_data(df, test_split, val_split)
    image_dir = os.path.join(os.path.dirname(isic2019csv), "ISIC_2019_Training_Input")
    train_files = [os.path.join(image_dir, f + ".jpg") for f in train_df.image]
    train_labels = np.argmax(train_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    val_files = [os.path.join(image_dir, f + ".jpg") for f in val_df.image]
    val_labels = np.argmax(val_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    test_files = [os.path.join(image_dir, f + ".jpg") for f in test_df.image]
    test_labels = np.argmax(test_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    return train_files, train_labels, val_files, val_labels, test_files, test_labels

def slit_data(df, test_split, val_split, seed=7):
    indices = np.array(range(df.shape[0]))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_point_1 = int(indices.shape[0] * test_split)
    split_point_2 = int(indices.shape[0] * (val_split + test_split))
    test_indices = indices[0:split_point_1]
    val_indices = indices[split_point_1:split_point_2]
    train_indices = indices[split_point_2::]
    train_df = df.take(train_indices)
    test_df = df.take(test_indices)
    val_df = df.take(val_indices)
    return train_df, test_df, val_df

def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov2_d19':
        from models.yolov2_d19 import YOLOv2D19 as yolo_net
        cfg = config.yolov2_d19_cfg

    elif model_name == 'yolov2_r50':
        from models.yolov2_r50 import YOLOv2R50 as yolo_net
        cfg = config.yolov2_r50_cfg

    elif model_name == 'yolov2_slim':
        from models.yolov2_slim import YOLOv2Slim as yolo_net
        cfg = config.yolov2_slim_cfg

    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3 as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_spp':
        from models.yolov3_spp import YOLOv3Spp as yolo_net
        cfg = config.yolov3_d53_cfg

    elif model_name == 'yolov3_tiny':
        from models.yolov3_tiny import YOLOv3tiny as yolo_net
        cfg = config.yolov3_tiny_cfg
    else:
        print('Unknown model name...')
        exit(0)

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = cfg['train_size']
        val_size = cfg['val_size']
    else:
        train_size = val_size = cfg['train_size']

    # Model ENA
    if args.ema:
        print('use EMA trick ...')

    # dataset and evaluator

    data_dir = os.path.join(args.data_root)
    num_classes = 2

    train_files_n, val_files_n, test_files_n = get_directory("/media/stamas01/Data/datasets/SD-198/260/SD-260/", 0.03, 0.03)
    test_files_nhs, _, _ = get_directory("/media/stamas01/Data/datasets/NHS/data/", 0,0)
    train_files_p, train_labels_p, val_files_p, val_labels_p, test_files_p, test_labels_p = get_isic("/media/stamas01/Data/datasets/ISIC_2019/ISIC_2019_Training_GroundTruth.csv", 0.03, 0.03)


    dataset_positive_train = FileDetection(files=train_files_p, labels=train_labels_p, transform=SSDAugmentation(train_size))
    dataset_positive_val = FileDetection(files=val_files_p, labels=val_labels_p, transform=SSDAugmentation(val_size))

    dataset_nhs_val = FileDetection(files=test_files_nhs, labels=None, transform=BaseTransform(val_size))

    dataset_negative_train = FileDetection(files=train_files_n, labels=None, transform=SDAugmentation(train_size))
    dataset_negative_val = FileDetection(files=val_files_n, labels=None, transform=BaseTransform(val_size))

    evaluator = None


    # build model
    anchor_size = cfg['anchor_size_skin']
    net = yolo_net(device=device, 
                   input_size=train_size, 
                   num_classes=num_classes, 
                   trainable=False,
                   anchor_size=anchor_size, 
                   hr=hr)
    model = net

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device)
    # compute FLOPs and Params
    FLOPs_and_Params(model=model, size=train_size)


    model = model.train().to(device)
    # dataloader

    dataloader_nhs_val = torch.utils.data.DataLoader(
                    dataset=dataset_nhs_val,
                    shuffle=True,
                    batch_size=args.batch_size,
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )
    dataloader_positive_train = torch.utils.data.DataLoader(
                    dataset=dataset_positive_train,
                    shuffle=True,
                    batch_size=args.batch_size,
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )

    dataloader_positive_val = torch.utils.data.DataLoader(
                    dataset=dataset_positive_val,
                    shuffle=True,
                    batch_size=args.batch_size,
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )

    dataloader_negative_train = torch.utils.data.DataLoader(
                    dataset=dataset_negative_train,
                    shuffle=False,
                    batch_size=args.batch_size,
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )


    dataloader_negative_val = torch.utils.data.DataLoader(
                    dataset=dataset_negative_val,
                    shuffle=False,
                    batch_size=args.batch_size,
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )




    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=base_lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    batch_size = args.batch_size
    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset_positive_train) // (batch_size * args.num_gpu)

    best_map = -100.

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):

        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)


        for iter_i, ((images_p, targets_p), (images_n, targets_n)) in enumerate(zip(dataloader_positive_train, dataloader_negative_train)):
            # WarmUp strategy for learning rate
            if iter_i == 100:
                break
            if epoch < args.wp_epoch:
                tmp_lr = base_lr * pow((iter_i + epoch * epoch_size) * 1. / (args.wp_epoch * epoch_size), 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0:
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)


            # multi-scale trick
            images = torch.cat([images_p, images_n])
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = cfg['random_size_range']
                train_size = random.randint(r[0], r[1]) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)

            targets = targets_p+targets_n
            targets = [label.tolist() for label in targets]
            # visualize labels
            if args.vis:
                vis_data(images, targets, train_size)
                continue
            # make labels
            if model_name == 'yolov2_d19' or model_name == 'yolov2_r50' or model_name == 'yolov2_slim':
                targets = tools.gt_creator(input_size=train_size, 
                                           stride=net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )
            else:
                targets = tools.multi_gt_creator(input_size=train_size, 
                                                 strides=net.stride, 
                                                 label_lists=targets, 
                                                 anchor_size=anchor_size
                                                 )

            # to device
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # forward
            conf_loss, cls_loss, box_loss, iou_loss = model(images, target=targets)

            # compute loss
            total_loss = conf_loss + cls_loss + box_loss + iou_loss

            loss_dict = dict(conf_loss=conf_loss,
                             cls_loss=cls_loss,
                             box_loss=box_loss,
                             iou_loss=iou_loss,
                             total_loss=total_loss
                            )

            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            # check NAN for loss

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('conf loss',  loss_dict_reduced['conf_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('cls loss',  loss_dict_reduced['cls_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('box loss',  loss_dict_reduced['box_loss'].item(),  iter_i + epoch * epoch_size)
                    tblogger.add_scalar('iou loss',  loss_dict_reduced['iou_loss'].item(),  iter_i + epoch * epoch_size)
                
                t1 = time.time()
                outstream = ('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                        '[Loss: conf %.2f || cls %.2f || box %.2f || iou %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict_reduced['conf_loss'].item(),
                           loss_dict_reduced['cls_loss'].item(), 
                           loss_dict_reduced['box_loss'].item(),
                           loss_dict_reduced['iou_loss'].item(),
                           train_size, 
                           t1-t0))

                print(outstream, flush=True)

                t0 = time.time()

        # evaluation

        if args.ema:
            model_eval = ema.ema
        else:
            model_eval = model.module if args.distributed else model

        # set eval mode
        model_eval.trainable = False
        model_eval.set_grid(val_size)
        model_eval.eval()
        filename = 0

        for iter_i, ((images_p, targets_p), (images_n, targets_n), (images_nhs, targets_nhs)) in enumerate(zip(dataloader_positive_val, dataloader_negative_val, dataloader_nhs_val)):
            for images, tag in zip([images_p, images_n, images_nhs], ["pos", "neg", "nhs"]):
                images = images.to(device)
                bboxes, scores, cls_inds = model_eval(images)

                if len(bboxes) == 0:
                    continue
                img = images[0]
                img = denormalize(img, np.array([0.406, 0.456, 0.485]), np.array([0.225, 0.224, 0.229]))
                d = bboxes
                ind = np.argmax(scores)
                d = d[ind]
                #x_min = img.shape[2]*d[0]
                #x_max = img.shape[2]*d[2]
                #y_min = img.shape[2]*d[1]
                #y_max = img.shape[2]*d[3]

                x_min = img.shape[2]*d[0]
                x_max = img.shape[2]*d[2]
                y_min = img.shape[2]*d[1]
                y_max = img.shape[2]*d[3]


                img = img.detach().cpu().numpy()
                img = np.swapaxes(img, 0, 2)

                #img = cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                #cv2.imshow('gt', img)
                #cv2.waitKey(0)


                img = Image.fromarray(np.uint8(img * 256))
                draw = ImageDraw.Draw(img)
                draw.rectangle((y_min,x_min,   y_max,x_max), outline="blue", width=5)
                img.save("see/" + str(filename) + "_" + tag + ".jpg")
                filename += 1

                # wait for all processes to synchronize
                if args.distributed:
                    dist.barrier()

        # set train mode.
        model_eval.trainable = True
        model_eval.set_grid(train_size)
        model_eval.train()
    
    if args.tfboard:
        tblogger.close()


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    img = img.copy()

    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
