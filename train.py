from __future__ import division

import os
import random
import argparse
import torch
import torch.optim as optim
from data.custom import FileDetection
from data import BaseTransform, detection_collate
import tools
from utils.augmentations import SSDAugmentation, SDAugmentation
from utils.modules import ModelEMA
import pandas as pd
import models
import data.utils
import utils.training
from tqdm import tqdm
from utils.com_paras_flops import FLOPs_and_Params
from test import test


def parseargs():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Integer Value - Batch size (must be divisible by 2)')
    parser.add_argument('--base_lr', default=1e-3, type=float,
                        help='Integer Value - initial learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Integer Value - minimum learning rate')
    parser.add_argument('--lr_decay_step', default=10, type=float,
                        help='Integer Value - Number of epoch after which the learning rate is decayed.')
    parser.add_argument('--warmup_size', type=int, default=2,
                        help='Integer Value - The upper bound of warm-up')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='String Value - Number of workers used in dataloading')
    parser.add_argument('--log_dir', default='log/', type=str,
                        help='String Value - where the training log + the best method weights is are saved.')
    # model
    parser.add_argument('--model_name', default='yolo_v2',
                        help='String Value - yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')

    # dataset
    parser.add_argument("--isic_csv", "-p",
                        type=str,
                        help='String Value - The path to the ISIC dataset csv file.',
                        )
    parser.add_argument("--negative_dir", "-n",
                        type=str,
                        help='String Value - The path to the folder containing only negative examples i.e. healthy skin.',
                        )
    parser.add_argument("--val-split", type=float,
                        default=0.05,
                        help="Floating Point Value - The percentage of data to be used for validation.")

    return parser.parse_args()


def train(model_name, log_dir, negative_dir, isic_csv, batch_size, val_split, warmup_size, lr_decay_step,
          base_lr, min_lr, num_workers):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Model: ', model_name)

    # LOAD YOLO NET WITH A GIVEN ARCHITECTURE AND THE BELONGING CFG
    yolo_net = models.model_dict[model_name]
    yolo_net_cfg = models.model_cfg_dict[model_name]

    # CREATE THE LOG DIR IF DOESN'T EXIST
    os.makedirs(log_dir, exist_ok=True)

    # GET DATASETS

    image_mean = (0.406, 0.456, 0.485)
    image_std = (0.225, 0.224, 0.229)

    input_size = yolo_net_cfg["size"]
    train_files_n, _, val_files_n = data.utils.get_directory(negative_dir, 0, val_split)
    train_files_p, train_labels_p, _, _, val_files_p, val_labels_p = data.utils.get_isic(isic_csv, 0, val_split)

    dataset_positive_train = FileDetection(files=train_files_p, labels=train_labels_p,
                                           transform=SSDAugmentation(input_size, image_mean, image_std))
    dataset_positive_val = FileDetection(files=val_files_p, labels=val_labels_p,
                                         transform=BaseTransform(input_size, image_mean, image_std))

    dataset_negative_train = FileDetection(files=train_files_n, labels=None,
                                           transform=SDAugmentation(input_size, image_mean, image_std))
    dataset_negative_val = FileDetection(files=val_files_n, labels=None,
                                         transform=BaseTransform(input_size, image_mean, image_std))

    # CREATE THE DATALOADERS
    dataloader_positive_train = torch.utils.data.DataLoader(dataset=dataset_positive_train, shuffle=True, batch_size=batch_size,
                                                            collate_fn=detection_collate,
                                                            num_workers=num_workers, pin_memory=True)
    dataloader_positive_val = torch.utils.data.DataLoader(dataset=dataset_positive_val, shuffle=False, batch_size=batch_size,
                                                            collate_fn=detection_collate,
                                                            num_workers=num_workers, pin_memory=True)
    dataloader_negative_train = torch.utils.data.DataLoader(dataset=dataset_negative_train, shuffle=True, batch_size=batch_size,
                                                            collate_fn=detection_collate,
                                                            num_workers=num_workers, pin_memory=True)
    dataloader_negative_val = torch.utils.data.DataLoader(dataset=dataset_negative_val, shuffle=False, batch_size=batch_size,
                                                            collate_fn=detection_collate,
                                                            num_workers=num_workers, pin_memory=True)

    epoch_size = min(len(dataset_positive_train), len(dataset_negative_train))

    # BUILD THE MODEL
    model = yolo_net(device=device,
                     input_size=yolo_net_cfg['size'],
                     num_classes=8,
                     trainable=False,
                     anchor_size=yolo_net_cfg['anchor_size'],
                     hr=False)
    model = model.to(device)
    model.trainable = True

    # CREATE OPTIMIZER

    optimizer = optim.SGD(model.parameters(),
                          lr=base_lr,
                          momentum=0.9,
                          weight_decay=5e-4
                          )

    lr_scheduler = utils.training.StepLRWithWarmUP(optimizer,
                                                   warmup_size=warmup_size * epoch_size,
                                                   step_size=lr_decay_step,
                                                   min_lr=min_lr,
                                                   gamma=0.1
                                                   )

    # EMA... whatever it means...
    ema = ModelEMA(model)

    # DO TRAINING
    best_model_file = os.path.join(log_dir, 'best_model.pth')
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()

    for epoch in range(0, 2):  # yolo_net_cfg["max_epoch"]):
        conf_loss = cls_loss = box_loss = iou_loss = 0
        p_bar = tqdm(zip(dataloader_positive_train, dataloader_negative_train),
                     total=epoch_size,
                     desc=f"Training epoch {epoch}")

        # TRAIN FOR AN EPOCH
        model.set_grid(input_size)
        model.train()
        for iter_i, ((images_p, targets_p), (images_n, targets_n)) in enumerate(p_bar):
            #if iter_i == 10:
            #    break
            lr_scheduler.step()
            images = torch.cat([images_p, images_n])
            targets = targets_p + targets_n
            targets = [label.tolist() for label in targets]
            # multi-scale trick
            if iter_i % 10:
                # randomly choose a new size
                r = yolo_net_cfg['random_size_range']
                input_size = random.randint(r[0], r[1]) * 32
                model.set_grid(input_size)

            # Rescale image to their new resolution.
            images = torch.nn.functional.interpolate(images, size=input_size, mode='bilinear', align_corners=False)

            # Convert the one hot target representation to yolo target.
            targets = tools.gt_creator(model_name=model_name,
                                       input_size=input_size,
                                       stride=model.stride,
                                       label_lists=targets,
                                       anchor_size=yolo_net_cfg["anchor_size"]
                                       )

            # TO DEVICE
            images = images.to(device)
            targets = torch.tensor(targets).float().to(device)

            # FORWARD PASS
            _conf_loss, _cls_loss, _box_loss, _iou_loss = model(images, target=targets)
            conf_loss += _conf_loss.item()
            cls_loss += _cls_loss.item()
            box_loss += _box_loss.item()
            iou_loss += _iou_loss.item()

            # COMPUTE LOSS
            total_loss = _conf_loss + _cls_loss + _box_loss + _iou_loss

            # BACKPROP
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # EMA
            ema.update(model)

            # DISPLAY TRAINING INFO
            p_bar.set_postfix({'[Losses -> total': f"{total_loss.item():.3f}",
                               'conf': f"{_conf_loss.item():.3f}",
                               'cls': f"{_cls_loss.item():.3f}",
                               'box': f"{_box_loss.item():.3f}",
                               'iou': f"{_iou_loss.item():.3f}",
                               '], LR': lr_scheduler.get_lr(),
                               'size': f"{input_size}"})

        # VALIDATE AFTER EPOCH

        model.set_grid(yolo_net_cfg['size'])
        model.eval()

        df_train = df_train.append({'conf loss': conf_loss / epoch_size,
                                    'class loss': cls_loss / epoch_size,
                                    'box loss': box_loss / epoch_size,
                                    'iou loss': iou_loss / epoch_size}, ignore_index=True)

        p_bar = tqdm(zip(dataloader_positive_val, dataloader_negative_val),
                     total=epoch_size,
                     desc=f"Validating after epoch {epoch}")
        # VALIDATE
        conf_loss = cls_loss = box_loss = iou_loss = 0
        with torch.no_grad():
            for iter_i, ((images_p, targets_p), (images_n, targets_n)) in enumerate(p_bar):
                # if iter_i == 10:
                #     break
                images = torch.cat([images_p, images_n])
                targets = targets_p + targets_n
                targets = [label.tolist() for label in targets]
                # Convert the one hot target representation to yolo target.
                targets = tools.gt_creator(model_name=model_name,
                                           input_size=yolo_net_cfg["size"],
                                           stride=model.stride,
                                           label_lists=targets,
                                           anchor_size=yolo_net_cfg["anchor_size"]
                                           )
                targets = torch.tensor(targets).float().to(device)
                # TO DEVICE
                images = images.to(device)
                # FORWARD PASS
                _conf_loss, _cls_loss, _box_loss, _iou_loss = model(images, target=targets)
                conf_loss += _conf_loss.item()
                cls_loss += _cls_loss.item()
                box_loss += _box_loss.item()
                iou_loss += _iou_loss.item()

                # COMPUTE LOSS
                total_loss = _conf_loss + _cls_loss + _box_loss + _iou_loss

                # DISPLAY VALIDATION INFO
                p_bar.set_postfix({'[Losses -> total': f"{total_loss.item():.3f}",
                                   'conf': f"{_conf_loss.item():.3f}",
                                   'cls': f"{_cls_loss.item():.3f}",
                                   'box_loss': f"{_box_loss.item():.3f}",
                                   'iou_loss': f"{_iou_loss.item():.3f}",
                                   '], size': f"{yolo_net_cfg['anchor_size']}"})

            df_val = df_val.append({'conf loss': conf_loss / epoch_size,
                                    'class loss': cls_loss / epoch_size,
                                    'box loss': box_loss / epoch_size,
                                    'iou loss': iou_loss / epoch_size}, ignore_index=True)

            # If the current model is so far the best
            if df_val.sum(axis=1).idxmin() == df_val.shape[0] - 1:
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

    df_train.to_csv(os.path.join(log_dir, 'train_log.csv'))
    df_val.to_csv(os.path.join(log_dir, 'val_log.csv'))
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    model.eval()
    test(model, device, dataloader_positive_val, log_dir=os.path.join(log_dir, "positive_val/"))
    test(model, device, dataloader_negative_val, log_dir=os.path.join(log_dir, "negative_val/"))


if __name__ == '__main__':
    args = parseargs()
    train(**args.__dict__)
