from __future__ import division

import os
import random
import argparse
import torch
import torch.optim as optim
from data.custom import FileDetection
from data import detection_collate
import tools
from utils.augmentations import TransformTrain, TransformTest
from utils.modules import ModelEMA
import pandas as pd
import models
import data.utils
from tqdm import tqdm
from test import test
from torch.optim.lr_scheduler import StepLR
import torchvision
from configparser import ConfigParser



def parseargs():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Integer Value - Batch size (must be divisible by 2)')
    parser.add_argument('--base_lr', default=1e-4, type=float,
                        help='Integer Value - initial learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Integer Value - minimum learning rate')
    parser.add_argument('--lr_decay_step', default=5, type=float,
                        help='Integer Value - Number of epoch after which the learning rate is decayed.')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='String Value - Number of workers used in dataloading')
    parser.add_argument('--log_dir', default='log/', type=str,
                        help='String Value - where the training log + the best method weights is are saved.')
    # model
    parser.add_argument('--model_name', default='yolo_v2',
                        help='String Value - yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')

    parser.add_argument("--val-split", type=float,
                        default=0.05,
                        help="Floating Point Value - The percentage of data to be used for validation.")

    return parser.parse_args()


def get_dataloaders(cfg, val_split, input_size, batch_size,num_workers):
    config_parser = ConfigParser(os.environ)
    config_parser.read(cfg)
    train_loaders = []
    val_loaders = []
    for section in config_parser.sections():
        _get_data = data.utils.get_isic \
            if os.path.basename(config_parser[section]["dataset"]) == "ISIC_2019_Training_GroundTruth.csv"\
            else data.utils.get_directory
        train_files, train_labels, _, _, val_files, val_labels = \
            _get_data(config_parser[section]["dataset"], 0, val_split)

        transform = TransformTrain(input_size,
                                   crop_scale=config_parser.getfloat(section, "crop_scale"),
                                   random_shrink_ratio=config_parser.getfloat(section, "random_shrink_ratio"),
                                   random_brightness=config_parser.getfloat(section, "random_brightness"),
                                   random_contrast=config_parser.getfloat(section, "random_contrast"),
                                   random_saturation=config_parser.getfloat(section, "random_saturation"),
                                   random_hue=config_parser.getfloat(section, "random_hue")
                                   )
        dataset_positive_train = FileDetection(name = section,files=train_files, labels=train_labels,
                                               transform=transform)

        dataset_positive_val = FileDetection(name = section, files=val_files, labels=val_labels,
                                             transform=TransformTest(input_size))
        dataloaders = []
        for i, dataset in enumerate([dataset_positive_train, dataset_positive_val]):
            dataloaders.append(torch.utils.data.DataLoader(dataset=dataset, shuffle=i==0,
                                                                    batch_size=batch_size,
                                                                    collate_fn=detection_collate,
                                                                    num_workers=num_workers,
                                                                    pin_memory=True))
        train_loader, val_loader = dataloaders
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders


def train(model_name, log_dir, batch_size, val_split, lr_decay_step,
          base_lr, min_lr, num_workers):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Model: ', model_name)

    # LOAD YOLO NET WITH A GIVEN ARCHITECTURE AND THE BELONGING CFG
    yolo_net = models.model_dict[model_name]
    yolo_net_cfg = models.model_cfg_dict[model_name]

    # CREATE THE LOG DIR IF DOESN'T EXIST
    os.makedirs(log_dir, exist_ok=True)

    # GET DATASETS
    input_size = yolo_net_cfg["size"]
    train_loaders, val_loaders = get_dataloaders("config_datasets.ini",val_split, input_size, batch_size, num_workers)


    epoch_size_train = min([len(loader.dataset) for loader in train_loaders]) // batch_size
    epoch_size_val = min([len(loader.dataset) for loader in val_loaders]) // batch_size
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

    # lr_scheduler = utils.training.StepLRWithWarmUP(optimizer,
    #                                               warmup_size=warmup_size * epoch_size_train,
    #                                               step_size=lr_decay_step * epoch_size_train,
    #                                               min_lr=min_lr,
    #                                               gamma=0.1
    #                                               )
    lr_scheduler = StepLR(optimizer, step_size=lr_decay_step)

    # EMA... whatever it means...
    ema = ModelEMA(model)

    # DO TRAINING
    best_model_file = os.path.join(log_dir, 'best_model.pth')
    df_train = pd.DataFrame()
    df_val = pd.DataFrame()



    for epoch in range(0, yolo_net_cfg["max_epoch"]):
        conf_loss = cls_loss = box_loss = iou_loss = total_loss = 0
        p_bar = tqdm(zip(*train_loaders),
                     total=epoch_size_train,
                     desc=f"Training epoch {epoch}")

        # TRAIN FOR AN EPOCH
        model.set_grid(input_size)
        model.train()

        for iter_i, image_target_list in enumerate(p_bar):
            # if iter_i == 10:
            #    break
            images = torch.cat([image_target_pair[0] for image_target_pair in image_target_list])
            targets = [target for image_target_pair in image_target_list for target in image_target_pair[1]]

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
            total_loss = conf_loss + cls_loss + box_loss + iou_loss
            _total_loss = _conf_loss + _cls_loss + _box_loss + _iou_loss

            # BACKPROP
            _total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # EMA
            ema.update(model)

            # DISPLAY TRAINING INFO
            p_bar.set_postfix({'[Losses -> total': f"{total_loss / (iter_i + 1):.3f}",
                               'conf': f"{conf_loss / (iter_i + 1):.3f}",
                               'cls': f"{cls_loss / (iter_i + 1):.3f}",
                               'box': f"{box_loss / (iter_i + 1):.3f}",
                               'iou': f"{iou_loss / (iter_i + 1):.3f}",
                               '], LR': lr_scheduler.get_lr(),
                               'size': f"{input_size}"})

        lr_scheduler.step()
        # VALIDATE AFTER EPOCH

        model.set_grid(yolo_net_cfg['size'])
        model.eval()

        df_train = df_train.append({'conf loss': conf_loss / epoch_size_train,
                                    'class loss': cls_loss / epoch_size_train,
                                    'box loss': box_loss / epoch_size_train,
                                    'iou loss': iou_loss / epoch_size_train}, ignore_index=True)

        p_bar = tqdm(zip(*val_loaders),
                     total=epoch_size_val,
                     desc=f"Validating after epoch {epoch}")
        # VALIDATE
        conf_loss = cls_loss = box_loss = iou_loss = total_loss = 0
        with torch.no_grad():
            for iter_i, image_target_pair in enumerate(p_bar):
                images = torch.cat([image_target_pair[0] for image_target_pair in image_target_list])
                targets = [target for image_target_pair in image_target_list for target in image_target_pair[1]]

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
                total_loss = conf_loss + cls_loss + box_loss + iou_loss

                # DISPLAY VALIDATION INFO
                p_bar.set_postfix({'[Losses -> total': f"{total_loss / (iter_i + 1):.3f}",
                                   'conf': f"{conf_loss / (iter_i + 1):.3f}",
                                   'cls': f"{cls_loss / (iter_i + 1):.3f}",
                                   'box_loss': f"{box_loss / (iter_i + 1):.3f}",
                                   'iou_loss': f"{iou_loss / (iter_i + 1):.3f}",
                                   '], size': f"{yolo_net_cfg['size']}"})

            df_val = df_val.append({'total loss': total_loss / epoch_size_val,
                                    'conf loss': conf_loss / epoch_size_val,
                                    'class loss': cls_loss / epoch_size_val,
                                    'box loss': box_loss / epoch_size_val,
                                    'iou loss': iou_loss / epoch_size_val}, ignore_index=True)

            # If the current model is so far the best
            if df_val.sum(axis=1).idxmin() == df_val.shape[0] - 1:
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))

    df_train.to_csv(os.path.join(log_dir, 'train_log.csv'))
    df_val.to_csv(os.path.join(log_dir, 'val_log.csv'))
    model.load_state_dict(torch.load(os.path.join(log_dir, 'best_model.pth')))
    model.eval()

    for data_loader in val_loaders:
        test(model, device, data_loader, log_dir=os.path.join(log_dir, data_loader.dataset.name+"_val/"))


if __name__ == '__main__':
    args = parseargs()
    train(**args.__dict__)
