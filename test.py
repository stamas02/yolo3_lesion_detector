import os
import argparse
import torch
from data.custom import FileDetection
from data import BaseTransform, detection_collate
from utils.augmentations import SDAugmentation
import models
import data.utils
from tqdm import tqdm
from utils.visualization import viz_annotation
import numpy as np
image_mean = (0.406, 0.456, 0.485)
image_std = (0.225, 0.224, 0.229)


def parseargs():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic

    parser.add_argument('--model_file', default=1, type=str,
                        help='String Value - Path to the saved model file')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='String Value - Number of workers used in dataloading')
    parser.add_argument('--log_dir', default='log/', type=str,
                        help='String Value - where the training log + the best method weights is are saved.')
    # model
    parser.add_argument('--model_name', default='yolo_v2',
                        help='String Value - yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')

    # dataset
    parser.add_argument("--dataset_csv", "-d",
                        type=str,
                        help='String Value - The path to the ISIC style dataset csv file.',
                        )

    return parser.parse_args()


def test(model, device, data_loader, log_dir):
    os.makedirs(log_dir, exist_ok=True)

    model = model.to(device)
    model.trainable = False
    model.eval()

    p_bar = tqdm(data_loader, total=len(data_loader), desc=f"Testing")
    for iter_i, (images, targets) in enumerate(p_bar):
        targets = [label[0].tolist() for label in targets]
        # TO DEVICE
        images = images.to(device)
        # FORWARD PASS
        bboxes, scores, cls_inds = model(images)
        best_bb_ind = np.argmax(scores)
        bbox = bboxes[best_bb_ind]
        output_filename = str(scores[best_bb_ind]).replace(".","_")+"_"+os.path.basename(data_loader.dataset.files[iter_i])
        output_filename = os.path.join(log_dir,output_filename)
        viz_annotation(images[0], bbox, output_filename, image_mean, image_std, target=targets[0])


def main(model_name, model_file, dataset_csv, log_dir, num_workers):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    yolo_net = models.model_dict[model_name]
    yolo_net_cfg = models.model_cfg_dict[model_name]
    input_size = yolo_net_cfg["size"]

    test_files_p, test_labels_p, _, _, _, _ = data.utils.get_isic(dataset_csv, 0, 0)

    # CREATE THE DATALOADERS
    dataset = FileDetection(files=test_files_p, labels=test_labels_p,
                            transform=BaseTransform(input_size, image_mean, image_std))
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              shuffle=False,
                                              batch_size=1,
                                              collate_fn=detection_collate,
                                              num_workers=num_workers, pin_memory=True)

    # BUILD THE MODEL
    model = yolo_net(device=device,
                     input_size=input_size,
                     num_classes=8,
                     trainable=False,
                     anchor_size=yolo_net_cfg['anchor_size'],
                     hr=False)
    #model.load_state_dict(torch.load(model_file))
    test(model, device, data_loader, log_dir)


if __name__ == '__main__':
    args = parseargs()
    main(**args.__dict__)
