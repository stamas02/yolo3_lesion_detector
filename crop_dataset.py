import os
import argparse
import torch
from data.custom import FileDetection
from data import BaseTransform, detection_collate
from utils.augmentations import TransformTest
import models
import data.utils
from tqdm import tqdm
from utils.visualization import torch_to_pil
import numpy as np



def parseargs():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic

    parser.add_argument('--model_file', default=1, type=str,
                        help='String Value - Path to the saved model file')
    parser.add_argument('--num_workers', default=1, type=int,
                        help='String Value - Number of workers used in dataloading')
    parser.add_argument('--dst_dir', default='log/', type=str,
                        help='String Value - where the cropped version of the dataset is saved.')
    # model
    parser.add_argument('--model_name', default='yolo_v2',
                        help='String Value - yolov2_d19, yolov2_r50, yolov2_slim, yolov3, yolov3_spp, yolov3_tiny')

    # dataset
    parser.add_argument("--dataset_csv", "-d",
                        type=str,
                        help='String Value - The path to the ISIC style dataset csv file.',
                        )

    return parser.parse_args()



def crop_dataset(model, device, data_loader, src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    model = model.to(device)
    model.trainable = False
    model.eval()

    p_bar = tqdm(data_loader, total=len(data_loader), desc=f"Testing")
    for iter_i, (image, _) in enumerate(p_bar):

        # TO DEVICE
        image = image.to(device)
        # FORWARD PASS
        bboxes, scores, cls_inds = model(image)
        best_bb_ind = np.argmax(scores)
        bbox = bboxes[best_bb_ind]
        output_filename = dst_dir+data_loader.dataset.files[iter_i][len(src_dir)::]

        image = torch_to_pil(image.detach().cpu()[0])
        x_min, y_min, x_max, y_max = map(lambda x: image.size[0] * x, bbox[0:4])
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        cropped_image.save(output_filename)


def main(model_name, model_file, dataset_csv, dst_dir, num_workers):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    yolo_net = models.model_dict[model_name]
    yolo_net_cfg = models.model_cfg_dict[model_name]
    input_size = yolo_net_cfg["size"]

    test_files_p, test_labels_p, _, _, _, _ = data.utils.get_isic(dataset_csv, 0, 0)

    # CREATE THE DATALOADERS
    dataset = FileDetection(files=test_files_p, labels=test_labels_p,
                            transform=TransformTest(input_size))
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
    src_dir = dataset_csv.split(".")[0]+"/"
    dst_dir = os.path.join(dst_dir,os.path.basename(os.path.dirname(src_dir))+"/")
    crop_dataset(model, device, data_loader, src_dir, dst_dir)


if __name__ == '__main__':
    args = parseargs()
    main(**args.__dict__)
