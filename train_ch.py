from __future__ import division

import os
import argparse
import data.utils
from utils.chclassifier import CHClassifier
from PIL import Image
import numpy as np
import pickle
from tqdm import tqdm
import cv2

config_dictionary = {'remote_hostname': 'google.com', 'remote_port': 80}

# Step 2
with open('config.dictionary', 'wb') as config_dictionary_file:
    # Step 3
    pickle.dump(config_dictionary, config_dictionary_file)

def parseargs():
    parser = argparse.ArgumentParser(description='Color histogram based lesion detection.')
    # basic
    parser.add_argument('--log_dir', default='log/', type=str,
                        help='String Value - where the training log + the best method weights is are saved.')

    # dataset
    parser.add_argument("--isic_csv", "-p",
                        type=str,
                        help='String Value - The path to the ISIC dataset csv file.')


    return parser.parse_args()


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def test(log_dir,  isic_csv):

    dataset = "/media/stamas01/Data/datasets/NHS_cropped_yolo/data/"
    files, _, _ = data.utils.get_directory(dataset, 0, 0)

    mean_hist = np.load(os.path.join(log_dir, "mean_hist.npy"))

    distances = []
    for f in tqdm(files, desc="training"):
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        distances.append(cv2.compareHist(mean_hist, hist, cv2.HISTCMP_CORREL))

    indices = np.argsort(distances)[::-1]

    images = []
    for i in indices[:784]:
        f = files[i]
        img = Image.open(f).resize((128, 128))
        images.append(img)

    grid = image_grid(images, 28, 28)

    grid.save(os.path.join(log_dir, "nhs_likelihoods.jpg"))

def train(log_dir,  isic_csv):
    # CREATE THE LOG DIR IF DOESN'T EXIST
    os.makedirs(log_dir, exist_ok=True)

    # GET DATASETS
    train_files_p, train_labels_p, _, _, val_files_p, val_labels_p = data.utils.get_isic(isic_csv, 0, 0)
    mean_hist = None
    for f in tqdm(train_files_p[0:1000], desc="training"):
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])

        if mean_hist is None:
            mean_hist = hist
        else:
            mean_hist += hist

    mean_hist = cv2.normalize(mean_hist, mean_hist).flatten()

    likelihoods = []
    for f in tqdm(train_files_p[0:1000], desc="training"):
        img = cv2.imread(f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                            [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        likelihoods.append(cv2.compareHist(mean_hist, hist, cv2.HISTCMP_CORREL))


    indices = np.argsort(likelihoods)[::-1]
    images = []
    for i in indices[0:25]:
        f = train_files_p[i]
        img = Image.open(f).resize((128,128))
        images.append(img)

    grid = image_grid(images, 5,5)

    grid.save(os.path.join(log_dir, "lowest_likelihoods.jpg"))

    np.save(os.path.join(log_dir, "mean_hist.npy"), mean_hist)

    #with open(os.path.join(log_dir, "stat.txt"), 'w') as f:
    #    f.write(f'max:{max(likelihoods)}')
    #    f.write(f'min:{min(likelihoods)}')
    #    f.write(f'mean:{np.mean(likelihoods)}')

    #with open(os.path.join(log_dir, "ch_classifier.pkl"), 'wb') as file:
    #    pickle.dump(classifier, file)









if __name__ == '__main__':
    args = parseargs()
    train(**args.__dict__)
    test(**args.__dict__)
