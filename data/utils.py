import pandas as pd
import os
from glob import glob
import numpy as np


def get_directory(negative_dir, test_split, val_split):
    result = [y for x in os.walk(negative_dir) for y in glob(os.path.join(x[0], '*.*'))]
    df = pd.DataFrame({"images": result})
    train_df, test_df, val_df = slit_data(df, test_split, val_split)
    return train_df["images"].tolist(), None, test_df["images"].tolist(), None, val_df["images"].tolist(), None


def get_isic(isic_csv, test_split, val_split):
    df = pd.read_csv(os.path.join(isic_csv))
    train_df, test_df, val_df = slit_data(df, test_split, val_split)

    if os.path.basename(isic_csv) == "ISIC_2019_Training_GroundTruth.csv":
        image_dir = os.path.join(os.path.dirname(isic_csv), "ISIC_2019_Training_Input")
    else:
        image_dir = os.path.join(os.path.dirname(isic_csv),os.path.basename(isic_csv).split(".")[0])

    train_files = [os.path.join(image_dir, f + ".jpg") for f in train_df.image]
    train_labels = np.argmax(train_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    val_files = [os.path.join(image_dir, f + ".jpg") for f in val_df.image]
    val_labels = np.argmax(val_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    test_files = [os.path.join(image_dir, f + ".jpg") for f in test_df.image]
    test_labels = np.argmax(test_df.drop(["image", "UNK"], axis=1).to_numpy(), axis=1)
    return train_files, train_labels, test_files, test_labels, val_files, val_labels


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
