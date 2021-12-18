import torch.utils.data as data
from PIL import Image
import numpy as np


class FileDetection(data.Dataset):
    """ISIC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self,
                 name = "Default",
                 files = None,
                 labels = None,
                 transform=None):
        self.name = name
        self.files = files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        # load an image
        img = Image.open(self.files[index]).convert("RGB")
        # create a bounding box for a positive image (that has a label)
        if self.labels is None:
            target = [[0.0, 0.0, 0.0, 0.0, 0]]
        else:
            target = [[0.0, 0.0, 1.0, 1.0, self.labels[index]]]
        target = np.array(target)

        if self.transform is not None:
            img, boxes = self.transform(img, target[:, :4])
            target[:, :4] = boxes

        return img, target

    def __len__(self):
        return len(self.files)


