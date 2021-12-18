import cv2
import numpy as np
from numpy import random
from torchvision.transforms import functional as F
from torchvision import transforms
import torch
from PIL import Image


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class CenterFullCrop(torch.nn.Module):
    """Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
    """

    def __init__(self):
        super().__init__()

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.

        Returns:
            PIL Image or Tensor: Cropped image.
        """
        new_size = min(img.size[0], img.size[1])
        return F.center_crop(img, (new_size, new_size))

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class RandomShrinkWithBB(torch.nn.Module):
    """ Shrinks the image randomly so that it keeps its original dimensions by filling the rest  with its mean color"""

    def __init__(self, ratio):
        """

        Args:
            ratio: Ratio by with the image is shrink compared to its original size. E.g 0.5 shrinks the image half size.
        """
        self.ratio = ratio

    def __call__(self, img, bboxes = None):
        """

        Args:
            img: The image to be shrank. PIL Image
            bboxes: [[xmin, ymin, xmax, ymax],..,[]], shape [num of bounding boxes, 4].

        Returns: shrank image and the accordingly resized bounding boxes.

        """
        width, height = img.size
        fill_color = np.mean(img, axis=(0, 1), dtype=int)
        new_img = Image.new(img.mode, (width, height), tuple(fill_color))
        shrink_ratio = random.uniform(self.ratio, 1)
        shrinked_width, shrinked_height = (int(shrink_ratio * width), int(shrink_ratio * height))
        shrinked_img = img.resize((shrinked_width, shrinked_height))
        left_pos = random.randint(0, width - shrinked_width) if width - shrinked_width > 0 else 0
        top_pos = random.randint(0, height - shrinked_height) if height - shrinked_height > 0 else 0
        new_img.paste(shrinked_img, (left_pos, top_pos))
        if bboxes is not None:
            for i, bbox in enumerate(bboxes):
                bboxes[i][0] = ((bboxes[i][0] * shrinked_width) + left_pos) / width
                bboxes[i][1] = ((bboxes[i][1] * shrinked_height) + top_pos) / height
                bboxes[i][2] = ((bboxes[i][2] * shrinked_width) + left_pos) / width
                bboxes[i][3] = ((bboxes[i][3] * shrinked_height) + top_pos) / height

        return new_img, bboxes


class TransformTrain(object):
    def __init__(self, size=416,
                 crop_scale=1,
                 random_shrink_ratio=1,
                 random_brightness = 0,
                 random_contrast = 0,
                 random_saturation = 0,
                 random_hue = 0):
        self.size = size
        self.augment_1 = transforms.Compose([
            CenterFullCrop(),
            transforms.RandomResizedCrop(size, (crop_scale, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ])
        self.augment_2 = RandomShrinkWithBB(random_shrink_ratio)
        self.augment_3 = transforms.Compose([
            transforms.ColorJitter(brightness=random_brightness,
                                   contrast = random_contrast,
                                   saturation=random_saturation,
                                   hue=random_hue),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    def __call__(self, img, bbox = None):
        img = self.augment_1(img)
        img, bbox = self.augment_2(img, bbox)
        img = self.augment_3(img)
        return (img, bbox) if bbox is not None else img


class TransformTest(object):
    def __init__(self, size=416):
        self.size = size
        self.augment = transforms.Compose([
            CenterFullCrop(),
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, img, bbox=None):
        return (self.augment(img), bbox) if bbox is not None else self.augment(img)
