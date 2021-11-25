import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
import torch

transform = transforms.ToPILImage(mode='RGB')

def denormalize(image, mean, std):
    image *= std
    image += mean
    image *= 255
    return image


def viz_annotation(image, box, save_to, image_mean=[0, 0, 0], image_std=[1, 1, 1], target = None):
    image = image.detach().cpu()
    image = image * torch.tensor(image_std).view(3, 1, 1)
    image = image + torch.tensor(image_mean).view(3, 1, 1)
    image = transform(image)

    #image = image.detach().cpu().numpy()
    #image = np.swapaxes(image, 0, 2)
    #image = denormalize(image, image_mean, image_std)

    # Denormalize box coordinates.

    # Convert image from pytorch to CV: CxHxW -> HxWxC



    #image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = map(lambda x: image.size[0] * x, box[0:4])
    draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)
    if target is not None:
        x_min_t, y_min_t, x_max_t, y_max_t = map(lambda x: image.size[0] * x, target[0:4])
        draw.rectangle((x_min_t, y_min_t, x_max_t, y_max_t), outline="green", width=5)
    image.save(save_to)
