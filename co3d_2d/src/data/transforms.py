import gin 
from typing import Tuple
import numpy as np
import torchvision.transforms as transforms
import torch
import functools

from PIL import Image
from co3d_2d.src.data.augmix import augment_and_mix

class Normalize(transforms.Normalize):
    
    mean = [123.68 / 255, 116.779 / 255, 103.939 / 255]
    std = [58.393 / 255, 57.12 / 255, 57.375 / 255]

    def __init__(self):
        super().__init__(self.mean, self.std)


@gin.configurable()
class ColorJitter(transforms.ColorJitter): 

    def __init__(
        self, 
        brightness: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.4, 
    ):
        super().__init__(
            brightness=brightness, 
            saturation=saturation, 
            hue=hue
        )


@gin.configurable()
class CenterCrop(object): 

    def __init__(
        self, 
        image_size: int = 224
    ):
        self.image_size = image_size
        self.transform = transforms.Resize(image_size)
        self.crop = transforms.CenterCrop(self.image_size)

    def __call__(self, x):
        return self.crop(self.transform(x))


@gin.configurable()
class RandomResizedCrop(transforms.RandomResizedCrop):

    def __init__(
        self, 
        image_size: Tuple[int, int] = (224, 224)
    ):
        super().__init__(image_size)


class ToTensor(transforms.ToTensor):
    pass


class Resize(transforms.Resize):

    def __init__(
        self, 
        image_size = 224,
    ):
        super().__init__(image_size)



class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super().__init__(p)



@gin.configurable()
class PCALoss(object):

    def __init__(self, alphastd=0.1):
        self.alphastd = alphastd
        self.eigval = torch.tensor([55.46, 4.794, 1.148]) / 255. 
        self.eigvec = torch.tensor([
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203]
        ])

    def __call__(self, x):
        dtype = x.dtype
        alpha = torch.normal(0, self.alphastd, size=(3,), dtype=dtype)
        rgb = (((self.eigvec * alpha) @ self.eigval[..., None]).T).squeeze(0)
        x = x + rgb[..., None, None]
        return x


@gin.configurable()
class AugMix(object): 

    def __init__(self, severity=3, width=3, depth=-1, alpha=1.):
        self.augmix = functools.partial(
            augment_and_mix, severity=severity, width=width, depth=depth, alpha=alpha
        )
        
    def __call__(self, x):
        return self.augmix(x)

@gin.configurable()
class BackgroundAug(object): 

    def __init__(self, rescale_range=[0.5, 1.5]):
        self.rescale_min = rescale_range[0]
        self.rescale_max = rescale_range[1]

    def __call__(self, fg, bg, mask):

        random_scale = np.random.random() * (self.rescale_max - self.rescale_min) + self.rescale_min
        random_size = (
            int(fg.size[0] * random_scale),
            int(fg.size[1] * random_scale)
        )
        
        fg = fg.resize(random_size)
        mask = mask.resize(random_size)

        fg_arr = np.asarray(fg)
        bg_arr = np.asarray(bg)
        mask_arr = (np.asarray(mask)[..., 0] / 255)

        bg_H, bg_W = bg_arr.shape[:2]
        fg_H, fg_W = fg_arr.shape[:2]

        H_start = max(0, (bg_H - fg_H) // 2)
        W_start = max(0, (bg_W - fg_W) // 2)
        
        H_end = min(bg_H, (bg_H + fg_H) // 2)
        W_end = min(bg_W, (bg_W + fg_W) // 2)

        mask_cropped = mask_arr[
            fg_H // 2 - (H_end - H_start) // 2:
            fg_H // 2 - (H_end - H_start) // 2 + (H_end - H_start),
            fg_W // 2 - (W_end - W_start) // 2:
            fg_W // 2 - (W_end - W_start) // 2 + (W_end - W_start),
            None
        ]

        bg_arr[H_start:H_end, W_start:W_end] = fg_arr[
            fg_H // 2 - (H_end - H_start) // 2:
            fg_H // 2 - (H_end - H_start) // 2 + (H_end - H_start),
            fg_W // 2 - (W_end - W_start) // 2:
            fg_W // 2 - (W_end - W_start) // 2 + (W_end - W_start),
        ] * mask_cropped + (1 - mask_cropped) * bg_arr[H_start:H_end, W_start:W_end]

        return Image.fromarray(bg_arr)