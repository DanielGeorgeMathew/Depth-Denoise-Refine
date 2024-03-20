from sklearn.utils import resample
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import collections
import kornia

try:
    import accimage
except ImportError:
    accimage = None
import random
import scipy.ndimage as ndimage

import pdb
import cv2


def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        applied_angle = random.uniform(-self.angle, self.angle)
        angle1 = applied_angle
        angle1_rad = angle1 * np.pi / 180

        image = ndimage.interpolation.rotate(
            image, angle1, reshape=self.reshape, order=self.order)
        depth = ndimage.interpolation.rotate(
            depth, angle1, reshape=self.reshape, order=self.order)

        image = Image.fromarray(image)
        depth = Image.fromarray(depth)

        return {'image': image, 'depth': depth}


class RandomHorizontalFlip(object):

    def __call__(self, sample):
        rgb, depth, gt_depth, albedo = sample['rgb'], sample['depth'], sample['gt_depth'], sample['albedo']

        if not _is_pil_image(rgb):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(image)))
        if not _is_pil_image(depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(mask)))
        if not _is_pil_image(gt_depth):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(mask)))

        if random.random() < 0.5:
            rgb = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            gt_depth = gt_depth.transpose(Image.FLIP_LEFT_RIGHT)
            albedo = albedo.transpose(Image.FLIP_LEFT_RIGHT)

        return {'rgb': rgb, 'depth': depth, 'gt_depth': gt_depth, 'albedo': albedo}


import matplotlib.pyplot as plt


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        rgb, depth, gt_depth, albedo = sample['rgb'], sample['depth'], sample['gt_depth'], sample['albedo']
        # print(image.size) 1242,375
        # print(depth.size)
        # exit()
        rgb = self.changeScale(rgb, self.size)
        depth = self.changeScale(depth, self.size)
        gt_depth = self.changeScale(gt_depth, self.size)
        albedo = self.changeScale(albedo, self.size)

        return {'rgb': rgb, 'depth': depth, 'gt_depth': gt_depth, 'albedo': albedo}

    def changeScale(self, img, size, interpolation=Image.BILINEAR):

        if not _is_pil_image(img):
            raise TypeError(
                'img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(size))

        if isinstance(size, int):
            w, h = img.size
            # h, w = img.shape
            if (w <= h and w == size) or (h <= w and h == size):
                return img
            if w < h:
                ow = size
                oh = int(size * h / w)
                return img.resize((ow, oh), interpolation)
            else:
                oh = size
                ow = int(size * w / h)
                return img.resize((ow, oh), interpolation)
        else:
            return img.resize(size[::-1], interpolation)


import PIL


class CenterCrop(object):
    def __init__(self, size_image):
        self.size_image = size_image

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        ### crop image and depth to (304, 228)
        # print(np.unique(depth))
        image = self.centerCrop(image, self.size_image)
        depth = self.centerCrop(depth, self.size_image)
        ### resize depth to (152, 114) downsample 2

        return {'image': image, 'mask': mask}

    def centerCrop(self, image, size):
        w1, h1 = image.size

        tw, th = size

        if w1 == tw and h1 == th:
            return image
        ## (320-304) / 2. = 8
        ## (240-228) / 2. = 8
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        image = image.crop((x1, y1, tw + x1, th + y1))

        return image


class RandomCrop(object):
    def __init__(self, size_image):
        self.size_image = size_image

    def __call__(self, sample):
        image, depth, gt_depth, albedo = sample['rgb'], sample['depth'], sample['gt_depth'], sample['albedo']
        w, h = image.size
        rand_x = random.randint(0, w - self.size_image[0])  # 480-30
        rand_y = random.randint(0, h - self.size_image[1])  # 640-30
        cropped_image = self.randomCrop(image, self.size_image, rand_x, rand_y)
        cropped_depth = self.randomCrop(depth, self.size_image, rand_x, rand_y)
        cropped_gt_depth = self.randomCrop(gt_depth, self.size_image, rand_x, rand_y)
        cropped_albedo = self.randomCrop(albedo, self.size_image, rand_x, rand_y)
        # while not (np.all(np.asarray(cropped_depth)) and np.all(np.asarray(cropped_gt_depth))):
        #     rand_x = random.randint(0, 450)  # 480-30
        #     rand_y = random.randint(0, 610)  # 640-30
        #     cropped_image = self.randomCrop(image, self.size_image, rand_x, rand_y)
        #     cropped_depth = self.randomCrop(depth, self.size_image, rand_x, rand_y)
        #     cropped_gt_depth = self.randomCrop(gt_depth, self.size_image, rand_x, rand_y)
        # tw, th = self.size_image
        # fig, ax = plt.subplots(2, 3)
        # ax[0][0].imshow(cv2.rectangle(np.asarray(image), (rand_x, rand_y), (tw + rand_x, th + rand_y), (0, 255, 0), 2))
        # ax[0][1].imshow(cv2.rectangle(np.asarray(depth), (rand_x, rand_y), (tw + rand_x, th + rand_y), (0, 255, 0), 2))
        # ax[0][2].imshow(cv2.rectangle(np.asarray(gt_depth), (rand_x, rand_y), (tw + rand_x, th + rand_y), (0, 255, 0), 2))
        # ax[1][0].imshow(np.asarray(cropped_image))
        # ax[1][1].imshow(np.asarray(cropped_depth))
        # ax[1][2].imshow(np.asarray(cropped_gt_depth))
        # plt.show()
        # exit()
        return {'rgb': cropped_image, 'depth': cropped_depth, 'gt_depth': cropped_gt_depth, 'albedo': cropped_albedo}

    def randomCrop(self, image, size, rand_x, rand_y):
        tw, th = size
        image = image.crop((rand_x, rand_y, tw + rand_x, th + rand_y))
        return image


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        rgb, depth, gt_depth, albedo = sample['rgb'], sample['depth'], sample['gt_depth'], sample['albedo']

        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        # ground truth depth of training samples is stored in 8-bit while test samples are saved in 16 bit
        rgb = self.rgb_to_tensor(rgb)
        depth = self.depth_to_tensor(depth)
        gt_depth = self.depth_to_tensor(gt_depth)
        albedo = self.albedo_to_tensor(albedo)
        return {'rgb': rgb, 'depth': depth, 'gt_depth': gt_depth, 'albedo': albedo}

    def albedo_to_tensor(self, pic):
        pic_array = (np.asarray(pic) / 255.0).astype(np.float32)
        # pic_tensor = torch.permute(torch.from_numpy(pic_array), [2, 0, 1])
        pic_tensor = torch.from_numpy(pic_array).unsqueeze(0)
        return pic_tensor

    def depth_to_tensor(self, pic):
        high_thresh = 65535
        low_thresh = 0
        thresh_range = (high_thresh - low_thresh) / 2.0
        pic_array = ((np.asarray(pic) - low_thresh) / thresh_range).astype(np.float32) - 1
        pic_tensor = torch.from_numpy(pic_array).unsqueeze(0)
        return pic_tensor

    def rgb_to_tensor(self, pic):
        pic_array = (np.asarray(pic) / 127.0).astype(np.float32) - 1.0
        pic_tensor = torch.permute(torch.from_numpy(pic_array), [2, 0, 1])
        return pic_tensor


class Lighting(object):

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']
        if self.alphastd == 0:
            return image

        alpha = image.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(image).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth}


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.var, self.var)

        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.var, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        image, depth = sample['image'], sample['depth']

        if self.transforms is None:
            return {'image': image, 'depth': depth}
        order = torch.randperm(len(self.transforms))
        for i in order:
            image = self.transforms[i](image)
        return {'image': image, 'depth': depth}


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        rgb, depth, gt_depth = sample['rgb'], sample['depth'], sample['gt_depth']
        rgb = self.normalize(rgb, self.mean, self.std)

        return {'rgb': rgb, 'depth': depth, 'gt_depth': gt_depth}

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.
        See ``Normalize`` for more details.
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        Returns:
            Tensor: Normalized image.
        """

        # TODO: make efficient
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        return tensor


from PIL import ImageEnhance


class RandomBrightness(object):
    def __init__(self, brightness=0.2):
        self.brightness = brightness

    def __call__(self, sample):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        rgb, depth, gt_depth, albedo = sample['rgb'], sample['depth'], sample['gt_depth'], sample['albedo']
        enhancer = ImageEnhance.Brightness(rgb)
        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        rgb = enhancer.enhance(brightness_factor)
        return {'rgb': rgb, 'depth': depth, 'gt_depth': gt_depth, 'albedo': albedo}

