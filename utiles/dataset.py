from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy
from einops import rearrange
from imgaug import augmenters as iaa
import random


def square_picture(image, mask, image_size):
    """
    任意图片正方形中心化
    :param image: 图片
    :param image_size: 输出图片的尺寸
    :return: 输出图片
    """
    h1, w1, _ = image.shape
    max_len = max(h1, w1)
    fx = image_size / max_len
    fy = image_size / max_len
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    h2, w2, _ = image.shape
    background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
    background[:, :, :] = 127
    s_h = image_size // 2 - h2 // 2
    s_w = image_size // 2 - w2 // 2
    background[s_h:s_h + h2, s_w:s_w + w2] = image
    image = background.copy()
    background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
    background[s_h:s_h + h2, s_w:s_w + w2] = mask
    mask = background.copy()
    return image, mask


def randomly_adjust_brightness(image, lightness, saturation):
    # 颜色空间转换 BGR转为HLS
    image = image.astype(numpy.float32) / 255.0
    hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # 1.调整亮度（线性变换)
    hlsImg[:, :, 1] = (1.0 + lightness / float(100)) * hlsImg[:, :, 1]
    hlsImg[:, :, 1][hlsImg[:, :, 1] > 1] = 1
    # 饱和度
    hlsImg[:, :, 2] = (1.0 + saturation / float(100)) * hlsImg[:, :, 2]
    hlsImg[:, :, 2][hlsImg[:, :, 2] > 1] = 1
    # HLS2BGR
    lsImg = cv2.cvtColor(hlsImg, cv2.COLOR_HLS2BGR) * 255
    lsImg = lsImg.astype(numpy.uint8)
    return lsImg


def reset_image(image, mask, image_size, is_random_pation):
    h1, w1, _ = image.shape
    max_len = max(h1, w1)
    fx = image_size / max_len
    fy = image_size / max_len
    image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
    h2, w2, _ = image.shape
    background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
    if is_random_pation:
        background[:, :, :] = random.randint(0, 255)
        s_h = random.randint(0, image_size - h2)
        s_w = random.randint(0, image_size - w2)
    else:
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
    background[s_h:s_h + h2, s_w:s_w + w2] = image
    image = background.copy()
    background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
    background[s_h:s_h + h2, s_w:s_w + w2] = mask
    mask = background.copy()
    return image, mask


def sp_noise(image, prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = numpy.zeros(image.shape, numpy.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = random.randint(0, 255)
            elif rdn > thres:
                output[i][j] = random.randint(0, 255)
            else:
                output[i][j] = image[i][j]
    return output


def randon_crop(image, size=50):
    h, w, c = image.shape
    size = random.randint(2, size)
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    image[y:y + size, x:x + size] = 0
    return image


class MeterDataset(Dataset):

    def __init__(self, root='data', mode='train'):
        super(MeterDataset, self).__init__()
        self.dataset = []
        image_path = f'{root}/images/{mode}'
        mask_path = f'{root}/annotations/{mode}'
        for image_name in os.listdir(image_path):
            mask_name = image_name.split('.')[0] + '.png'
            self.dataset.append((f'{image_path}/{image_name}', f'{mask_path}/{mask_name}'))
        self.seq = iaa.SomeOf(
            n=(0, None),
            children=[
                # iaa.Affine(rotate=(-10, 10), scale=(0.9, 1.1)),
                # iaa.Grayscale(alpha=(0, 1), from_colorspace="BGR"),
                # iaa.ChannelShuffle(1),
                # iaa.Emboss(),
                iaa.GaussianBlur(sigma=(0, 3)),
                # iaa.Sharpen(alpha=(0, 0.2)),
                iaa.AdditiveGaussianNoise(loc=(-5, 5), scale=(0, 10))
            ],
            random_order=True,
        )
        self.fliplr = iaa.Fliplr()
        self.aff = iaa.Affine(scale=(0.95, 1.05), rotate=(-150, 150),order=0,random_state=False)
        self.is_train = True if mode == 'train' else False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image_path, mask_path = self.dataset[item]
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path)
        if self.is_train:
            image, mask = reset_image(image, mask, 416, True)
            if random.random() < 0.5:
                image = sp_noise(image, 0.005)
            if random.random() < 0.5:
                image = randon_crop(image)
            if random.random() < 0.5:
                image = randomly_adjust_brightness(image, random.randint(-20, 20), random.randint(-20, 20))
            image = self.seq.augment_images([image])[0]
            if random.random()<0.5:
                image = self.fliplr.augment_images([image])[0]
                mask = self.fliplr.augment_images([mask])[0]
            if random.random()<0.5:
                aff = self.aff.to_deterministic()
                image = aff.augment_images([image])[0]
                mask = aff.augment_images([mask])[0]
                # mask = self.aff.deterministic
        else:
            image, mask = square_picture(image, mask, 416)

        mask = mask[:,:,0]
        mask_t = numpy.zeros((2,416,416),dtype=numpy.uint8)
        condition = mask==1
        mask_t[0,condition]=1
        condition = mask == 2
        mask_t[1, condition] = 1
        # condition = mask >0
        # image[condition] = (0,0,255)
        # cv2.imshow('a',image)
        # cv2.waitKey(1)
        # plt.imshow(mask_t[0])
        # plt.show()
        # plt.imshow(mask_t[1])
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        # exit()
        image = torch.tensor(image).float()/255
        image = rearrange(image,' h w c -> c h w')
        mask = torch.tensor(mask_t).float()
        # mask = torch
        return image, mask



if __name__ == '__main__':
    d = MeterDataset(mode='train')
    for i in range(len(d)):
        d[2]
