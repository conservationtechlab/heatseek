import glob
import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
import random

def compute_mean_std(clips_dir, sample_size=500):
    """
    Compute the mean and standard deviation of pixels in frames across the video clips
    Parameters:
        clips_dir (str) : path to the directory containing the clips
        sample_size (int): number of frames to sample for computation
    Returns:
        mean (np.float64): the mean of the pixels
        std (np.float64) : the standard deviation of values across pixels
    """
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    image_paths = glob.glob(os.path.join(clips_dir, '*', '*', 'frames', '*.jpg'))

    if len(image_paths) > sample_size:
        image_paths = random.sample(image_paths, sample_size)

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float64)
        pixel_sum += img.sum()
        pixel_sq_sum += (img ** 2).sum()
        pixel_count += img.size

    mean = pixel_sum / pixel_count
    std = np.sqrt(pixel_sq_sum / pixel_count - mean ** 2)
    return mean, std

class MaskToTensor:
    def __call__(self, sample):
        im = sample['image']
        im_tensor = torch.from_numpy(im).float() / 255
        im_tensor = torch.unsqueeze(im_tensor, 0)
        sample['image'] = im_tensor
        
        if 'mask' in sample.keys():
            mask = sample['mask']
            mask_tensor = torch.from_numpy(np.asarray(mask, np.int64))
            mask_tensor = mask_tensor // 255
            sample['mask'] = mask_tensor
        
        return sample
    
class MaskNormalize:
    def __init__(self, mean, std):
        """Normalize image, leave mask unchanged"""
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        im = sample['image']
        im_norm = TF.normalize(im, self.mean, self.std)
        
        sample['image'] = im_norm
        return sample

class MaskCompose:
    def __init__(self, transform_list):
        """Chain together transforms in transform.
         Expect transform on both image and mask in dict
         
         Args:
             transform_list (list): list of custom augmentations
         """
        self.transform_list = transform_list
    
    def __call__(self, sample):
        for transform in self.transform_list:
            sample  = transform(sample)
        return sample
    
class MaskImgAug:
    def __call__(self, sample):
        im = sample['image'].astype(np.uint8)
        
        # gaussian blur with 40% probability
        if np.random.random() < 0.4:
            sigma = np.random.uniform(0.0, 6.0)
            ksize = int(6 * sigma + 1)
            if ksize % 2 == 0:
                ksize += 1
            im = cv2.GaussianBlur(im, (ksize, ksize), sigma)
        
        # poisson noise with 40% probability
        if np.random.random() < 0.4:
            lam = np.random.uniform(0, 30)
            noise = np.random.poisson(lam, im.shape).astype(np.float64)
            im = np.clip(im.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        
        sample['image'] = im.astype(float)
        return sample
    
class MaskRandomCrop:
    """Rotate by one of the given angles."""

    def __init__(self, crop_size):
        # size of square crop
        self.crop_size = crop_size

    def __call__(self, sample):
        im = sample['image']
        mask = sample['mask']

        im_height = im.shape[0]
        im_width = im.shape[1]
        top = np.random.randint(im_height - self.crop_size)
        left = np.random.randint(im_width - self.crop_size)

        im_crop = im[top:top+self.crop_size, left:left+self.crop_size]
        mask_crop = mask[top:top+self.crop_size, left:left+self.crop_size]
        
        sample['image'] = im_crop
        sample['mask'] = mask_crop

        return sample

class Mask2dMultiplyAndAddToBrightness():
    def __init__(self, add, multiply):
        """Add and multiply image values by given amount randomly within range.
        
        Expects image to be between 0 and 255.
        
        Args:
            add: either number of tuple, if tuple randomly choose from range
            multiply: either number ot tuple, if tuple randomly choose from range
        """
        self.add = add
        self.multiply = multiply
        self.rng = np.random.default_rng()
        
    def __call__(self, sample):
        im = sample['image'].astype(np.float64)
        if isinstance(self.add, tuple):
            add_val = self.rng.uniform(self.add[0], self.add[1])
        else:
            add_val = self.add
        if isinstance(self.multiply, tuple):
            multiply_val = self.rng.uniform(self.multiply[0], self.multiply[1])
        else:
            multiply_val = self.multiply
        im += add_val
        im *= multiply_val
        im = np.maximum(im, 0)
        im = np.minimum(im, 255)
        
        sample['image'] = im
        
        return sample

class MaskContrast:
    def __init__(self, contrast_factor):
        """Change image contrast, leave mask unchanged
        
        Args:
            contrast_factor: single value or tuple"""
        self.contrast_factor = contrast_factor
        self.rng = np.random.default_rng()

    def __call__(self, sample):
        im = sample['image']
        if isinstance(self.contrast_factor, tuple):
            contrast_factor = self.rng.uniform(self.contrast_factor[0], self.contrast_factor[1])
        else:
            contrast_factor = self.contrast_factor
        im_height, im_width = im.shape
        aprox_mean = np.mean([im[0,0], im[-1, 0], im[0, -1], im[-1, -1], 
                              im[im_height//2, im_width//2], im[-im_height//2, -im_width//2]])
        im_contrast = aprox_mean + contrast_factor * (im - aprox_mean)
        im_contrast = np.maximum(im_contrast, 0)
        im_contrast = np.minimum(im_contrast, 255)
        sample['image'] = im_contrast
        return sample
