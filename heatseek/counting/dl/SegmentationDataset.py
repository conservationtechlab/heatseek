import torch.utils.data as data
import cv2    

class SegmentationDataset(data.Dataset):
    def __init__(self, pairs, transform=None, keep_orig=False):
        """
        Args:
            pairs (list of tuples): list of (frame_path, mask_path) pairs
            transform (callable, optional): Optional transforms to be applied on a sample
            keep_orig (bool): keep un-transformed image
        """
        self.img_files = [p[0] for p in pairs]
        self.mask_files = [p[1] for p in pairs]
        self.keep_orig = keep_orig
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        sample = {'image': image[2:-2, 2:-2], 'mask': mask[2:-2, 2:-2]}
        if self.keep_orig:
            sample['orig'] = image
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.img_files)
