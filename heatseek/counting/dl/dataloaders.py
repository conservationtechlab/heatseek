import numpy as np
import datetime
from SegmentationDataset import SegmentationDataset
from transforms import split_train_val
import torch.utils.data as data

def worker_init_fn(worker_id):
    np.random.seed(datetime.datetime.now().microsecond + worker_id * 1000000)

def load_dataloader(clips_dir, transform, split='train', batch_size=4, num_workers=7, pin_memory=True):
    train_pairs, val_pairs = split_train_val(clips_dir)
    pairs = train_pairs if split == 'train' else val_pairs
    dataset = SegmentationDataset(pairs, transform)
    return data.DataLoader(dataset, batch_size=batch_size,
                          shuffle=(split == 'train'), num_workers=num_workers,
                          pin_memory=pin_memory, worker_init_fn=worker_init_fn)
