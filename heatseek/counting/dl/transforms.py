from augmentations import compute_mean_std, MaskCompose, MaskRandomCrop, MaskToTensor, MaskNormalize, Mask2dMultiplyAndAddToBrightness, MaskContrast, MaskImgAug
import os

def get_transforms(clips_dir):
    """
    Build the three augmentation pipelines used in curriculum training.

    Computes per-dataset mean and std from the clips in `clips_dir` and
    constructs none, easy, and hard transform pipelines of increasing
    augmentation intensity. All three pipelines share the same crop size
    and normalization strategy: mean-centering by the dataset mean (scaled
    to [0,1]) with no std scaling (std=1.0), preserving the radiometric
    intensity range.

    Augmentation progression:
        none  — crop and normalize only; used for initial warmup training.
        easy  — adds mild brightness/contrast jitter and imgaug transforms;
                 brightness multiply in (0.85, 1.0), additive shift in (-15, 0).
        hard  — same structure as easy but with wider jitter ranges;
                 brightness multiply in (0.7, 1.0), additive shift in (-25, 0),
                 contrast factor in (0.6, 1.2).

    Parameters:
        clips_dir (str): Directory containing training clips, passed
            to compute_mean_std() to derive normalization statistics.

    Returns:
        none_data_transforms (MaskCompose): Minimal pipeline for warmup phase.
        easy_data_transforms (MaskCompose): Moderate augmentation pipeline.
        hard_data_transforms (MaskCompose): Heavy augmentation pipeline for
            final training phase.

    Note:
        std=1.0 means normalization only subtracts the mean without rescaling
        by standard deviation. This is intentional for radiometric data where
        the absolute intensity scale is meaningful and should be preserved
        relative to the mean.
    """
  
    mean, std = compute_mean_std(clips_dir)

    none_data_transforms = MaskCompose([
        MaskRandomCrop(224),
        MaskToTensor(),
        MaskNormalize(mean=mean/255, std=1.0)
    ])

    easy_data_transforms = MaskCompose([
        MaskRandomCrop(224),
        Mask2dMultiplyAndAddToBrightness(multiply=(0.85, 1.0), add=(-15, 0)),
        MaskContrast(contrast_factor=(0.8, 1.2)),
        MaskImgAug(),
        MaskToTensor(),
        MaskNormalize(mean=mean/255, std=1.0)
    ])

    hard_data_transforms = MaskCompose([
        MaskRandomCrop(224),
        Mask2dMultiplyAndAddToBrightness(multiply=(0.7, 1.0), add=(-25, 0)),
        MaskContrast(contrast_factor=(0.6, 1.2)),
        MaskImgAug(),
        MaskToTensor(),
        MaskNormalize(mean=mean/255, std=1.0)
    ])

    return none_data_transforms, easy_data_transforms, hard_data_transforms

def split_train_val(clips_dir, train_split=0.8, total_samples=5000):
    """
    Split frame/mask pairs into training and validation sets, sampled evenly
    across all clips.
    Parameters:
        clips_dir (str): Parent directory containing video subdirectories,
                         each containing clip folders with frames/ and masks/
        train_split (float): Fraction of samples for training. Default: 0.8
        total_samples (int): Total number of frame/mask pairs to use. Default: 5000
    Returns:
        train_pairs (list of tuples): (frame_path, mask_path) pairs for training
        val_pairs (list of tuples): (frame_path, mask_path) pairs for validation
    """
    # Collect all valid frame/mask pairs across the full directory structure
    all_pairs = []

    for video_dir in sorted(os.listdir(clips_dir)):
        video_path = os.path.join(clips_dir, video_dir)

        if not os.path.isdir(video_path):
            continue

        for clip_dir in sorted(os.listdir(video_path)):
            clip_path = os.path.join(video_path, clip_dir)

            if not os.path.isdir(clip_path):
                continue

            frames_path = os.path.join(clip_path, 'frames')
            masks_path = os.path.join(clip_path, 'masks')

            if not os.path.isdir(frames_path) or not os.path.isdir(masks_path):
                continue

            for filename in sorted(os.listdir(frames_path)):
                frame_path = os.path.join(frames_path, filename)
                mask_path = os.path.join(masks_path, filename.replace('.jpg', '.png'))

                if os.path.exists(mask_path):
                    all_pairs.append((frame_path, mask_path))

    if len(all_pairs) > total_samples:
        step = len(all_pairs) / total_samples
        all_pairs = [all_pairs[int(i * step)] for i in range(total_samples)]

    train_end = int(train_split * len(all_pairs))
    train_pairs = all_pairs[:train_end]
    val_pairs = all_pairs[train_end:]

    return train_pairs, val_pairs
