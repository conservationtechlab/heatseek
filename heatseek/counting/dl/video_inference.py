import os
import time
import cv2
import numpy as np
import torch
import torch.utils.data as data
from bat_seg_models import UNETTraditional
from augmentations import compute_mean_std, MaskCompose, MaskToTensor, MaskNormalize
import bat_functions
from BatIterableDataset import BatIterableDataset
import argparse

def denorm_image(im, mean):
    """
    Reverses mean normalization on a float image array and converts it to uint8.

    Adds back the per-channel mean (scaled to [0, 1]) to a normalized image,
    rescales pixel values to [0, 255], clips to valid range, and casts to uint8.

    Parameters:
        im (np.ndarray): Normalized image array of shape (H, W, C) with float
            values expected in roughly [-1, 1] or [0, 1] range.
        mean (np.ndarray or list): Per-channel mean values in [0, 255] range,
            shape (C,), used during the original normalization step.

    Returns:
        np.ndarray: Denormalized image as uint8 with pixel values in [0, 255],
            same spatial shape as input.
    """
    im += mean / 255
    im *= 255
    im = np.maximum(im, 0)
    im = np.minimum(im, 255)
    return im.astype(np.uint8)

def load_model(model_file):
    """
    Load a trained UNETTraditional model from a state dict file and prepare
    it for inference.

    Automatically selects CUDA if available, otherwise falls back to CPU.
    The model is set to evaluation mode (gradients and dropout disabled)
    before being returned.

    Parameters:
        model_file (str or path-like): Path to a saved state dict file
            (.pth or .pt) produced by torch.save().

    Returns:
        model (UNETTraditional): Trained model in eval mode, moved to the
            appropriate device (CUDA or CPU).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNETTraditional(1, 2, should_pad=False)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.train(False)
    return model

def get_augmentor(mean):
    """
    Build an image augmentation pipeline that converts an image/mask pair
    to tensors and applies mean normalization.

    Parameters:
        mean (np.ndarray or float): Per-channel mean values in [0, 255] range,
            divided by 255 internally to normalize to [0, 1] scale.

    Returns:
        MaskCompose: A composed transform that applies, in order:
            1. MaskToTensor  — converts image and mask to torch.Tensor.
            2. MaskNormalize — subtracts mean/255 from the image (std=1.0).
    """
    return MaskCompose([
        MaskToTensor(),
        MaskNormalize(mean=mean/255, std=1.0)
    ])

def run_inference(video_path, output_folder, model, mean,
                  bat_prob_thresh=0.6, batch_size=2,
                  save_every_n_frames=18000, early_stop=None):
    """Run model inference on a video file and save detections to disk."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, 'example-frames'), exist_ok=True)

    augmentor = get_augmentor(mean)
    bat_dataset = BatIterableDataset([video_path], augmentor=augmentor)
    dataloader = data.DataLoader(bat_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=0, pin_memory=True)

    centers_list = []
    contours_list = []
    sizes_list = []
    rects_list = []
    num_frames = 0
    t0 = None

    for batch_ind, batch in enumerate(dataloader):
        if batch_ind == 0:
            print('started...')
            t0 = time.time()
        if early_stop and batch_ind >= early_stop:
            break

        im_batch = batch['image'].to(device)

        with torch.no_grad():
            outputs = model(im_batch)
            masks = (outputs[:, 1].cpu().numpy() > np.log(bat_prob_thresh)).astype(np.uint8)

            for ind, mask in enumerate(masks):
                centers, areas, contours, _, _, rects = bat_functions.get_blob_info(mask)
                centers_list.append(centers)
                sizes_list.append(areas)
                contours_list.append(contours)
                rects_list.append(rects)
                if save_every_n_frames and num_frames % save_every_n_frames == 0:
                    im_name = f'frame_{num_frames:07d}.jpg'
                    im_file = os.path.join(output_folder, 'example-frames', im_name)
                    im = denorm_image(np.squeeze(batch['image'][ind].numpy()).copy(), mean)
                    cv2.imwrite(im_file, im)
                num_frames += 1

        if batch_ind % 1000 == 0 and t0 is not None:
            print(f'batch {batch_ind}, frame {num_frames}, time {time.time()-t0:.1f}s')

    total_time = time.time() - t0
    print(f'{total_time:.1f}s total, {total_time/max(batch_ind,1)/batch_size:.3f}s per frame')
    bat_dataset.get_read_frame_info()

    return centers_list, contours_list, sizes_list, rects_list

def save_detections(centers_list, contours_list, sizes_list, rects_list,
                    output_folder, num_contour_files=15):
    """Save per-frame detections to disk."""
    file_num = 0
    new_contours = []
    for frame_ind, cs in enumerate(contours_list):
        if frame_ind % int(len(contours_list) / num_contour_files) == 0:
            file_name = f'contours-compressed-{file_num:02d}.npy'
            np.save(os.path.join(output_folder, file_name),
                    np.array(new_contours, dtype=object))
            new_contours = []
            file_num += 1
        new_contours.append([])
        for c in cs:
            cc = np.squeeze(cv2.approxPolyDP(c, 0.1, closed=True))
            new_contours[-1].append(cc)
    file_name = f'contours-compressed-{file_num:02d}.npy'
    np.save(os.path.join(output_folder, file_name),
            np.array(new_contours, dtype=object))
    np.save(os.path.join(output_folder, 'size.npy'), np.array(sizes_list, dtype=object))
    np.save(os.path.join(output_folder, 'rects.npy'), np.array(rects_list, dtype=object))
    np.save(os.path.join(output_folder, 'centers.npy'), np.array(centers_list, dtype=object))
    print(f'Saved detections for {len(centers_list)} frames to {output_folder}')

def run_video_inference(video_path, output_folder, model_file, clips_dir,
                        bat_prob_thresh=0.6, batch_size=2,
                        save_every_n_frames=18000, early_stop=None):
    """Top level function — load model, run inference, save detections."""
    model = load_model(model_file)
    mean, _ = compute_mean_std(clips_dir)
    centers_list, contours_list, sizes_list, rects_list = run_inference(
        video_path, output_folder, model, mean,
        bat_prob_thresh, batch_size, save_every_n_frames, early_stop)
    save_detections(centers_list, contours_list, sizes_list, rects_list, output_folder)

def main():
    parser = argparse.ArgumentParser(description='Get centers and contours from detections from model inference')
    parser.add_argument('--vid_path', type=str, help='Path of video')
    parser.add_argument('--output_folder', type=str, help='Path to the folder where you want your outputs saved')
    parser.add_argument('--model_filepath', type=str, help='Filepath of trained model')
    parser.add_argument('--clips_dir', type=str, help='Directory containing the clips (faster way to calculate mean and std)')
    args = parser.parse_args()
    run_video_inference(video_path=args.vid_path, output_folder=args.output_folder, model_file=args.model_filepath, clips_dir=args.clips_dir)

if __name__ == '__main__':
    main()
    