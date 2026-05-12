import cv2
import numpy as np
import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_background_temp(image_path, min_val, max_val):
    """Get background temperature of an image taken by a radiometric camera.
    Parameters:
    - image_path: path to the image file (e.g. frame.png)
    - min_val: minimum temperature value corresponding to pixel value 0
    - max_val: maximum temperature value corresponding to pixel value 255
    """
    frame = cv2.imread(image_path)          #read the frame

    if frame is None:
        raise ValueError("Could not read image")
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert to grayscale dimensions
    
    k100 = (gray.astype(np.float32) / 255.0) * (max_val - min_val) + min_val    #formula to convert pixel values to temperature in Kelvin (FLIR Boson)
    temp_celsius = (k100 / 100.0) - 273.15
    background_temp = np.median(temp_celsius)
    return background_temp          #return the estimated background temperature in Celsius based on the median pixel value of the thermal map

def get_mask(image_path, min_val, max_val, threshold):
    """Get a segmentation mask of objects hotter than the background by a threshold.
    Parameters:
    - image_path: path to the image file
    - min_val: minimum temperature value corresponding to pixel value 0
    - max_val: maximum temperature value corresponding to pixel value 255
    - background_temp: estimated background temperature in °C
    - threshold: how many °C above background a pixel must be to be masked
    """
    frame = cv2.imread(image_path)     

    if frame is None:
        raise ValueError("Could not read image")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k100 = (gray.astype(np.float32) / 255.0) * (max_val - min_val) + min_val
    temp_celsius = (k100 / 100.0) - 273.15
    background_temp = get_background_temp(image_path, min_val, max_val)
    mask = (temp_celsius > (background_temp + threshold)).astype(np.uint8) * 255        #create a binary mask where pixels hotter than background + threshold are set to 255 (white) and others are 0 (black)
    return mask



def main():
    parser = argparse.ArgumentParser(description='Generate thermal segmentation mask for a single image')
    parser.add_argument('--clips_dir', type=str, help='Path to the clips folder')
    parser.add_argument('--min_val', type=float, default=28000)
    parser.add_argument('--max_val', type=float, default=32000)
    parser.add_argument('--threshold', type=float, default=5.0)
    args = parser.parse_args()

    for session_folder in os.listdir(args.clips_dir):
        session_path = os.path.join(args.clips_dir, session_folder)

        if not os.path.isdir(session_path):
            continue

        for clip_folder in os.listdir(session_path):
            clip_path = os.path.join(session_path, clip_folder)

            if not os.path.isdir(clip_path):
                continue

            frames_path = os.path.join(clip_path, 'frames')
            masks_path = os.path.join(clip_path, 'masks')

            if not os.path.isdir(frames_path):
                continue

            os.makedirs(masks_path, exist_ok=True)

            for filename in os.listdir(frames_path):
                image_path = os.path.join(frames_path, filename)
                mask = get_mask(image_path, args.min_val, args.max_val, args.threshold)
                output_file = os.path.join(masks_path, filename.replace('.jpg', '.png'))
                cv2.imwrite(output_file, mask)

            logging.info(f"Finished processing {clip_folder}")

if __name__ == "__main__":
    main()
