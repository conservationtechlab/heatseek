import cv2
import subprocess
import os
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_vids(input_vid, output_dir, pixel_threshold, count_threshold, padding, min_gap):
    """Scans the input video for motion and crops out clips of motion events.
    Parameters:
    - input_vid: path to the input video file
    - output_dir: directory where the output clips will be saved
    - threshold: sensitivity for motion detection (lower = more sensitive)
    - padding: seconds of padding to add before and after each detected motion event
    - min_gap: minimum gap in seconds to merge nearby motion events into a single clip
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(input_vid)
    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_frame = None
    motion_times = []
    logging.info("Scanning for motion")

    while True:             #while there are frames to read
        ret, frame = cap.read()    #ret returns true if frame is read correctly, frame is the image array         

        if not ret:         #if there are no more frames left to read
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert the frame to grayscale for easier processing

        if prev_frame is not None:   #if there is a previous frame to compare to (i.e. not the first frame)
            diff = cv2.absdiff(prev_frame, gray)   #calculate the absolute difference between the current frame and the previous frame

            # if diff.mean() > threshold:
            if (diff > pixel_threshold).sum() > count_threshold:   #if the number of pixels that changed significantly is greater than the count threshold, we consider it motion
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000   #get the current time in milliseconds
                motion_times.append(t)    #add the time to the list of motion times

        prev_frame = gray    #update the previous frame to the current frame for the next iteration
        
    cap.release()    #release the video capture object

    if not motion_times:    #if no motion was detected, print a message and exit
        logging.info("No motion detected. Try lowering threshold")
        return

    logging.info(f"Motion found at {len(motion_times)} frames, merging into clips")
    clips = []
    start = motion_times[0]
    end = motion_times[0]

    for t in motion_times[1:]:   #iterate through the motion times starting from the second one
        if t - end < min_gap:    #if the time between the current motion time and the end of the last clip is less than the minimum gap, we consider it part of the same clip
            end = t    #update the end time of the current clip to the current motion time
        else:
            clips.append((start, end))   #if it's not part of the same clip, add the current clip to the list of clips
            start = t  #start a new clip with the current motion time as the start
            end = t    #and also set the end to the current motion time for now

    clips.append((start, end))   #add the last clip to the list of clips
    logging.info(f"Found {len(clips)} clip(s):")

    for i, (s, e) in enumerate(clips):   #iterate through the clips and print their start and end times
        s_pad = max(0, s - padding)   #add padding to the start time, ensuring it doesn't go below 0
        e_pad = e + padding   #add padding to the end time
        logging.info(f"Clip {i+1}: {s_pad:.1f}s → {e_pad:.1f}s")    #print the clip number and its start and end times

        out_path = os.path.join(output_dir, f"clip_{i+1:03d}.mp4")   #create the output path for the clip
        subprocess.run([    #use ffmpeg to extract the clip from the original video
            "ffmpeg", "-y",    #-y to overwrite existing files without asking
            "-ss", str(s_pad),    #start time for the clip
            "-to", str(e_pad),    #end time for the clip
            "-i", input_vid,  #input video file
            "-c", "copy",     #copy the video and audio streams without re-encoding
            out_path          #output path for the clip
        ])

    logging.info(f"Done! Clips saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Crop motion events from a video into separate clips")
    parser.add_argument("--input", required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output clips")
    parser.add_argument("--pixel_threshold", type=float, default=15, help="Sensitivity for motion detection (lower = more sensitive)")
    parser.add_argument("--count_threshold", type=int, default=20, help="Minimum number of pixels that must change to consider it motion")
    parser.add_argument("--padding", type=float, default=2.0, help="Seconds of padding to add before and after each detected motion event")
    parser.add_argument("--min_gap", type=float, default=2.0, help="Minimum gap in seconds to merge nearby motion events into a single clip")
    args = parser.parse_args()
    crop_vids(args.input, args.output_dir, args.pixel_threshold, args.count_threshold, args.padding, args.min_gap)

if __name__ == "__main__":
    main()
