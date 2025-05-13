import cv2
import os
import json
import argparse

annotations = {}
current_points = []
#TODO add save folder path
def click_event(event, x, y, flags, param):
    global current_points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(param, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow("Annotator", param)

def annotate_image(image_path):
    global current_points
    current_points = []
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    display_img = img.copy()
    cv2.imshow("Annotator", display_img)
    cv2.setMouseCallback("Annotator", click_event, display_img)

    print(f"Annotating {image_path} — Press 's' to save, 'n' to skip.")
    while True:
        key = cv2.waitKey(0)
        if key == ord('s'):
            return current_points
        elif key == ord('n'):
            return None
        elif key == 27:  # ESC key
            print("Exiting.")
            exit()

def annotate_folder(image_dir, output_json):
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png"))])

    for fname in image_files:
        img_path = os.path.join(image_dir, fname)
        pts = annotate_image(img_path)
        if pts is not None:
            annotations[fname] = pts

    cv2.destroyAllWindows()
    with open(output_json, "w") as f:
        json.dump(annotations, f, indent=2)
    print(f"Annotations saved to {output_json}")
