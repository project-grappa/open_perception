import os
import cv2
import numpy as np
import argparse


titles = ["The can on the Right",
          "The toothpaste on the rear left of the keyboard",
          "The yellow box on the right of the plastic cube",
          "The marron can in front of the cylinder orange",
          "The red object"]

def find_main_rectangle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    return x+10, y, w-10, h-10

def crop_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            img_path = os.path.join(input_folder, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue
            rect = find_main_rectangle(image)
            if rect:
                x, y, w, h = rect
                cropped = image[y:y+h, x:x+w]
                out_path = os.path.join(output_folder, fname)
                cv2.imwrite(out_path, cropped)
            else:
                print(f"Could not find main rectangle in {fname}")

if __name__ == "__main__":
    input_folder = "/home/arthur/Desktop/CMU/research/motorcortex/motor_cortex/motor_cortex/perception/open_vocab_perception_pipeline/docs/assets/detections/input"
    output_folder = "/home/arthur/Desktop/CMU/research/motorcortex/motor_cortex/motor_cortex/perception/open_vocab_perception_pipeline/docs/assets/detections/output"
    crop_images(input_folder, output_folder)
