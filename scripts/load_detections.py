"""
given an output folder, load teh mask.pkl file, show it using cv2.imshow
Load the elements.pkl file and show the bounding boxes, segmentation masks, and label.
"""
from open_perception.utils.visualization import draw_elements_detections, draw_segmentations, get_color, merge_segmentations
import cv2
import numpy as np
import os
import pickle
from open_perception.pipeline.element import Element

def load_detections(output_folder):
    mask_path = os.path.join(output_folder, 'mask.pkl')
    elements_path = os.path.join(output_folder, 'elements.pkl')

    if not os.path.exists(mask_path) or not os.path.exists(elements_path):
        raise FileNotFoundError("mask.pkl or elements.pkl not found in the specified output folder")

    with open(mask_path, 'rb') as f:
        mask = pickle.load(f)

    with open(elements_path, 'rb') as f:
        elements = [Element.from_dict(el) for el in pickle.load(f)]


    # save mask as a npy
    mask = np.array(mask, dtype=np.uint8)
    cv2.imwrite(os.path.join(output_folder, 'mask.png'), mask)
    # save as .npy file
    np.save(os.path.join(output_folder, 'mask.npy'), mask)

    image = cv2.imread(os.path.join(output_folder, 'frame.png'))
    if image is None:
        # load image from first element
        image = elements[0].detection_frame
        if image is None:
            raise FileNotFoundError("image.png not found in the specified output folder, or in the first element")

    draw_segmentations(image, elements)
    draw_elements_detections(image, elements)

    cv2.imshow('Segmentations', mask)
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_folder = 'output'
    load_detections(output_folder)
