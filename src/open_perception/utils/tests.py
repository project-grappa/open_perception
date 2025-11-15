import numpy as np
import cv2
import os
import shutil

from open_perception.utils.common import recursive_filter_keys
from open_perception.utils.visualization import draw_elements_detections
from open_perception.utils.common import detections_to_elements
from open_perception.utils.vlm import vlm_messages_to_pdf, debug_info_to_pdf
from open_perception import __file__ as open_perception_path

tests_path = os.path.abspath(
    os.path.join(os.path.dirname(open_perception_path), "..", "..", "tests")
)


def get_dummy_image(name="cat_and_dog.png"):
    dummy_image = cv2.imread(tests_path + f"/{name}")
    print(dummy_image.shape)
    return dummy_image


def clear_detections_folder(test_name=None):
    output_folder = os.path.join(tests_path, "detections")
    if test_name is not None:
        output_folder = os.path.join(output_folder, test_name)
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)


def save_detection_overlay(image, detections, detector_name, test_name):
    """
    Save detection overlay image with bounding boxes and labels.
    Args:
        image (np.ndarray): Input image on which detections are drawn.
        detections (list[dict]): List of detections, each containing 'bbox', 'class', 'prob', and 'id'.
        detector_name (str): Name of the detector.
        test_name (str): Name of the test, used to create a subfolder for saving.
    """

    overlay = image.copy()
    elements = detections_to_elements(detections)
    overlay = draw_elements_detections(overlay, elements)

    output_folder = os.path.join(tests_path, f"detections/{test_name}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{detector_name}.png")
    cv2.imwrite(output_path, overlay)


def save_debug_info(debug_infos, detector_name, test_name, format="pdf"):
    """
    Save debug information to a file in the specified format (txt or pdf).
    Args:
        debug_infos (list[dict]): Debug information to be saved.
        detector_name (str): Name of the detector.
        test_name (str): Name of the test.
        format (str): Format to save the debug info, either "txt" or "pdf".
    """
    assert format in ["txt", "pdf"], "Format must be either 'txt' or 'pdf'"

    output_folder = os.path.join(tests_path, f"detections/{test_name}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{detector_name}_debug_info.{format}")
    debug_info_list = [debug_infos] if isinstance(debug_infos, dict) else debug_infos
    if format == "txt":
        with open(output_path, "w") as f:
            for d in debug_info_list:
                f.write(str(d) + "\n")
    elif format == "pdf":
        # Convert debug info to PDF
        print(f"Saving debug info to PDF at {output_path}")
        debug_info_to_pdf(
            debug_info_list,
            pdf_file_path=output_path,
            title=f"TEST: {test_name} - CLASS: {detector_name}",
        )


def save_vlm_messages(messages, detector_name, test_name, format="pdf"):
    """
    Save VLM messages to a file in the specified format (txt or pdf).
    Args:
        messages (list[dict]): List of VLM messages.
        detector_name (str): Name of the detector.
        test_name (str): Name of the test.
        format (str): Format to save the messages, either "txt" or "pdf".
    """
    assert format in ["txt", "pdf"], "Format must be either 'txt' or 'pdf'"

    output_folder = os.path.join(tests_path, f"detections/{test_name}")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{detector_name}_vlm_messages.{format}")
    if format == "txt":
        with open(output_path, "w") as f:
            for message in messages:
                f.write(str(message) + "\n")
    elif format == "pdf":
        # Convert messages to PDF
        print(f"Saving VLM messages to PDF at {output_path}")
        vlm_messages_to_pdf(
            messages,
            pdf_file_path=output_path,
            title=f"TEST: {test_name} - CLASS: {detector_name}",
        )


def assert_output_format(detections):
    for detection in detections:
        assert "bbox" in detection
        assert isinstance(detection["bbox"], np.ndarray)
        assert detection["bbox"].shape == (2, 2)
        assert "class" in detection
        assert isinstance(detection["class"], str)
        assert "prob" in detection
        assert isinstance(detection["prob"], float)
        assert 0 <= detection["prob"] <= 1
        assert "id" in detection
        assert isinstance(detection["id"], int)
