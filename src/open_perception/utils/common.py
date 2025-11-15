"""
Common utility functions
"""

from typing import Dict, List
from open_perception.pipeline.element import Element
from datetime import datetime
import numpy as np
import json
import cv2
from .visualization import get_color


def get_dict_leafs(d: Dict):
    """
    Get all leaf nodes of a nested dictionary, list or tuples
    """
    if isinstance(d, dict):
        for k, v in d.items():
            yield from get_dict_leafs(v)
    elif isinstance(d, (list, tuple)):
        for item in d:
            yield from get_dict_leafs(item)
    else:
        yield d


def detections_to_elements(
    detections: list[Dict], frame: np.array = None, frame_idx=0, sensor_name: str = None
) -> list[Element]:
    """
    Convert a dictionary of detections to a list of Element objects
    """
    elements = []

    for i, det in enumerate(detections):
        element = Element(
            element_id=det["id"],
            class_name=det["class"],
            detection_prob=det["prob"],
            bbox=det["bbox"],
            detection_frame=frame,
            detection_frame_idx=frame_idx,
            is_parent=det.get("is_parent", False),
            color=get_color(i),
        )

        element.meta["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        element.meta["sensor"] = sensor_name
        elements.append(element)
    return elements


def recursive_filter_keys(message, keys_to_filter=["image_url", "image_urls"]):
    """
    Recursively filter out keys from a nested dictionary or list.
    """
    if isinstance(message, dict):
        return {
            k: recursive_filter_keys(v, keys_to_filter)
            for k, v in message.items()
            if k not in keys_to_filter
        }
    elif isinstance(message, list):
        return [recursive_filter_keys(item, keys_to_filter) for item in message]
    else:
        return message


def cast_serializable(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif "array" in str(type(obj)) or "ndarray" in str(type(obj)):
        return obj.tolist()
    elif "tensor" in str(type(obj)) or "Tensor" in str(type(obj)):
        return obj.cpu().detach().numpy().tolist()
    elif isinstance(obj, int) or isinstance(obj, str) or isinstance(obj, float):
        return obj
    elif isinstance(obj, dict):
        return {key: cast_serializable(obj[key]) for key in obj.keys()}
    elif isinstance(obj, list):
        return [cast_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple([cast_serializable(item) for item in obj])
    elif isinstance(obj, set):
        return set([cast_serializable(item) for item in obj])
    elif isinstance(obj, np.ndarray):
        return cast_serializable(obj.tolist())
    else:
        # check if is serializable
        try:
            json.dumps(obj)
            return obj
        except:
            print(f"Object of type {type(obj)} is not serializable")
            return str(obj)


def load_video_frames(video_path):  # using cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


from PIL import Image, ImageDraw


def combine_images_with_labels(
    images, vertical=False, index=True, max_size=(20000, 200), offset=1
):
    if vertical:
        # vertical concatenation
        all_images = Image.new(
            "RGB",
            (
                max([min(image.width, max_size[0]) + 10 for image in images]),
                sum([min(image.height, max_size[1]) + 20 for image in images]),
            ),
        )
    else:
        all_images = Image.new(
            "RGB",
            (
                sum([min(image.width, max_size[0]) + 10 for image in images]),
                max([min(image.height, max_size[1]) + 20 for image in images]),
            ),
        )
    # all_images.paste(images[0], (0, 0))
    total_dist = 0
    for i, image in enumerate(images):
        # scale image if it is bigger than max_size
        scale = 1
        if image.width > max_size[0] or image.height > max_size[1]:
            scale = min(max_size[0] / image.width, max_size[1] / image.height)

        if vertical:
            all_images.paste(
                image.resize((int(image.width * scale), int(image.height * scale))),
                (20 if index else 0, total_dist),
            )
        else:
            all_images.paste(
                image.resize((int(image.width * scale), int(image.height * scale))),
                (total_dist, 20 if index else 0),
            )
        # add a text above each image
        text = f"{i + offset}"
        image_draw = ImageDraw.Draw(all_images)
        image_draw.text(
            (0, total_dist) if vertical else (total_dist, 0), text, fill="white"
        ) if index else None
        total_dist += 10 + min(
            (image.height if vertical else image.width),
            max_size[1] if vertical else max_size[0],
        )
    return all_images


def combine_images_grid(images, ncols, max_size=(20000, 200)):
    if not isinstance(images[0], Image.Image):
        images_pil = [Image.fromarray(im) for im in images]
    else:
        images_pil = images
    combined_images = []
    for i in range(0, len(images), ncols):
        img = combine_images_with_labels(
            images_pil[i : i + ncols],
            vertical=False,
            index=True,
            max_size=max_size,
            offset=i + 1,
        )
        combined_images.append(img)

    final_image = combine_images_with_labels(
        combined_images, vertical=True, index=False, max_size=max_size
    )
    return final_image


def combine_images(images, border=5, bg=(0, 0, 0), save=None, vertical=False):
    # resize the images to have the same height
    height = max([im.shape[0] for im in images])
    width = max([im.shape[1] for im in images])
    if vertical:
        new_images = [
            cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])))
            for im in images
        ]
    else:
        new_images = [
            cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height))
            for im in images
        ]
    # border
    for i, im in enumerate(new_images[:-1]):
        if vertical:
            new_images[i] = cv2.copyMakeBorder(
                im, 0, border, 0, 0, cv2.BORDER_CONSTANT, value=bg
            )
        else:
            new_images[i] = cv2.copyMakeBorder(
                im, 0, 0, 0, border, cv2.BORDER_CONSTANT, value=bg
            )

    # concatenate the images
    if vertical:
        combined = np.vstack(new_images)
    else:
        combined = np.hstack(new_images)
    if save is not None:
        cv2.imwrite(save, combined)
    return combined
