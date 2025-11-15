"""
base_models.py

Provides abstract base classes for perception models.
"""

import os
import abc
import numpy as np
from typing import List, Optional, Any

import cv2
import torch


# =============================================================================
# Abstract Base Classes
# =============================================================================


class BaseModel(abc.ABC):
    """
    The top-level abstract base class. It enforces a uniform interface
    for all perception models in the pipeline.
    """

    def __init__(self, model_config=None):
        """
        :param model_config: A dictionary containing model-specific configurations
                             (paths, hyperparameters, etc.).
        """
        self.model_config = model_config if model_config else {}
        self.model = None  # Placeholder for the loaded model

    @abc.abstractmethod
    def load_model(self):
        """
        Load the underlying ML model and store it in self.model.
        Must be implemented by derived classes.
        """
        pass

    @abc.abstractmethod
    def run_inference(self, image):
        """
        Run inference on a given image (or batch of images).
        Must be implemented by derived classes.

        :param image: A NumPy array or similar, representing an image.
        :return: Model-specific output (could be bounding boxes, masks, etc.)
        """

        raise NotImplementedError

    def preprocess(self, image):
        """
        Optional: Preprocess an input image before feeding it into the model.
        Can be overridden by subclasses or left as-is if not needed.

        :param image: Input image in a NumPy array or similar format.
        :return: Processed image
        """
        return image

    def postprocess(self, raw_output):
        """
        Optional: Postprocess model outputs before returning them.
        Can be overridden by subclasses or left as-is if not needed.

        :param raw_output: The raw output from the model's forward pass.
        :return: Postprocessed results
        """
        return raw_output


class BaseDetectionModel(BaseModel):
    """
    Abstract class for detection models.
    Enforces that detection models return bounding boxes, classes, and probs.
    """
    debug = False  # Enable debug mode for additional logging
    display_debug_info = True # Whether to display debug information or just collect it
    debug_info: Optional[list[dict]] = None  # Placeholder for debug information

    @abc.abstractmethod
    def run_inference(self, image, **kwargs):
        """
        :param image: Input image as a NumPy array (H x W x C).
        :return:
            detections: list of dicts or a structured array, each entry containing:
                {
                  'bbox': [x1, y1, x2, y2],
                  'class': str or int,
                  'prob': float
                }
        """
        pass

    @abc.abstractmethod
    def update_classes(self, classes: list[str]):
        pass


class BaseSegmentationModel(BaseModel):
    """
    Abstract class for segmentation models.
    Enforces that segmentation models return masks (or similar).
    """

    @abc.abstractmethod
    def run_inference(self, image):
        """
        :param image: Input image as a NumPy array (H x W x C).
        :return:
            masks: Typically a list of binary masks or a single multi-class mask.
        """
        pass

