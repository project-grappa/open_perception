"""
test_detection_models.py

This module contains tests for the detection models in the Open Perception project.
It verifies the following:
 - Test if all detection models can be loaded.
 - Output of all detection models is in the correct format.
    [{  "bbox": List of np.array [[x1, y1], [x2, y2]] of ints within the image size,
        "class": str
        "prob": float between 0 and 1
        "id": int
    }, ...]

 - Capability of all detection models to detect a simple object in a test image.
 - Capability of all detection models to detect multiple objects in a test image.
 - Capability of integration with the multi-granular object detection model.
    Detecting an object though a chain of parent classes e.g. dog.eyes or dog.nose
"""

import pytest
from pytest_subtests import SubTests
from open_perception.perception.detection_models import (
    DummyDetectionModel,
    GroundingDinoModel,
    GroundingDinoModelHF,
    YoloWorldModel,
    MultigranularDetectionModel,
    LookUpDetectionModel,
)
import numpy as np
import cv2
import os
import shutil
from open_perception.utils.tests import (
    clear_detections_folder,
    save_detection_overlay,
    assert_output_format,
    get_dummy_image,
)

@pytest.fixture
def dummy_image():
    dummy_image = get_dummy_image()
    return dummy_image
    # return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)


@pytest.fixture
def detection_models():
    return [
        DummyDetectionModel(model_config={}),
        # GroundingDinoModel(model_config={}),
        GroundingDinoModelHF(model_config={}),
        YoloWorldModel(model_config={}),
        # LookUpDetectionModel(model_config={}),
    ]


# @pytest.fixture(scope="session", autouse=True)
# def clear_detections_folder_fixture():
#     clear_detections_folder()


def test_load_models(detection_models, subtests: SubTests):
    for model in detection_models:
        model_class = model.__class__.__name__
        with subtests.test(msg=model_class):
            model.load_model()
            assert model.model is not None


def test_simple_object_detection(detection_models, dummy_image, subtests: SubTests):
    for model in detection_models:
        model_class = model.__class__.__name__
        with subtests.test(msg=model_class):
            model.load_model()
            model.update_classes(["dog"])
            detections = model.run_inference(dummy_image)
            assert len(detections) > 0
            assert_output_format(detections)

            # save detection overlay
            save_detection_overlay(
                dummy_image,
                detections,
                model.__class__.__name__,
                "simple_object_detection",
            )


def test_multiple_objects_detection(detection_models, dummy_image, subtests: SubTests):
    clear_detections_folder("multiple_objects_detection")
    for model in detection_models:
        model_class = model.__class__.__name__
        with subtests.test(msg=model_class):
            model.load_model()
            model.update_classes(["cat", "dog"])
            detections = model.run_inference(dummy_image)
            assert len(detections) > 1
            assert_output_format(detections)
            
            # save detection overlay
            save_detection_overlay(
                dummy_image,
                detections,
                model.__class__.__name__,
                "multiple_objects_detection",
            )


def test_multigranular_detection(detection_models, dummy_image, subtests: SubTests):
    clear_detections_folder("multigranular_detection")

    for base_model in detection_models:
        model_class = base_model.__class__.__name__
        with subtests.test(msg=model_class):
            model = MultigranularDetectionModel(detection_model=base_model)
            model.load_model()
            model.update_classes(["dog.eyes", "dog.nose"])
            detections = model.run_inference(dummy_image)
            assert len(detections) > 0
            assert_output_format(detections)

            # save detection overlay
            save_detection_overlay(
                dummy_image, detections, model_class, "multigranular_detection"
            )
