import pytest
import numpy as np
from open_perception.perception.vlm_based_models import VLMDetectionWithSearch, VLMVerifier, VLMDescriptionModel
from open_perception.perception.detection_models import DummyDetectionModel, GroundingDinoModelHF, MultigranularDetectionModel
from open_perception.utils.visualization import draw_elements_detections
from open_perception.utils.common import detections_to_elements
from open_perception.utils.vlm import vlm_messages_to_pdf
import numpy as np
import cv2
import os
import shutil
from open_perception.utils.config_loader import load_config
from open_perception.utils.tests import (
    clear_detections_folder,
    save_debug_info,
    save_detection_overlay,
    assert_output_format,
    get_dummy_image,
)

default_config = load_config()

@pytest.fixture
def dummy_image():
    dummy_image = get_dummy_image()
    return dummy_image

@pytest.fixture
def baseball_image():
    baseball_image = get_dummy_image("baseball.png")
    return baseball_image

@pytest.fixture
def detection_model():
    detection_model = GroundingDinoModelHF(model_config={})
    detection_model = MultigranularDetectionModel(model_config={}, detection_model=detection_model)
    detection_model.debug = True
    detection_model.display_debug_info = False

    return detection_model

@pytest.fixture
def dummy_detection_model():
    return DummyDetectionModel()

@pytest.fixture
def vlm_search_config():
    return default_config.pipeline.perception.vlm.search

@pytest.fixture
def vlm_config():
    return default_config.pipeline.perception.vlm.verification

@pytest.fixture
def vlm_description_config():
    return default_config.pipeline.perception.vlm.description

def test_vlm_search_run_inference(dummy_image, vlm_search_config, detection_model):
    test_name = "vlm_search_run_inference"
    clear_detections_folder(test_name)
    model = VLMDetectionWithSearch(
        vlm_config=vlm_search_config,
        detection_model=detection_model
    )
    model.debug = True
    model.display_debug_info = False

    model.load_model()
    model.update_classes(["eye of the gray animal"])
    detections = model.run_inference(dummy_image)
    assert isinstance(detections, list)
    assert all("bbox" in d and "class" in d for d in detections)
    # assert len(detections) == 2
    save_detection_overlay(
                dummy_image,
                detections,
                model.__class__.__name__,
                test_name,
            )
    save_debug_info(
        model.debug_info,
        model.__class__.__name__,
        test_name,
    )
    

def test_vlm_verification_run_inference(dummy_image, vlm_config, detection_model):
    test_name = "vlm_verification_run_inference"
    clear_detections_folder(test_name)

    model = VLMVerifier(
        vlm_config=vlm_config,
        detection_model=detection_model
    )
    model.debug = True
    model.display_debug_info = False

    model.load_model()
    model.update_classes(["left animal"])

    # get initial detections from base detection model for comparison
    base_model_detections = model.detection_model.run_inference(dummy_image)
    save_detection_overlay(
                dummy_image,
                base_model_detections,
                model.detection_model.__class__.__name__,
                test_name,
            )
    assert isinstance(base_model_detections, list)
    assert len(base_model_detections) > 1
    
    # run inference to deambiguate the detections
    detections = model.run_inference(dummy_image, precomputed_detections=base_model_detections)
    save_detection_overlay(
                dummy_image,
                detections,
                model.__class__.__name__,
                test_name,
            )
    save_debug_info(
        model.debug_info,
        model.__class__.__name__,
        test_name,
    )
    assert isinstance(detections, list)
    assert len(detections) == 1


def test_vlm_verification_wrong_detections(dummy_image, vlm_config, dummy_detection_model):
    test_name = "vlm_verification_wrong_detections"
    clear_detections_folder(test_name)
    
    model = VLMVerifier(
        vlm_config=vlm_config,
        detection_model=dummy_detection_model
    )
    model.debug = True
    model.display_debug_info = False

    model.load_model()
    model.update_classes(["bowl", "marker"]) # classes unrelated to the image
    
    # get initial random detections from the base model
    dummy_model_detections = model.detection_model.run_inference(dummy_image)
    save_detection_overlay(
                dummy_image,
                dummy_model_detections,
                model.detection_model.__class__.__name__,
                test_name,
            )
    assert isinstance(dummy_model_detections, list)
    assert len(dummy_model_detections) > 0
    
    # run inference with the dummy model detections trying to detect wrong detections
    model_detections = model.run_inference(dummy_image, dummy_model_detections)
    save_detection_overlay(
                dummy_image,
                model_detections,
                model.__class__.__name__,
                test_name,
            )
    save_debug_info(
        model.debug_info,
        model.__class__.__name__,
        test_name,
    )
    assert isinstance(model_detections, list)
    assert len(model_detections) == 0

def test_vlm_description(dummy_image, vlm_description_config, detection_model):
    test_name = "vlm_description"
    clear_detections_folder(test_name)

    model = VLMDescriptionModel(
        vlm_config=vlm_description_config,
        detection_model=detection_model
    )
    model.debug = True
    model.display_debug_info = False
    model.load_model()
    model.update_classes(["dog", "cat"])

    detections = model.run_inference(dummy_image)
    save_detection_overlay(
                dummy_image,
                detections,
                model.__class__.__name__,
                test_name,
            )
    save_debug_info(
        model.debug_info,
        model.__class__.__name__,
        test_name,
    )
    assert isinstance(detections, list)
    assert len(detections) > 0
    assert all("description" in d for d in detections)
    assert all("class" in d for d in detections)
    assert all("bbox" in d for d in detections)


def test_vlm_search_with_verification(baseball_image, vlm_config, vlm_search_config):
    test_name = "vlm_search_verification"
    clear_detections_folder(test_name)

    detection_model = GroundingDinoModelHF(model_config={"box_threshold": 0.15, "text_threshold": 0.15})
    detection_model = MultigranularDetectionModel(model_config={}, detection_model=detection_model)
    detection_model.debug = True
    detection_model.display_debug_info = False

    verifier_model = VLMVerifier(
        vlm_config=vlm_config,
        detection_model=detection_model
    )
    verifier_model.debug = True
    verifier_model.display_debug_info = False

    # uses a VLM verifier as a detection model, verifying every detection requested by the search model  
    model = VLMDetectionWithSearch(
        vlm_config=vlm_search_config,
        detection_model=verifier_model
    )
    model.debug = True
    model.display_debug_info = False

    model.load_model()
    model.update_classes(["guy in red on left"])
    detections = model.run_inference(baseball_image)
    # assert len(detections) == 2
    save_detection_overlay(
                baseball_image,
                detections,
                model.__class__.__name__,
                test_name,
            )
    save_debug_info(
        model.debug_info,
        model.__class__.__name__,
        test_name,
    )
    assert isinstance(detections, list)
    assert all("bbox" in d and "class" in d for d in detections)