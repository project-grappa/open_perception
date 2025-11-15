"""
segmentation_models.py

This module contains the implementation of segmentation models for the Open Perception project.

Classes:
    SAM2Model: A segmentation model based on the SAM2 architecture. This model uses a detection bbox or mask as a prompt to start tracking objects in a video stream.

"""

import os
import numpy as np
from typing import List
from scipy.special import expit
from .base_models import BaseSegmentationModel

import cv2
import torch
from open_perception.logging.base_logger import Logger


class SAM2Model(BaseSegmentationModel):
    """
    Placeholder for SAM2 or an advanced segmentation model.
    """

    def __init__(self, model_config):
        super().__init__(model_config)
        logger_config = model_config.get("logging", None)
        self.logger = Logger.get_logger(type(self).__name__, logger_config)

    def load_model(self):
        file_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint_folder_path =os.path.join(file_path, "../../../checkpoints/")
        config_path = os.path.join(file_path, "../../../third_party/segment-anything-2-real-time/sam2/configs/")
        
        self.logger.info(self.model_config)

        ckpt_path = self.model_config.get("checkpoint","sam2.1_hiera_small.pt")
        if ckpt_path[0] != "/":
            ckpt_path = os.path.join(checkpoint_folder_path, ckpt_path)

        model_cfg = self.model_config.get("model_cfg",
                                os.path.join(config_path,"sam2.1/sam2.1_hiera_l.yaml"))

        self.logger.info(f"[SAM2Model] Loading SAM2 from {ckpt_path}...")
        self.logger.info(f"[SAM2Model] Using config file {model_cfg}...")
        try:
            from sam2.build_sam import build_sam2_camera_predictor
        except ImportError:
            self.logger.error(
                "SAM2 not found. Please install the package segment-anything-2-real-time."
            )
            raise ImportError
        
        self.logger.info(f"[SAM2Model] Loading SAM2 from {ckpt_path}...")
        self.model = build_sam2_camera_predictor(model_cfg, ckpt_path)
        self.frame_count = 0
        self.prompt_count = 0

    def update_thresholds(self, thresholds):
        # self.model.set_thresholds(thresholds)
        self.logger.info("[SAM2Model] Updating thresholds")

    def preprocess(self, frame):
        return frame

    def run_inference(self, frame, frame_idx):
        """
        Perform segmentation on a given frame.
        Return the seg_results
        """
        self.logger.info("[SAM2Model] Running inference")

        if self.prompt_count == 0:
            return []
        if frame is None:
            return []

        # 1. Preprocess
        frame = self.preprocess(frame)
        if len(self.model.condition_state.get("images", [])) == 0:
            self._load_first_frame(frame)

        # 2. Forward pass
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_obj_ids, out_mask_logits = self.model.track(frame)
        self.frame_count += 1

        seg_results = []
        for i in range(0, len(out_obj_ids)):
            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            ) * 255
            seg_result = {
                "mask": out_mask,
                "obj_id": out_obj_ids[i],
                "prob": np.mean(expit(out_mask_logits[i].cpu().numpy())),
                "frame_idx": frame_idx,
            }
            seg_results.append(seg_result)
        return seg_results

    def _load_first_frame(self, frame):
        self.logger.info("[SAM2Model] Loading first frame")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.model.load_first_frame(frame)

    def add_new_prompt(self, frame, ann_frame_idx, ann_obj_id, bbox=None, mask=None):
        self.logger.info("[SAM2Model] Adding new prompt")

        if len(self.model.condition_state.get("images", [])) == 0:
            self._load_first_frame(frame)
            self.frame_count += 1

        if self.model.condition_state["tracking_has_started"]:
            self.logger.info("[SAM2Model] Resetting model")
            self.model.reset_state()

        if bbox is not None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, out_obj_ids, out_mask_logits = self.model.add_new_prompt(
                    frame_idx=len(self.model.condition_state["images"]) - 1,
                    obj_id=ann_obj_id,
                    bbox=bbox,
                )
        if mask is not None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, out_obj_ids, out_mask_logits = self.model.add_new_mask(
                    frame_idx=len(self.model.condition_state["images"]) - 1,
                    obj_id=ann_obj_id,
                    mask=mask,
                )
        self.prompt_count += 1

    def reset(self):
        self.logger.info("[SAM2Model] Resetting model")
        self.model.condition_state = {}
        self.model.frame_idx = 0
        self.prompt_count = 0
        # if self.model.condition_state.get("tracking_has_started", None):
        #     self.model.reset_state()
        #     self.prompt_count = 0
        # else:
        #     self.logger.info("[SAM2Model] Model has not started tracking yet")
        # pass

