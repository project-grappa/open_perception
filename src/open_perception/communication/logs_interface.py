"""
logs_interface.py

Implements a LogsInterface class to load an analyze the logs of guidance system.
"""

import threading
import time
import numpy as np
from open_perception.communication.base_interface import BaseInterface
import cv2
import json
import os
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
# from motor_cortex.layers.utils import get_heatmap, plot_heatmaps, interpolate_guidance, plot_scene_with_heatmaps, compute_object_state, fig_to_array
# from motor_cortex.common.utils import as_video, combine_images

from typing import List
from open_perception.pipeline.element import Element

class LogsInterface(BaseInterface):
    def __init__(self, config=None):
        """
        Initialize the Log interface.
        :param config: TODO: complete this
        """
        super().__init__(config)
        self.exp_folder = config.get("exp_folder", "")
        self.policy = config.get("policy", "diffuser")
        self.policy_only = config.get("policy_only", False)
        self.time_steps = self._get_time_steps()
        self.current_time_step = 0

    def _get_time_steps(self):
        if self.policy_only:
            time_steps = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(f"{self.exp_folder}/detection/")]
        else:
            time_steps = [int(f.split("_")[-1].split(".")[0]) for f in os.listdir(f"{self.exp_folder}/guidance_data/") if "key_guidance_data" in f]
            time_steps = np.unique(time_steps)
        return time_steps

    def _load_exp_data(self, time_step, level=0):
        rgb_file = f"{self.exp_folder}/rgb/key_frame__{time_step:0>4}.png"
        rgb = plt.imread(rgb_file)

        pc_file = f"{self.exp_folder}/pc/key_frame_{time_step:0>4}.npz"
        pc = np.load(pc_file, allow_pickle=True)["data"]

        if self.policy_only:
            guidance_scores = None
            output = None
            flatten_states = None
            model_output = None
        else:
            guidance_data_file = f"{self.exp_folder}/guidance_data/key_guidance_data_{level}_{time_step:0>4}.pkl"
            with open(guidance_data_file, 'rb') as handle:
                guidance_data = pickle.load(handle)
            guidance_scores = torch.tensor(guidance_data["guidance_score"])
            output = guidance_data["output"]
            flatten_states = torch.tensor(guidance_data["flatten_states"])
            model_output = guidance_data["original_model_output"]

        obj_states_file = f"{self.exp_folder}/obj_states/key_states_{time_step:0>4}.npy"
        obj_states = np.load(obj_states_file, allow_pickle=True).item()

        meta_file = f"{self.exp_folder}/../../meta.json"
        with open(meta_file, 'r') as f:
            meta = json.load(f)

        info_file = f"{self.exp_folder}/guidance_info/key_guidance_info_{time_step:0>4}.json"
        with open(info_file, 'r') as f:
            info = json.load(f)

        return pc, rgb, guidance_scores, flatten_states, output, model_output, obj_states, meta, info

    def get_synced_frame_and_pc(self):
        if self.current_time_step >= len(self.time_steps):
            return None

        time_step = self.time_steps[self.current_time_step]
        pc, rgb, guidance_scores, flatten_states, output, model_output, obj_states, meta, info = self._load_exp_data(time_step)

        self.current_time_step += 1
        return {"front": {"rgb": rgb, "point_cloud": pc, "index": time_step}}

    def publish_results(self, detections: list[Element], id: int = 0, sensor_name:str = None):
        # Implement the method to publish results if needed
        pass

    def open(self):
        # Implement any initialization if needed
        pass

    def close(self):
        # Implement any cleanup if needed
        pass

    def update(self, updates=None):
        # Implement any updates if needed
        pass

