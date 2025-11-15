"""
api_interface.py

Implements an APIInterface class to handle API communication.
"""

from flask import Flask, request, jsonify
import threading
import time
import numpy as np
from open_perception.communication.base_interface import BaseInterface
from open_perception.utils.common import get_dict_leafs
import cv2
import json
import collections
import struct
from typing import List
from open_perception.pipeline.element import Element
from open_perception.logging.base_logger import Logger
from datetime import datetime
from open_perception.utils.common import cast_serializable
import pickle

class APIConfigParser:
    def __init__(self, config={}):
        # Parse configuration
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 5000)
        self.channels_config = config.get("channels", {})
        self.channels = list(get_dict_leafs(config.get("channels", [])))
        self.cameras = self.channels_config.get("cameras", {})
        
        self.locate_channel = self.channels_config.get("locate", "locate")
        self.remove_channel = self.channels_config.get("remove", "remove")
        self.acknowledgement_channel = self.channels_config.get("acknowledgement", "perception_ak")
        self.reset_channel = self.channels_config.get("reset", "reset")
        self.update_config_channel = self.channels_config.get("update_config", "update_config")
        self.all_detections_channel = self.channels_config.get("all_detections", "all_detections")
        # stores the last received frames for each camera for syncing purposes
        self.channels_to_camera = {}

        for camera_name, camera in self.cameras.items():
            for modality, channel in camera.items():
                self.channels.append(channel+"_meta")
                #check if for repeated channels
                if channel in self.channels_to_camera.keys():
                    raise ValueError("Repeated channels in configuration, each sensor of each camera should have an unique name")
                
                self.channels_to_camera[channel+"_meta"] = {"camera": camera_name, "modality": modality}

class APIInterface(BaseInterface, APIConfigParser):
    def __init__(self, config={}):
        """
        Initialize the API interface.
        :param config: Nested Dictionary with configuration parameters. Example:
        {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 5000,
            "channels":
            "cameras":[
                "front": {
                    "rgb": "front_rgb",
                    "depth": "front_depth",
                    "point_cloud": "front_point_cloud",
                    "extrinsics": "front_extrinsics",
                    "intrinsics": "front_intrinsics"},
                "top":
                    "rgb": "top_rgb",
                    "depth": "top_depth",
                    "point_cloud": "top_point_cloud",
                    "extrinsics": "top_extrinsics",
                    "intrinsics": "top_intrinsics"
                ],
            "acknowledgement": "perception_ak",
            "info": "guidance_info"
            "locate": "locate" # locate the object in the image
            "remove": "remove" # remove the object from tracking list
            "reset": "reset" # reset the perception pipeline
        }
        """
        super().__init__(config)
        APIConfigParser.__init__(self, config)

        self._running = False
        self.frame_idx = 0

        self.detections_updated = False

        self.camera_buffers = {}
        self.camera_buffer_size = 10  # You can set this to any desired number of frames
        for channel_meta, channel_dict in self.channels_to_camera.items():
            camera_name = channel_dict["camera"]
            modality = channel_dict["modality"]
            if camera_name not in self.camera_buffers.keys():
                self.camera_buffers[camera_name] = {}
            self.camera_buffers[camera_name][modality] = collections.deque(maxlen=self.camera_buffer_size)

        self.logger = Logger.get_logger(__name__, config.get("logging", None))
        self.logger.debug("APIInterface initialized with config: %s", config)
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        self.logger.debug("Setting up routes for the API.")
        @self.app.route('/start', methods=['POST'])
        def start_interface():
            config = request.json
            self.start()
            return jsonify({"status": "started"}), 200

        @self.app.route('/stop', methods=['POST'])
        def stop_interface():
            self.stop()
            return jsonify({"status": "stopped"}), 200

        @self.app.route('/publish', methods=['POST'])
        def publish_results():
            data = request.json
            detections = [Element.from_dict(d) for d in data['detections']]
            id = data.get('id', 0)
            self.publish_results(detections, id)
            return jsonify({"status": "published"}), 200

        @self.app.route('/frame', methods=['GET'])
        def get_frame():
            frames = self.get_frame()
            return jsonify(frames), 200

        @self.app.route('/point_cloud', methods=['GET'])
        def get_point_cloud():
            point_cloud = self.get_point_cloud()
            return jsonify(point_cloud), 200

        @self.app.route('/synced_frame_and_pc', methods=['GET'])
        def get_synced_frame_and_pc():
            synced_data = self.get_synced_frame_and_pc()
            return jsonify(synced_data), 200

    def start(self):
        """
        Starts the API server.
        """
        self.logger.debug("Starting the API server.")
        self._running = True
        self.app.run(host=self.host, port=self.port)

    def stop(self):
        """
        Stops the API server.
        """
        self.logger.debug("Stopping the API server.")
        self._running = False
        #

    def _get_frames_dict(self, modality: str):
        """
        Get the camera and modality from the modality name.
        """
        self.logger.debug("Getting frames for modality: %s", modality)
        camera_frames = {}
        with self.lock:
            for camera_name, camera in self.camera_buffers.items():
                for modality_name, modality in camera.items():
                    if len(self.camera_buffers[camera_name].get(modality_name,[])) > 0:
                        frame = self.camera_buffers[camera_name][modality_name][-1]["data"]
                        index = self.camera_buffers[camera_name][modality_name][-1]["meta"].get("id",None)
                        camera_frames[camera_name] = {modality_name:frame, "index":int(index)}
        return camera_frames

    def get_point_cloud(self):
        """
        Retrieve a point cloud from the interface.
        """
        self.logger.debug("Retrieving point cloud.")
        camera_frames = self._get_frames_dict("point_cloud")
        return camera_frames
    
    def get_frame(self):
        """
        Retrieve an image/frame from the interface.
        """
        self.logger.debug("Retrieving frame.")
        camera_frames = self._get_frames_dict("rgb")
        return camera_frames   

    def get_synced_frame_and_pc(self):
        """
        Retrieve an image/frame and point cloud from synchronized sensors.

        :return: A dict of tuple of {sensor_name: (frame, point_cloud) where:
                 - frame is a NumPy array (H x W x C) in BGR, RGB, or grayscale.
                 - point_cloud is a structured point cloud (e.g., Nx3 NumPy array).
                 Return None if no frame or point cloud is available or on error.
        """
        self.logger.debug("Retrieving synchronized frame and point cloud.")
        camera_frames = {}
        with self.lock:
            for camera_name, camera in self.camera_buffers.items():

                last_frame_id = {}
                for modality_name, modality in camera.items():
                    if modality_name == "depth":
                        continue # ignore depth modality

                    if len(self.camera_buffers[camera_name][modality_name]) == 0:
                        last_frame_id[modality_name] = None
                        continue
                    last_frame_id[modality_name] = self.camera_buffers[camera_name][modality_name][-1]["meta"].get("id",None)
                
                # check if any modality is none
                if None in last_frame_id.values():
                    continue
                
                # check if the ids are different
                if len(set(last_frame_id.values()))>1:
                    continue

                frame_rgb = self.camera_buffers[camera_name]["rgb"][-1]["data"]
                pc = self.camera_buffers[camera_name]["point_cloud"][-1]["data"]
                index = last_frame_id["rgb"]
                if index == -1: # use default frame index counter
                    index = self.frame_idx
                camera_frames[camera_name] = {"rgb":frame_rgb, "point_cloud":pc, "index":int(index)}

        return camera_frames
    
    def publish_results(self, detections: List[Element], id:int=0):
        """
        Publish detection results to a remote service or communication channel.

        :param detections: List of Element objects representing detected objects.
        :param id: Optional ID of the detection result (e.g., frame index).
        """
        self.logger.debug("Publishing results with id: %d", id)
        with self.lock:
            self.computing = False
            self.detections = detections
            self.detections_id = id
            self.detections_updated = True

    def open(self):
        """
        Perform any initialization or resource allocation required to start
        retrieving sensor data or controlling the interface.

        For example, in a ROS-based interface, this might start subscribers;
        in a camera-based interface, it might open a device or file.
        """
        self.logger.debug("Opening the API interface.")
        raise NotImplementedError

    def close(self):
        """
        Gracefully shuts down the API server.
        """
        self.logger.debug("Closing the API interface.")
        self._running = False
        self.logger.info("[APIInterface] API server stopped.")

    def update(self, updates=None):
        """
        Update the internal state or retrieve new data from the interface.
        """
        self.logger.debug("Updating the API interface with updates: %s", updates)
        pass

if __name__ == "__main__":
    import os
    from open_perception.utils.config_loader import load_config
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_path = str(os.path.join(script_path, "../../../config/default.yaml"))
    config = load_config(config_path)
    api_config = config["communication"]["api"]
    
    # Enable the API interface
    api_config["enabled"] = True
    api_config["logging"] = {"level": "DEBUG", "format": "%(asctime)s - %(message)s"}
    
    interface = APIInterface(config=api_config)
    interface.logger.setLevel("DEBUG")
    interface.start()