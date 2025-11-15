"""
redis_interface.py

Implements a RedisInterface class to handle pub/sub communication.
"""

import subprocess
import redis
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
class RedisConfigParser:
    def __init__(self, config={}):
        # Parse configuration
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 6379)
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

class RedisInterface(BaseInterface, RedisConfigParser):
    def __init__(self, config={}):
        """
        Initialize the Redis interface.
        :param config: Nested Dictionary with configuration parameters. Example:
        {
            "enabled": False,
            "host": "127.0.0.1",
            "port": 6379,
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
        RedisConfigParser.__init__(self, config)
        # Parse configuration
        # self.host = self.config.get("host", "127.0.0.1")
        # self.port = self.config.get("port", 6379)
        # self.channels_config = self.config.get("channels", {})
        # self.channels = list(get_dict_leafs(self.config.get("channels", [])))
        # self.cameras = self.channels_config.get("cameras", [])
        
        # self.locate_channel = self.channels_config.get("locate", "locate")
        # self.remove_channel = self.config.get("remove", "remove")
        # self.acknowledgement_channel = self.channels_config.get("acknowledgement", "perception_ak")
        # self.reset_channel = self.channels_config.get("reset", "reset")

        # # stores the last received frames for each camera for syncing purposes
        # self.channels_to_camera = {}
        # # self.camera_buffers = {}
        # # self.camera_buffer_size = 10  # You can set this to any desired number of frames

        # for camera_name, camera in self.cameras.items():
        #     # self.camera_buffers[camera_name] = {}
        #     for modality, channel in camera.items():
        #         self.channels.append(channel+"_meta")
        #         # self.camera_buffers[camera_name][modality] = collections.deque(maxlen=10)
        #         #check if for repeated channels
        #         if channel in self.channels_to_camera.keys():
        #             raise ValueError("Repeated channels in configuration, each sensor of each camera should have an unique name")
                
        #         self.channels_to_camera[channel+"_meta"] = {"camera": camera_name, "modality": modality}

        self._redis_client = None
        self._pubsub = None
        self._subscribe_thread = None
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


        self.logger = Logger.get_logger(type(self).__name__, config.get("logging", None))
        
    def start(self):
        """
        Establishes Redis connection and starts the subscription thread if channels are provided.
        """
        # 1. Create a Redis client
        try:
            self._redis_client = redis.Redis(host=self.host, port=self.port)
            self._redis_client.ping()
            self.logger.info("Successfully connected to redis")
        except Exception:
            
            # launch redis server as subprocess
            self.logger.info(f"starting redis server on port {self.port}")
            subprocess.Popen(["redis-server", "--port", str(self.port)])
            time.sleep(3)
            os.system("redis-cli CONFIG SET protected-mode no")
            self._redis_client = redis.Redis(host=self.host, port=self.port)
        self.reset_redis()
        
        # 2. If we have channels to subscribe to, set up pubsub
        if self.channels:
            self._pubsub = self._redis_client.pubsub()
            for channel in self.channels:
                if channel in [self.all_detections_channel, self.acknowledgement_channel]:
                    continue
                self._pubsub.subscribe(channel)
            
            self._running = True
            self._subscribe_thread = threading.Thread(target=self._subscriber_loop, daemon=True)
            self._subscribe_thread.start()

        self.logger.info(f"[RedisInterface] Connected to Redis at {self.host}:{self.port}")
        if self.channels:
            self.logger.info(f"[RedisInterface] Subscribed to channels: {self.channels}")
            
    def _subscriber_loop(self):
        """
        Continuously listens for messages on subscribed channels in a separate thread.
        """
        while True:
            with self.lock:
                if not self._running:
                    break
            # check if new messages are available
            self._retrieve_messages()
            if self.detections_updated:

                with self.lock:
                    detections = self.detections
                    id = self.detections_id
                for det in detections:
                    # publish the detections to redis using pickle
                    self.publish(det.class_name, json.dumps(cast_serializable(det.to_dict())))
                
                self.logger.info(f"Published {len(detections)} detections to Redis")
                self.publish(self.all_detections_channel, pickle.dumps([el.to_dict() for el in detections], protocol=pickle.HIGHEST_PROTOCOL))
                self.logger.info(f"\n\n\nPublished detections, len:  {len(detections)}")
                ak = {'id': id, 'stamp': datetime.timestamp(datetime.now())}
                self.publish(self.acknowledgement_channel, json.dumps(ak))
                
                with self.lock:
                    self.detections_updated = False
            time.sleep(0.001)
            # time.sleep(0.01)  # Sleep briefly to yield control


    def _retrieve_messages(self):

        message = self._pubsub.get_message()
        if message:
            if message['type'] == 'message':           
                self.handle_message(message)

    def _clear_channel(self, channel):
        """
        Clear any messages on the specified channel.
        """
        self._redis_client.delete(channel)

    def handle_message(self, message):
        """
        Callback for processing incoming messages.
        """
        channel = message['channel'].decode('utf-8')
        data_raw = message['data']
        self.logger.info(f"Received message on channel {channel}, {message['type']}")
        if not data_raw:
            self.logger.error(f"[RedisInterface] Error retrieving data from channel: {channel}")
            return
        
        # handle each possible message topic
        if channel == self.locate_channel:
            # data_raw = self._redis_client.get(channel)
            data = json.loads(data_raw)
            if not isinstance(data, list):
                data = [data]
            for obj_data in data:
                class_name = obj_data.get("object_name")
                if not class_name:
                    self.logger.error(f"[RedisInterface] Error retrieving obj_data from 'locate' channel, 'class_name': {class_name} in {data_raw}")
                
                class_info = {
                    "class_name": class_name,
                    "compute_pose": obj_data.get("compute_pose", True),
                    "compute_mask": obj_data.get("compute_mask", True),
                    "track": obj_data.get("track", True)
                    }
                with self.lock:
                    self.objects_to_add.append(class_info)
                self.logger.info(f"Received locate request for object: {class_name}")
                
        elif channel == self.reset_channel:
            self._trigger_reset()
            self._clear_channel(channel)

        elif channel == self.remove_channel:
            # data_raw = self._redis_client.get(channel)
            data = json.loads(data_raw)
            class_name = data
            
            with self.lock:
                self.objects_to_remove.append(class_name)
            # self._clear_channel(channel)

        elif channel in self.channels_to_camera.keys():
            # self.logger.info("[RedisInterface] Received meta data from camera sensor")

            # camera meta data
            camera = self.channels_to_camera[channel]["camera"]
            modality = self.channels_to_camera[channel]["modality"]

            self.logger.info(f"Received data for camera: {camera}, modality: {modality}")
            if "_meta" in channel:
                meta_data_raw = self._redis_client.get(channel) # get the meta data
                camera_data_raw = self._redis_client.get(channel[:-5]) # get the data from the actual camera channel
                if not meta_data_raw or not camera_data_raw:
                    self.logger.error(f"[RedisInterface] Error retrieving data for camera: {camera}, modality: {modality}")
                    self.logger.error(f"[RedisInterface] Error retrieving data for camera: {camera}, modality: {modality}")
                    return
                meta_data = json.loads(meta_data_raw)
                # parse the data modality
                if "rgb" in modality:
                    camera_data = self._decode_rgb(camera_data_raw)
                elif "depth" in modality:
                    camera_data = self._decode_depth(camera_data_raw)
                elif "point_cloud" in modality:
                    camera_data = self._decode_point_cloud(camera_data_raw)
                else:
                    camera_data = None
                # update available detections if any
                if "detections" in meta_data.keys():
                    self.logger.info(f"[RedisInterface] Received detections for camera: {camera}, modality: {modality}")
                    detections = meta_data["detections"]
                    with self.lock:
                        self.updates["available_detections"] = detections
                # update circular camera buffer
                if camera_data is not None:
                    with self.lock:
                        if "rgb" in modality: # update default frame index counter
                            self.frame_idx += 1
                        self.camera_buffers[camera][modality].append({"data": camera_data, "meta": meta_data})
                else:
                    self.logger.error(f"[RedisInterface] Error decoding data for camera: {camera}, modality: {modality}")
        
        elif channel == self.update_config_channel:
            data = json.loads(data_raw)
            self.logger.info(f"[RedisInterface] Updating configuration: {data}")
            self.updates["config"] = data
        else:
            # other data
            pass
    
    # ----------------------- Decoding methods -----------------------
    def _decode_rgb(self, encoded):
        h, w = struct.unpack('>II',encoded[:8])
        rgb = np.frombuffer(encoded, dtype=np.uint8, offset=8).reshape(h,w,3)
        return rgb
    
    def _decode_depth(self, encoded):
        h, w = struct.unpack('>II',encoded[:8])
        depth = np.frombuffer(encoded, dtype=np.uint8, offset=8).reshape(h,w,1)
        return depth
    
    def _decode_point_cloud(self, encoded):
        h, w = struct.unpack('>II',encoded[:8])
        pc = np.frombuffer(encoded, dtype=np.float64, offset=8).reshape(h,w,3)
        return pc
    
    
    def reset_redis(self):
        self.logger.info("Resetting redis")
        self._redis_client.flushall()
        # remove all objects with keys on redis
        keys = self._redis_client.keys()
        # self.logger.debug("redis keys: ", keys)
        for key in keys:
            key_dec = key.decode("utf-8")
            if key_dec in self.active_objects.keys() or key in self.active_objects.keys():
                self.logger.info(f"Deleting key: {key_dec}")
                self._redis_client.delete(key_dec)


    def publish(self, channel, message):
        """
        Publish a message to a Redis channel.
        :param channel: The channel name as a string
        :param message: The message to publish (string, bytes, etc.)
        """
        if not self._redis_client:
            raise ConnectionError("[RedisInterface] Redis client not initialized. Call start() first.")
        self._redis_client.publish(channel, message)


    # ====================================== main interface methods ======================================
    def _get_frames_dict(self, modality_name: str):
        """
        Get the camera and modality from the modality name.
        """
        camera_frames = {}
        with self.lock:
            for camera_name, camera in self.camera_buffers.items():
                # for modality_name, modality in camera.items():
                if len(self.camera_buffers[camera_name].get(modality_name,[])) > 0:
                    frame = self.camera_buffers[camera_name][modality_name][-1]["data"]
                    index = self.camera_buffers[camera_name][modality_name][-1]["meta"].get("id",None)
                    camera_frames[camera_name] = {modality_name:frame, "index":int(index)}
        return camera_frames

    def get_point_cloud(self):
        """
        Retrieve a point cloud from the interface.
        """
        camera_frames = self._get_frames_dict("point_cloud")
        return camera_frames
    
    def get_frame(self):
        """
        Retrieve an image/frame from the interface.
        """
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
    
    def publish_results(self, detections: list[Element], id: int = 0, sensor_name:str = None):
        """
        Publish detection results to a remote service or communication channel.

        :param detections: List of Element objects representing detected objects.
        :param id: Optional ID of the detection result (e.g., frame index).
        :param sensor_name: Optional name of the sensor that produced the detections.
        """
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
        raise NotImplementedError

    
    def close(self):
        """
        Gracefully shuts down the subscription thread and closes the Redis connection.
        """
        self._running = False
        if self._subscribe_thread and self._subscribe_thread.is_alive():
            self._subscribe_thread.join(timeout=2)

        if self._pubsub:
            self._pubsub.close()

        if self._redis_client:
            self._redis_client.close()

        self.logger.info("[RedisInterface] Redis connection closed.")

    def update(self, updates=None):
        """
        Update the internal state or retrieve new data from the interface.
        """
        pass


if __name__ == "__main__":
    import os
    from open_perception.utils.config_loader import load_config
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_path = str(os.path.join(script_path, "../../../config/default.yaml"))
    config = load_config(config_path)
    redis_config = config["communication"]["redis"]
    interface = RedisInterface(config=redis_config)
    interface.start()
    time.sleep(60)

