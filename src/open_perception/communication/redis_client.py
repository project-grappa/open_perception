import pickle
import threading
from time import sleep
import redis
import json
import numpy as np
from typing import Any, List, Dict
from open_perception.pipeline.element import Element
from open_perception.communication.base_client import BaseClient
from open_perception.communication.redis_interface import RedisConfigParser
import struct
from open_perception.utils.common import cast_serializable
from open_perception.logging.base_logger import Logger
from datetime import datetime

class RedisClient(BaseClient, RedisConfigParser):
    def __init__(self, config: Dict = {}):
        super().__init__()

        # 
        RedisConfigParser.__init__(self, config)
        
        self._redis_client = None
        self.lock = threading.Lock()
        self._detections = []
        self._acknowledgement = None
        self.last_messages_timestamps = {}
        
        self.logger = Logger.get_logger(type(self).__name__, config.get("logging", None))

    def connect(self):
        """Establish a connection to the Redis server."""
        try:
            self._redis_client = redis.Redis(host=self.host, port=self.port)
            self._pubsub = self._redis_client.pubsub()
            
            self._pubsub.subscribe(self.all_detections_channel)
            self._pubsub.subscribe(self.acknowledgement_channel)

            # start a separate thread to listen for messages
            self._running = True
            if not hasattr(self, "_subscriber_thread") or not self._subscriber_thread.is_alive():
                self._subscriber_thread = threading.Thread(target=self._subscriber_loop)
                self._subscriber_thread.start()

            self.logger.info(f"[RedisClient] Connected to Redis at {self.host}:{self.port}")
        except redis.ConnectionError as e:
            self.logger.error(f"[RedisClient] Connection error: {e}")
            self._running = False

    def _subscriber_loop(self):
        """
        Continuously listens for messages on subscribed channels in a separate thread.
        """
        while True:
            with self.lock:
                if not self._running:
                    break
            try:
                # check if new messages are available
                message = self._pubsub.get_message()
                if message:
                    if message['type'] == 'message':           
                        self._handle_message(message)
                sleep(0.01)
            except redis.ConnectionError as e:
                self.logger.error(f"[RedisClient] Connection error in subscriber loop: {e}")
                with self.lock:
                    self._running = False

    def _handle_message(self, message):
        """Handle a message received from the Redis server."""
        channel = message["channel"].decode()
        data = message["data"]
        self.logger.info(f"[RedisClient] Received message on channel {channel}, with data len {len(data)}")

        known_channel = True
        if channel == self.all_detections_channel:
            with self.lock:
                self._detections = [Element.from_dict(el) for el in pickle.loads(data)]
            self.logger.info(f"[RedisClient] Received {len(self._detections)} detections.")
        elif channel == self.acknowledgement_channel:
            with self.lock:
                self._acknowledgement = data
        else:
            known_channel = False
            self.logger.info(f"[RedisClient] Received message on unknown channel: {channel}")

        self.last_messages_timestamps[channel] = datetime.timestamp(datetime.now())

    def disconnect(self):
        """Close the connection to the Redis server."""
        if self._redis_client:
            self._redis_client.close()
            self._redis_client = None
            self.logger.info("[RedisClient] Redis connection closed.")
        
        with self.lock:
            self._running = False


    @property
    def connected(self):
        """Check if the client is connected to the Redis server."""
        return self._redis_client is not None
    
    def _assert_connection(self):
        if not self.connected:
            raise ConnectionError("[RedisClient] Not connected to Redis server.")
        
    def remove(self, obj_names: list[str]) -> None:
        """Remove an object from the Redis server."""
        self._assert_connection()
        
        # self._redis_client.set(self.remove_channel, json.dumps(obj_names))
        self._redis_client.publish(self.remove_channel, json.dumps(obj_names))

    def locate(self, obj_name: str,
               img: np.ndarray = None,
               point_cloud: np.ndarray = None,
               segment: bool = True,
               compute_pose: bool = True,
               track: bool = True,
               wait: bool = True,
               timeout: float = 15,
               temporary: bool = False) -> list[Element]:
        """Locate an object in an image."""
        self._assert_connection()
        
        if img is not None and point_cloud is not None:
            self.send_frame("front", rgb=img, point_cloud=point_cloud, wait=True)

        # Implementation for locating an object
        # Publish the locate request to the Redis server
        if not isinstance(obj_name, list):
            obj_names = [obj_name]
        else:
            obj_names = obj_name

        request = [{
                "object_name": obj_name,
                "compute_pose": compute_pose,
                "compute_mask": segment,
                "track": track
        } for obj_name in obj_names]
        # print(request)
        # self._redis_client.set(self.locate_channel, json.dumps(request))
        self._redis_client.publish(self.locate_channel, json.dumps(request))
        
        if wait:
            # Wait for an acknowledgement after all objects have been located 
            self._wait_for_response(self.all_detections_channel, timeout)
        elements = self.get_detection_elements()
        return elements

    def _wait_for_response(self, channel: str, timeout: float, start_time: float = None) -> Any:
        """Wait for a response on the given Redis channel."""
        if start_time is None:
            _start_time = datetime.timestamp(datetime.now())
        else:
            _start_time = start_time
        self._assert_connection()
        
        #check if channel in pubsub channels
        new_channel = False
        self.logger.info("Waiting for new response on channel: ", channel, "start time: ", _start_time)
        if not str.encode(channel) in self._pubsub.channels:
            new_channel = True
            self.logger.info(f"Subscribing to {channel}")
        
        # message = None
        while (datetime.timestamp(datetime.now()) - _start_time) < timeout:
            with self.lock:
                last_message_timestamp = self.last_messages_timestamps
            if channel in last_message_timestamp:

                # wait for a new message to arrive
                if last_message_timestamp[channel] > _start_time:
                    break
            # print(last_message_timestamp)
            
            sleep(0.01)

        #     message = self._pubsub.get_message()
        #     if message and message["type"] == "message":
        #         message = message["data"]
        #         break
        #     sleep(0.001)
        if new_channel:
            self._pubsub.unsubscribe(channel)
        # return message
    
    def _publish_to_redis(self, data:np.ndarray , key_name: str, meta=None) -> None:
        """Store given Numpy array 'data' in Redis under key 'key_name'"""
        if self._redis_client is None:
            raise ConnectionError("Redis server not available")

        h, w = data.shape[:2]
        shape = struct.pack('>II',h,w)
        encoded = shape + data.tobytes()
        # self._redis_client.set(n,encoded)
        self._redis_client.set(key_name,encoded)    

        if meta is not None:
            meta = cast_serializable(meta)
            self._redis_client.set(key_name+"_meta",json.dumps(meta))
            self._redis_client.publish(key_name+"_meta",json.dumps(meta))

    def send_frame(self, sensor_name: str,
                   rgb: np.ndarray = None,
                   point_cloud: np.ndarray = None,
                   meta: Dict = None,
                   wait: bool = False,
                   fake_pc: bool = True) -> None:
        """Send a frame to the Redis server."""

        self._assert_connection()
        if meta is None:
            meta = {}
        meta["id"] = meta.get("id", -1)
        meta["stamp"] = meta.get("stamp", datetime.timestamp(datetime.now()))

        if rgb is not None:
            rgb = rgb.astype(np.uint8)
            assert rgb.ndim == 3, "RGB image must have 3 dimensions (height, width, channels)"
            assert rgb.shape[2] == 3, "RGB image must have 3 channels (R, G, B)"
            self._publish_to_redis(rgb, f"{sensor_name}_rgb", meta)
        if point_cloud is not None:
            point_cloud = point_cloud.astype(np.float64)
            assert point_cloud.ndim == 3, "Point cloud must have 3 dimensions (height, width, channels)"
            assert point_cloud.shape[2] == 3, "Point cloud must have 3 channels (x, y, z)"

            self._publish_to_redis(point_cloud, f"{sensor_name}_point_cloud", meta)
        elif fake_pc:
            fake_pc = np.zeros_like(rgb).astype(np.float64)
            self._publish_to_redis(fake_pc, f"{sensor_name}_point_cloud", meta)

        if wait:
            self._wait_for_response(self.acknowledgement_channel, timeout=15, start_time=meta["stamp"])

    def get_object_state(self, object_name):
        """Get the state of an object."""
        self._assert_connection()
        # query redis for the object with name object_name
        obj_state = self._redis_client.get(object_name)
        return obj_state

    def get_detection_elements(self) -> list[Element]:
        """Get the detection elements."""
        with self.lock:
            detections = self._detections

        return detections
    def reset(self) -> None:
        """Reset the Redis server."""
        self._assert_connection()
        self._redis_client.publish(self.reset_channel, 1)
        
        # Implementation for resetting the Redis server
        pass

    def update_config(self, config: Dict) -> None:
        """Update the configuration."""
        self._assert_connection()
        # self._redis_client.set(self.update_config_channel, json.dumps(config))
        self._redis_client.publish(self.update_config_channel, json.dumps(config))

if __name__ == "__main__":
    client = RedisClient()
    client.connect()

    # Send a frame
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    point_cloud = np.random.rand(480, 640, 3)
    client.send_frame("front", rgb=rgb, point_cloud=point_cloud)
    client.locate("car", segment=True, wait=True)