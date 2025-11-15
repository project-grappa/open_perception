import requests
import json
import numpy as np
from typing import Any, List, Dict
from open_perception.pipeline.element import Element
from open_perception.communication.base_client import BaseClient
from open_perception.logging.base_logger import Logger
from datetime import datetime
from open_perception.communication.api_interface import APIConfigParser


class APIClient(BaseClient, APIConfigParser):
    def __init__(self, config: Dict = {}):
        super().__init__()
        self.host = config.get("host", "127.0.0.1")
        self.port = config.get("port", 5000)
        self.base_url = f"http://{self.host}:{self.port}"
        self.logger = Logger.get_logger(__name__, config.get("logging", None))

    def start(self):
        """Start the API interface."""
        response = requests.post(f"{self.base_url}/start", json={})
        self.logger.info(f"[APIClient] Start response: {response.json()}")

    def stop(self):
        """Stop the API interface."""
        response = requests.post(f"{self.base_url}/stop", json={})
        self.logger.info(f"[APIClient] Stop response: {response.json()}")
    
    def get_frame(self) -> Dict[str, Any]:
        """Retrieve an image/frame from the API interface."""
        response = requests.get(f"{self.base_url}/frame")
        return response.json()

    def get_point_cloud(self) -> Dict[str, Any]:
        """Retrieve a point cloud from the API interface."""
        response = requests.get(f"{self.base_url}/point_cloud")
        return response.json()

    def get_synced_frame_and_pc(self) -> Dict[str, Any]:
        """Retrieve an image/frame and point cloud from synchronized sensors."""
        response = requests.get(f"{self.base_url}/synced_frame_and_pc")
        return response.json()

    def locate(self, obj_name: str,
               img: np.ndarray = None,
               point_cloud: np.ndarray = None,
               segment: bool = True,
               compute_pose: bool = True,
               track: bool = True,
               wait: bool = True,
               timeout: float = 15,
               temporary: bool = False) -> List[Element]:
        """Locate an object in an image."""

        if img is not None and point_cloud is not None:
            self.send_frame("front", rgb=img, point_cloud=point_cloud, wait=True)

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

        response = requests.post(f"{self.base_url}/locate", json=request)
        detections = response.json().get("detections", [])
        elements = [Element(**detection) for detection in detections]
        return elements

if __name__ == "__main__":
    config = {
        "host": "127.0.0.1",
        "port": 5000
    }
    client = APIClient(config=config)
    client.start()

    # Example usage
    frame = client.get_frame()
    point_cloud = client.get_point_cloud()
    synced_data = client.get_synced_frame_and_pc()

    print("Frame:", frame)
    print("Point Cloud:", point_cloud)
    print("Synced Data:", synced_data)

    client.stop()
