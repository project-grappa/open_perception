import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod
from open_perception.pipeline.element import Element
from open_perception.utils.visualization import draw_segmentations, draw_elements_detections
class BaseClient:
    def __init__(self, config: Dict = {}):
        self.config = config
        self._running = True

    @property
    def connected(self):
        """Check if the client is connected to the interface server."""
        pass

    def connect(self):
        """Establish a connection to the interface server."""
        raise NotImplementedError


    def disconnect(self):
        """Close the connection to the interface server."""
        raise NotImplementedError

    @abstractmethod
    def locate(self, obj_name: str,
                    img: np.ndarray = None,
                    segment: bool = True,
                    wait: bool = True,
                    timeout: float=15,
                    temporary: bool=False) -> list[Element]:
        """Locate an object in an image.
        param obj_name: Name of the object to locate. or sequence of parent and child objects to locate.
        param img: numpy array image in which to locate the object.
        param segment: Whether to perform the segmentation or not. Default is True.
        param wait: Whether to wait for the result or not. Default is True.
        param timeout: Maximum time to wait for the result. Default is 15 seconds.
        param temporary: Whether the object is temporary or not. Default is False

        return: list of detection elements
        """
        raise NotImplementedError
        elements = []
        return elements


    @abstractmethod
    def send_frame(self, sensor_name:str,
                         rgb: np.ndarray=None,
                         point_cloud: np.ndarray=None,
                         meta: Dict = None) -> None:
        """Send a frame to the server.
        param sensor_name: Name of the sensor.
        param rgb: RGB image frame.
        param point_cloud: Point cloud data.
        param meta: Additional metadata.
        """
        raise NotImplementedError
        pass

    @abstractmethod
    def get_object_state(self, object_name: str) -> Dict:
        """Get the state of an object."""
        raise NotImplementedError

    @abstractmethod
    def get_objects(self) -> list[str]:
        """Get the list of objects being tracked."""
        raise NotImplementedError

    def reset(self) -> None:
        """Resets the perception pipeline."""
        raise NotImplementedError
    
    def update_logging(self, logging_config: Dict) -> None:
        """Update the logging configuration.
        param logging_config: Logging configuration.
        """
        raise NotImplementedError
    
    def generate_masked_frame(self, elements: list[Element], img: np.ndarray) -> np.ndarray:
        """Convert detection elements to a masked frame."""
        masked_frame = draw_elements_detections(img, elements)
        masked_frame = draw_segmentations(masked_frame, elements)
        return masked_frame
    
    def __del__(self):
        self._running = False