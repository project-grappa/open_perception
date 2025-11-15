"""
base_interface.py

Defines an abstract base class for perception interfaces. It outlines
the required methods for:
- Retrieving sensor data (frames, point clouds)
- Managing objects or classes to be detected
- Resetting detection/segmentation modules
- General lifecycle (e.g., open/close the interface)

Concrete interfaces (e.g., ROS, direct camera, file-based, etc.) should
subclass BaseInterface and implement these methods.
"""

import abc
import threading
from typing import List
from open_perception.pipeline.element import Element

class BaseInterface(abc.ABC):
    def __init__(self, config=None):
        """
        :param config: Optional dictionary for interface configuration
                       (e.g., camera settings, topic names, etc.).
        """
        self.config = config if config else {}
        # You could store or manage a list of currently "active" objects to detect
        self.active_objects = []
        
        # Whether the interface is "open" or initialized
        self.is_open = False
        self.frame = None
        self.point_cloud = None
        self.detections = []

        self.objects_to_add = []
        self.objects_to_remove = []

        self.updates = {}

        self.lock = threading.Lock()

    def is_enabled(self):
        """
        Check if the interface is enabled or configured.

        :return: True if the interface is enabled or configured, False otherwise.
        """
        enabled = self.config.get("enabled", False)
        return enabled
    
    @abc.abstractmethod
    def open(self):
        """
        Perform any initialization or resource allocation required to start
        retrieving sensor data or controlling the interface.

        For example, in a ROS-based interface, this might start subscribers;
        in a camera-based interface, it might open a device or file.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """
        Release resources or shut down connections. After calling this,
        get_frame() or get_point_cloud() may no longer be valid.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    def get_synced_frame_and_pc(self):
        """
        Retrieve an image/frame and point cloud from synchronized sensors.

        :return: A dict of tuple of {sensor_name: (frame, point_cloud) where:
                 - frame is a NumPy array (H x W x C) in BGR, RGB, or grayscale.
                 - point_cloud is a structured point cloud (e.g., Nx3 NumPy array).
                 Return None if no frame or point cloud is available or on error.
        """

        with self.lock:
            frame = self.frame if self.frame is not None else None
            pc = self.point_cloud if self.point_cloud is not None else None
            index = self.frame_index

        return {"front": {"rgb":frame, "point_cloud": pc, "index": index}}

    def get_frame(self):
        """
        Retrieve an image/frame from the sensor or data source.

        :return: A NumPy array (H x W x C) in BGR, RGB, or grayscale. 
                 Return None if no frame is available or on error.
        """
        with self.lock:
            frame = self.frame.copy() if self.frame is not None else None
        return frame
    

    def get_point_cloud(self):
        """
        Retrieve a point cloud from the sensor or data source.

        :return: A structured point cloud (e.g., Nx3 NumPy array.
                Return None if not available or unsupported.
        """
        with self.lock:
            pc= self.point_cloud.copy() if self.point_cloud is not None else None

        return pc

    def get_classes_to_add(self):
        """
        Retrieve the list of classes to add with additional information.
        """
        with self.lock:
            classes_to_add = self.objects_to_add.copy()
            self.objects_to_add.clear()
        return classes_to_add

    def get_classes_to_remove(self):
        """
        Retrieve the list of classes to remove.
        """

        with self.lock:
            classes_to_remove = self.objects_to_remove.copy()
            self.objects_to_remove.clear()
        return classes_to_remove

    def get_updates(self):
        """
        Retrieve any configuration updates or changes.

        :return: Dictionary of configuration updates (e.g., thresholds, settings).
        """
        with self.lock:
            updates = self.updates.copy()
            self.updates.clear()

        return updates
    
    def add_object(self, obj_name):
        """
        Add an object class or label to the list of objects to detect.

        :param obj_name: String name/label of the object or class to add.
        """
        with self.lock:
            
            self.objects_to_add.append(obj_name)

    def remove_object(self, obj_name):
        """
        Remove an object class or label from the list of objects to detect.

        :param obj_name: String name/label of the object or class to remove.
        """
        with self.lock:
            
            if obj_name in self.objects_to_add:
                self.objects_to_remove.append(obj_name)
        
    def remove_all_objects(self):
        """
        Clear all currently tracked/detected objects.
        """
        with self.lock:
            self.objects_to_remove = self.active_objects.copy()
    
    # @abc.abstractmethod
    def publish_results(self, detections: list[Element], id:int=0, sensor_name = None):
        """
        Publish detection results to a remote service or communication channel.

        :param detections: List of Element objects representing detected objects.
        :param id: Optional ID of the detection result (e.g., frame index).
        :param sensor_name: Optional name of the sensor that produced the detections.
        """
        raise NotImplementedError

    def update(self):
        """
        Update the internal state or retrieve new data from the interface.
        """
        return

    # @abc.abstractmethod
    def reset_detection_modules(self):
        """
        Reset or re-initialize the detection-related modules.

        This could be used to reload detection models, clear detection buffers,
        or re-sync with a remote detection service.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    def reset_segmentation_modules(self):
        """
        Reset or re-initialize the segmentation-related modules.

        Similar to reset_detection_modules, but specifically for segmentation.
        """
        raise NotImplementedError

    def _trigger_reset(self):
        with self.lock:
            self.updates["reset"] = True
    