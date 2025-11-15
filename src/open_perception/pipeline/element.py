from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import json

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
@dataclass
class Element:
    """
    Represents a single object/entity detected and tracked in the environment.

    Attributes:
        element_id (int): Unique identifier for this element (e.g., track ID).
        class_name (str): Label/class of the object (e.g., "car", "person").
        detection_prob (float): Detection confidence score from the detector.
        segmentation_prob (float): Segmentation confidence score (if applicable).
        bbox (Tuple[int, int, int, int]): Bounding box in pixel coordinates
            [x1, y1, x2, y2].
        segmentation_mask (Optional[np.ndarray]): The segmentation mask array.
            Could be a binary mask (H x W) or multi-channel mask.
        pose (Optional[list[float]]): Position in 3D space (e.g., [x, y, z])
            or a more complex structure.
        orientation (Optional[list[float]]): Orientation (e.g., quaternion [x, y, z, w])
            or Euler angles. Adjust to your needs.
        parent_element (Optional["Element"]): If this object is
            logically grouped or nested under another element (e.g., part-of relationships).
        meta (Dict[str, Any]): Additional metadata (timestamps, sensors, custom data).
        detection_frame (Optional[np.ndarray]): The image/frame associated with this detection.
            Could be stored for debugging or logging.
        detection_frame_idx (int): Numeric ID/index for the frame from which this was detected.
        color (Optional[Any]): Color information for visualization (e.g., RGB tuple).

    """

    element_id: int

    class_name: str = "unknown"
    detection_prob: float = 0.0
    segmentation_prob: float = 0.0

    # Typically [x1, y1, x2, y2]
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    segmentation_mask: Optional[np.ndarray] = None



    compute_pose: Optional[bool] = True
    compute_mask: Optional[bool] = True 
    track: Optional[bool] = True

    # -------- Multi granular search information -------
    parent_element: Optional["Element"] = None
    query: str = "" # query used to locate this element
    is_parent: bool = False

    # ----------------- 3D information -----------------
    points: Optional[np.ndarray] = None # points that belong to the object
    points_indices: Optional[np.ndarray] = None # indexes of the points that belong to the object

    pose: Optional[np.ndarray] = None #
    euler: Optional[np.ndarray] = None # Euler angles
    quat: Optional[np.ndarray] = None # quaternion xyzw
    rotation_matrix: Optional[np.ndarray] = None # 3x3 rotation matrix
    size: Optional[np.ndarray] = None
    open3d_bbox = None

    # ----------------- Metadata -----------------

    description: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    detection_frame: Optional[np.ndarray] = None
    detection_frame_idx: int = 0
    color: Optional[Any] = None
    color_name: Optional[str] = ""  # e.g., "red", "blue"
    
    is_reused: bool = False

    lost_count: Optional[int] = 0

    def __str__(self):
        return f"Element {self.element_id}: {self.class_name} @ {self.bbox}"

    def to_dict(self) -> Dict:
        """Convert the element to a dictionary."""
        
        element_dict = self.__dict__.copy()
        # handle non-serializable types
        if self.open3d_bbox:
            element_dict["open3d_bbox"] = {
                "center": self.open3d_bbox.center.tolist(),
                "extent": self.open3d_bbox.extent.tolist(),
                "R": self.open3d_bbox.R.tolist(),
            }
        return element_dict

    def to_json(self) -> str:
        """Serialize the element to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict) -> "Element":
        """Create an Element instance from a dictionary."""

        obj = cls(**data)

        # Load Open3D bounding box if available
        if data.get("open3d_bbox") and OPEN3D_AVAILABLE:
            bbox_data = data["open3d_bbox"]
            obj.open3d_bbox = o3d.geometry.OrientedBoundingBox(
                center=bbox_data["center"],
                extent=bbox_data["extent"],
                R=bbox_data["R"]
            )
        return obj

    @classmethod
    def from_json(cls, json_str: str) -> "Element":
        """Create an Element instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)