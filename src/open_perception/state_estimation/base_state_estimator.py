"""
state_estimator.py

Defines classes to compute the 3D position and bounding box of environment elements
given:
- A list of Elements (see your Element class),
- A camera frame (e.g., a 2D image),
- A point cloud.

Includes:
- BaseStateEstimator (abstract)
- PCAStateEstimator (example implementation of BaseStateEstimator)
"""

import open3d
import abc
from typing import List, Optional
import numpy as np

from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

from open_perception.pipeline.element import Element


class BaseStateEstimator(abc.ABC):
    """
    An abstract base class that defines how to estimate the 3D position and
    bounding box of Elements based on camera frames and point clouds.
    Subclasses must implement the estimate_state method.
    """

    def __init__(self, config: Optional[dict] = None):
        """
        :param config: Optional dictionary with configuration parameters for state estimation.
        """
        self.config = config if config else {}
        self.num_outlier_filtering = self.config.get('num_outlier_filtering', 3)

    @abc.abstractmethod
    def estimate_state(
        self,
        elements: list[Element],
        pointcloud: np.ndarray=None,
        depth_frame: Optional[np.ndarray]=None,
        frame_metadata: Optional[dict]=None,
        frame: np.ndarray=None,
    ) -> list[Element]:
        """
        Given a list of Element objects, a 2D frame, and a point cloud,
        compute or update the 3D position/orientation/bounding box in each element.

        :param elements: List of detected/tracked elements. The bounding boxes or segmentation
                         masks in each element can be used to isolate relevant points.
        :param frame: 2D camera frame (e.g., BGR or RGB image). This can help with color or
                      additional references, if needed.
        :param pointcloud: 3D point cloud (e.g., Nx3 or Nx6 if including color data).
        :return: The same list of elements, but with updated 3D information (pose, orientation,
                 or 3D bounding box).
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the state estimator, clearing any internal state or buffers.
        """
        pass

class PCAStateEstimator(BaseStateEstimator):
    """
    An example state estimator that uses PCA on the subset of points within each element’s
    bounding box or segmentation mask to determine orientation and possibly a 3D bounding box.
    """
    def __init__(self, config = None):
        super().__init__(config)

    def _filter_outliers(self, points: np.ndarray, k:int =3, points_indices = None) -> np.ndarray:
        """
        Filter out outlier points from the point cloud of the object.
        This is a placeholder method and should be implemented based on your specific needs.
        """
        # For example, you could remove points that are too far from the centroid
        # or have an unusual color value.
        if points_indices is None:
            points_indices = np.arange(points.shape[0])
        for i in range(self.num_outlier_filtering):
            pc_mean = np.average(points, axis=0) # Center the point cloud
            centered_point_cloud = points - pc_mean
            
            if points.shape[0] < 10:
                return points, points_indices
            # exclude ouliers further from 3 std
            mask = np.linalg.norm(centered_point_cloud, axis=1) < 3*np.std(centered_point_cloud)
            points = points[mask]
            points_indices = points_indices[mask]

        return points, points_indices
    
    def estimate_state(
        self,
        elements: list[Element],
        pointcloud: np.ndarray=None,
        depth_frame: Optional[np.ndarray]=None,
        frame_metadata: Optional[dict]=None,
        frame: np.ndarray=None,
    ) -> list[Element]:
        """
        Uses PCA (Principal Component Analysis) on the points corresponding to each element's
        bounding box or segmentation mask to compute:
        - 3D position: e.g., centroid of the point set
        - Orientation: e.g., principal axis from PCA
        - 3D bounding box: e.g., extents along principal axes

        Note: This method is left as a placeholder or minimal skeleton. Implement the actual
        PCA math, point extraction, coordinate transformations, etc. to suit your needs.

        :param elements: List of Element objects with 2D bounding boxes and segmentation
                         masks that can be used to filter the pointcloud.
        :param frame: 2D camera frame for reference (not strictly necessary for 3D estimation,
                      but could help if you need color or alignment checks).
        :param pointcloud: Nx3 or Nx6 array representing the 3D environment. Depending on your
                           sensor, Nx6 might include XYZ + RGB or other data.
        :return: List of Element with updated pose, orientation, and optional 3D bounding
                 box data.
        """
        if pointcloud is None and depth_frame is None:
            return elements

        for element in elements:
            

            # 1. Identify which points correspond to this element:
            object_pc, object_pc_indices = self._extract_points_for_element(element, pointcloud, depth_frame, frame_metadata)
            # object_pc, object_pc_indices = self._filter_outliers(object_pc, points_indices=object_pc_indices)
            if object_pc.shape[0] < 10: # check if there are enough points
                continue

            # 
            if self.config.get("fix_orientation_plane", "") != "":
                if self.config["fix_orientation_plane"] == "xy" or self.config["fix_orientation_plane"] == "yx":
                    plane_normal = np.array([0, 0, 1])
                elif self.config["fix_orientation_plane"] == "yz" or self.config["fix_orientation_plane"] == "zy":
                    plane_normal = np.array([1, 0, 0])
                elif self.config["fix_orientation_plane"] == "zx" or self.config["fix_orientation_plane"] == "xz":
                    plane_normal = np.array([0, 1, 0])
                else:
                    # get plane normal from string
                    plane_normal = np.array([int(x) for x in self.config["fix_orientation_plane"].split(",")])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                
                # project points to the plane
                dist_from_plane = np.dot(object_pc, plane_normal)
                projected_points_min = object_pc - (dist_from_plane[:, None]- np.min(dist_from_plane)) * plane_normal
                projected_points_max = object_pc - (dist_from_plane[:, None]- np.max(dist_from_plane)) * plane_normal

                # add points to the planes to fix the orientation
                bbox_pc = np.vstack([object_pc, projected_points_min, projected_points_max])
                open3d_bbox = open3d.geometry.OrientedBoundingBox().create_from_points(open3d.utility.Vector3dVector(bbox_pc), robust=True)
            else:
                # 2. Run PCA on pcd_subset to find principal axes, centroid
                open3d_bbox = open3d.geometry.OrientedBoundingBox().create_from_points(open3d.utility.Vector3dVector(object_pc), robust=True)
            
            # update the element with the new pose, orientation, and bounding box
            element.size = open3d_bbox.extent
            element.pose = open3d_bbox.center
            element.points = object_pc
            element.rotation_matrix = open3d_bbox.R
            element.open3d_bbox = open3d_bbox
            element.points_indices = object_pc_indices

            # assert len(element.points) == len(element.points_indices)

            print(f"Element {element.element_id}: pose={element.pose}, size={element.size}")

        return elements

    # If you want to implement helper functions, you could add them here:
    #

    def camera_parameters_from_metadata(self, frame_metadata):
        """
        Extract camera parameters from frame metadata.
        """
               
        camera_parameters = {}
        if frame_metadata:
            camera_parameters["intrinsics"] = frame_metadata.get("intrinsics", {}).get("intrinsic_matrix", None)
            camera_parameters["extrinsics"] = frame_metadata.get("extrinsics", None)
            camera_parameters["depth_scale"] = frame_metadata.get("depth_scale", None)

        
        return camera_parameters
    
    def _extract_points_for_element(self, element, pointcloud=None,  depth_frame=None, frame_metadata=None):
        """
        Given an Element and the full pointcloud, return only the points that fall
        within the segmentation mask.
        """
        
        object_mask = element.segmentation_mask
        if object_mask is None:
            return np.array([]), np.array([])
        
        object_mask = object_mask.astype(bool)
        binarized_mask = np.stack([object_mask[:,:,0]]*3, axis=-1)
        
        if pointcloud is not None:
            object_pc = pointcloud[binarized_mask]
            object_pc = object_pc.reshape(-1,3)
        
        elif depth_frame is not None and frame_metadata is not None:
            
            camera_parameters = self.camera_parameters_from_metadata(frame_metadata)
            
            # get points with open3d using only the valid points
            depth_frame = depth_frame.as_tensor()
            depth_frame[~binarized_mask] = 0 
            masked_depth_frame = open3d.t.geometry.Image(depth_frame)
            object_pc = open3d.t.geometry.PointCloud.create_from_depth_image(
                        masked_depth_frame, camera_parameters["intrinsics"], camera_parameters["extrinsics"],
                        camera_parameters["depth_scale"], np.inf, 1, False).point.positions.cpu().numpy()

        object_pc_indices = object_mask[:,:,0].flatten()
        object_pc_indices = np.where(object_pc_indices)[0]
        return object_pc, object_pc_indices


if __name__ == "__main__":
    # Example usage of the PCAStateEstimator
    # Create a dummy point cloud and Element objects
    pointcloud = np.random.rand(100, 3) * 10  # 100 random 3D points in [0, 10]
    pointcloud[:,2] = pointcloud[:,2] * 0.1  # Scale Z to be smaller
    pointcloud[:,1] = pointcloud[:, 1] * 0.5  # Scale Y to be smaller
    
    # rotate the pointcloud in 3D, by 30 degrees on the x axis
    r = R.from_euler('x', 30, degrees=True)
    pointcloud = r.apply(pointcloud)

    # shift the pointcloud in z
    pointcloud[:,2] += 5

    