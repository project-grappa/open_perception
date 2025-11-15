import cv2
import numpy as np

def rtvec_to_matrix(rvec, tvec, camera_matrix = None, include_intrinsics = True):
    def rtvec_to_matrix(rvec, tvec, camera_matrix=None, include_intrinsics=True):
        """
        Converts a rotation vector and a translation vector into a transformation matrix and its inverse.
        Args:
            rvec (np.ndarray): Rotation vector (3x1).
            tvec (np.ndarray): Translation vector (3x1).
            camera_matrix (np.ndarray, optional): Camera intrinsic matrix (3x3). Defaults to None.
            include_intrinsics (bool, optional): Whether to include camera intrinsics in the transformation. Defaults to True.
        Returns:
            tuple: A tuple containing:
                - T (np.ndarray): Transformation matrix from world coordinates to camera coordinates (4x4).
                - T_inv (np.ndarray): Inverse transformation matrix from camera coordinates to world coordinates (4x4).
        """
    
    # Credits: https://stackoverflow.com/questions/73340550/how-does-opencv-projectpoints-perform-transformations-before-projecting
    # breakpoint()
    # World2Cam
    R, _ = cv2.Rodrigues(rvec)
    T = np.hstack((R, tvec))
    if include_intrinsics:
        T = camera_matrix @ T

    T = np.vstack((T, np.array([[0, 0, 0, 1]])))
    # Cam2World
    R_inv = np.linalg.inv(R)
    tvec_inv = -R_inv @ tvec
    T_inv = np.hstack((R_inv, tvec_inv))
    if include_intrinsics:
        T_inv = np.linalg.inv(camera_matrix) @ T_inv
    T_inv = np.vstack((T_inv, np.array([[0, 0, 0, 1]])))

    return T, T_inv

def get_homogenous_points(points):
    """
    Converts a list or NumPy array of points into their homogeneous coordinates.
    Parameters
    ----------
    points : array-like of shape (n, m)
        The original points, where n is the number of points and m is the dimensionality.
    Returns
    -------
    numpy.ndarray
        The homogeneous coordinates (including a row of ones), transposed to facilitate
        further transformations.
    """

    points = np.array(points)
    points = np.hstack((points, np.ones((points.shape[0], 1))))
    points = points.T
    return points

def homogenous_to_cartesian(points):
    """
    Converts a set of points from homogeneous coordinates to Cartesian coordinates.
    Parameters:
    points (numpy.ndarray): A 2D array of shape (3, N) representing N points in homogeneous coordinates.
    Returns:
    numpy.ndarray: A 2D array of shape (N, 2) representing N points in Cartesian coordinates.
    """
    
    points = points[:2] / points[2]
    points = points.T
    # points = points[:, :-1] 
    return points

def center_point(P_org):
    """
    Calculate the center point of a set of points and return the centered points.
    Parameters:
    P_org (numpy.ndarray): A 2D array where each column represents a point in space.
    Returns:
    tuple: A tuple containing:
        - P_bar (numpy.ndarray): The mean point (center) of the original points.
        - P (numpy.ndarray): The centered points obtained by subtracting the mean point from the original points.
    """
    
    P_bar = np.mean(P_org, axis= 1)
    P = P_org - P_bar[:, np.newaxis]
    return P_bar, P

def get_transformation_params(P_original: np.array, Q_original: np.array):
    """
    Calculate the rotation matrix and translation vector that align two sets of points.
    This function computes the optimal rotation matrix (R) and translation vector (trans)
    that align the set of points P_original to the set of points Q_original using the
    Kabsch algorithm.
    Args:
        P_original (np.array): A numpy array of shape (n, m) representing the original set of points.
        Q_original (np.array): A numpy array of shape (n, m) representing the target set of points.
    Returns:
        tuple: A tuple containing:
            - R (np.array): The rotation matrix of shape (m, m).
            - trans (np.array): The translation vector of shape (m,).
    """

    # Get centroids and center points
    P_bar, P = center_point(P_original)
    Q_bar, Q = center_point(Q_original)

    C = P @ Q.T

    U, _ , Vh = np.linalg.svd(C)
    R = Vh.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vh[-1, : ] *= -1
        R = Vh.T @ U.T

    trans = Q_bar - (R @ P_bar)

    return R, trans

def apply_transform(P, R, T):
    Q = (P @ R.T) + T
    return Q



def depth_to_pointcloud(depth_image, K, E):
    """
    Converts a depth image to a 3D point cloud using camera intrinsics and extrinsics.

    Parameters:
    - depth_image: The depth image (a 2D numpy array of depth values in meters).
    - K: Camera intrinsic matrix (3x3).
    - E: Camera extrinsic matrix (4x4) combining rotation and translation.

    Returns:
    - pointcloud: A 3D point cloud in the world coordinate system (N x 3 array).
    """

    R = E[:3, :3]  # Rotation part of the extrinsic matrix
    t = E[:3, 3]   # Translation part of the extrinsic matrix

    # Get the height and width of the depth image
    height, width = depth_image.shape[:2]

    # Create a grid of pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    # Normalize pixel coordinates to camera coordinates
    x = (u - K[0, 2]) * depth_image / K[0, 0]  # fx and cx are in K[0, 0] and K[0, 2]
    y = (v - K[1, 2]) * depth_image / K[1, 1]  # fy and cy are in K[1, 1] and K[1, 2]
    z = depth_image

    # Stack the (x, y, z) coordinates
    points_camera = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Apply the extrinsics (rotation + translation) to convert to world coordinates
    points_world = np.dot(R, points_camera.T).T + t

    return points_world
