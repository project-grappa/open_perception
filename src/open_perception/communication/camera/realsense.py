import os
import time
import numpy as np
import pyrealsense2 as rs

import open_perception.communication.camera.utils as U

class RealSenseConnector():
    """
    A class to manage the connection and data retrieval from an Intel RealSense camera.
    Attributes:
    -----------
    cam_serial : str
        The serial number of the RealSense camera.
    resolution : tuple
        The resolution of the camera (width, height).
    fps : int, optional
        Frames per second (default is 30).
    connection_timeout : int, optional
        Time to wait for the camera to reset (default is 4 seconds).
    align_to : rs.align, optional
        Stream to align to (default is rs.stream.depth).
    invert_xy : bool, optional
        Whether to invert the x and y coordinates (default is True).
    Methods:
    --------
    get_depth_color_image(depth_frame):
        Returns a colorized depth image from the depth frame.
    get_frames(as_numpy=False):
        Retrieves the color and depth frames from the camera.
    get_aligned_frames(frames, as_numpy=False):
        Retrieves the aligned color and depth frames.
    get_depth_pc(depth_frame, R=None, T=None):
        Returns the point cloud and vertices from the depth frame.
    get_cam_coord(coords, frames):
        Converts pixel coordinates to camera coordinates.
    shutdown():
        Stops the camera pipeline and shuts down the camera.
    """


    def __init__(self, 
                 resolution: tuple, 
                 fps: int = 30,
                 connection_timeout: int = 4, 
                 align_to: rs.align = rs.stream.color,
                 invert_xy: bool = True,
                 cam_serial: str = None):
        """
        Initializes the RealSenseConnector class with the provided parameters.
            Args:
                resolution (tuple): The resolution of the camera (width, height).
                fps (int, optional): The frames per second for the camera. Defaults to 30.
                connection_timeout (int, optional): The timeout duration for the camera connection in seconds. Defaults to 4.
                align_to (rs.align, optional): The stream to align to (rs.stream.color of rs.stream.depth). Defaults to rs.stream.color.
                invert_xy (bool, optional): Whether to invert the x and y coordinates. Defaults to True.
                cam_serial (str): The serial number of the camera.
        """
        
        # Initialize the RealSenseConnector class with the provided parameters
        # Reset connected devices for cable issues
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise Exception("No RealSense devices were found.")

        if cam_serial is None:
            self.cam_serial = devices[0].get_info(rs.camera_info.serial_number) 
            print(f"[INFO] Camera serial not provided. Using {self.cam_serial} as default.")
        else:
            self.cam_serial = cam_serial

        for dev in devices:
            dev.hardware_reset()
        # Wait to allow the device to reset
        time.sleep(connection_timeout)
        
        # Init camera parameters
        self.width, self.height = resolution

        # CONSTANT PARAMETERS (OBTAINED FROM REALSENSE-VIEWER)
        self.max_depth_clamp = 30000.0
        self.min_z = 0.3
        self.max_z = 2.7

        # Init pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.cam_serial)
        self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, fps)
        
        profile = self.pipeline.start(self.config)
        
        self.align = rs.align(align_to)
        self.align.as_hole_filling_filter()

        # Get calibration parameters
        self.intr = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
        self.camera_matrix = np.array([[self.intr.fx, 0, self.intr.ppx],
                                       [0, self.intr.fy, self.intr.ppy],
                                       [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.array(self.intr.coeffs)

        # Set the correct frame
        self.correct_frame = invert_xy
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()


    def get_depth_color_image(self, depth_frame):
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        return colorized_depth
    
    def get_frames(self, as_numpy: bool = False) -> tuple:
        frames = self.pipeline.wait_for_frames()
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()

        if as_numpy:
            color = np.asanyarray(color.get_data())
            depth = np.asanyarray(depth.get_data())
        
        return frames, color, depth
    
    def get_aligned_frames(self, frames, as_numpy: bool = False) -> tuple:
        aligned_frames = self.align.process(frames)
        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()

        if as_numpy:
            color = np.asanyarray(color.get_data())
            depth = np.asanyarray(depth.get_data())
        
        return aligned_frames, color, depth
    
    def get_depth_pc(self, depth_frame, R = None, T = None):
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  # xyz
        verts[:, [0, 1]] = verts[:, [1, 0]]
        verts[:, 2] *= -1
        
        if R is not None and T is not None:
            verts = U.apply_transform(verts, R,T)
        return points, verts
    
    def get_cam_coord(self, coords, frames):
        px, py = coords
        color_frame, depth_frame = frames
        
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth = depth_frame.get_distance(px, py)
        if self.correct_frame:
            real_y ,real_x, real_z = rs.rs2_deproject_pixel_to_point(color_intrin, [px,py], depth)
        else:
            real_x ,real_y, real_z = rs.rs2_deproject_pixel_to_point(color_intrin, [px,py], depth)
        
        return np.array([real_x, real_y, -real_z], dtype= np.float32)
    
    def shutdown(self):
        self.pipeline.stop()
        print(f"[INFO] Camera {self.cam_serial} has been shutdown.")
        
