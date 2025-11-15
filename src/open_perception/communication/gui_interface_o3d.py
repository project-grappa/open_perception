from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import CancelledError, TimeoutError
import json
import os
import pickle
import time
import cv2
import threading
import numpy as np
from typing import List, Union, Any, List, Dict, Tuple
from open_perception.pipeline.element import Element

from open_perception.communication.base_interface import BaseInterface
from open_perception.utils.visualization import draw_elements_detections, draw_segmentations, get_color, merge_segmentations
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
from scipy.spatial.transform import Rotation as R
import open3d
from open_perception.logging.base_logger import Logger
from open_perception.utils.common import combine_images
from datetime import datetime

from open_perception.communication.gui import Viewer3D, Viewer2D
from open_perception.communication.camera.utils import depth_to_pointcloud

import open3d.visualization.gui as gui
from easydict import EasyDict

# from contextlib import contextmanager
import cProfile, pstats

class CameraPipeline:
    """Class to handle the camera input from different sources:
    Webcam, Realsense, Video, Image, Redis"""

    def __init__(self, update_view_func,
                 device=None, logger=None, config={}):
        """
        Args:
            update_view_func (callback): Callback to update display elements for a frame.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
            config (dict): Configuration dictionary.      
        """
        self.config = config
        self.logger = logger if logger else Logger.get_logger(type(self).__name__)

        self.camera_source = config.get("camera_source", "realsense")
        self.camera_sources = ["webcam", "realsense", "video", "image"]
        self.external_sources = ["redis", "api"]
        self.camera_sources += self.external_sources

        self.output_folder = config.get("output_folder", "output")

        self.cap: cv2.VideoCapture = None # for webcam
        self.realsense_cam = None # for realsense
        self.frame_index = 0


        self.update_view_func = update_view_func
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if open3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = open3d.core.Device(self.device)


        
        
        # pointclous config. TODO parse correctly
        self.depth_max = 3.0  # m
        self.pcd_stride = 1  # downsample point cloud, may increase frame rate
        self.status_message = "Camera not initialized."

        
        # FLAGS
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False
        self.flag_capture = False
        self.recording = False  # Are we currently recording
        self.flag_record = False  # Request to start/stop recording
        self.flag_exit = False
        self.cv_capture = threading.Condition()  # condition variable
        
        self.rgbd_frame = None
        self._start_camera()

        # load camera matrices
        self.extrinsics = self.frame_metadata.extrinsics
        
        self.intrinsics = self.frame_metadata.intrinsics.intrinsic_matrix

        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Capture-Save')
        self.lock = threading.Lock()

    @property
    def max_points(self):
        """Max points in one frame for the camera or RGBD video resolution."""
        return int(self.frame_metadata.width * self.frame_metadata.height)

    @property
    def vfov(self):
        """Camera or RGBD video vertical field of view."""
        return np.rad2deg(2 * np.arctan(self.intrinsics[1, 2].item() /
                                        self.intrinsics[1, 1].item()))
    

    def _setup_realsense(self, camera_config_file=None):
        """Configure the RealSense pipeline and start streaming."""

        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{now}.bag"
        self.realsense_cam = open3d.t.io.RealSenseSensor()
        try:
            if camera_config_file:
                with open(camera_config_file) as ccf:
                    self.realsense_cam.init_sensor(open3d.t.io.RealSenseSensorConfig(json.load(ccf)), filename=filename)
            else:
                self.realsense_cam.init_sensor(filename=filename)
            self.realsense_cam.start_capture(start_record=False)
            self.frame_metadata = self.realsense_cam.get_metadata()
            self.status_message = f"Camera {self.frame_metadata.serial_number} opened."
        except Exception as e:
            self.logger.error(f"[CameraPipeline] Error setting up RealSense sensor: {e}")
            self.status_message = "Error setting up RealSense sensor."
            self.realsense_cam = None
            self.frame_metadata = None

        # from open_perception.communication.camera.realsense import RealSenseConnector
        # try:
        #     self.realsense_cam = RealSenseConnector(
        #         resolution = (640, 480), 
        #         invert_xy= True,
        #         cam_serial = self.config.get('realsense_cam_serial', None) 
        #     )
        #     self.logger.info("[GUIInterface] RealSense camera setup complete.")
        # except Exception as e:
        #     self.logger.error(f"[GUIInterface] Error setting up RealSense camera: {e}")
        #     self.realsense_cam = None


    def get_metadata_from_config(self, config):
        if config is None:
            config = {}
        width = config.get("width", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = config.get("height", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fov = config.get("fov", 60)
        extrinsics = config.get("extrinsic_matrix", [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
        extrinsics =  open3d.core.Tensor(extrinsics)
        intrinsic_matrix = config.get("intrinsic_matrix", None)
        if intrinsic_matrix is None:
            fx = width / (2 * np.tan(np.deg2rad(fov) / 2))
            fy = height / (2 * np.tan(np.deg2rad(fov) / 2))
            cx = width / 2
            cy = height / 2
            intrinsic_matrix = [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ]
        intrinsic_matrix = open3d.core.Tensor(
                                    intrinsic_matrix,
                                    dtype=open3d.core.Dtype.Float32,
                                    device=self.o3d_device)

        metadata = EasyDict({
            "width": int(width),
            "height": int(height),
            "device_name": "CV2 Webcam",
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "intrinsics": {
                "intrinsic_matrix": intrinsic_matrix
            },
            "extrinsics": extrinsics,                                                 
            "serial_number": "",
            "depth_scale": 1.0
        })
        return metadata

    def set_camera_source(self, source):
        
        last_capture_flag = self.flag_capture == True
        self.flag_capture = False
        with self.cv_capture:
            self.cv_capture.notify_all()

        print(self.future_frame.cancel())
        self._stop_camera()
        self.camera_source = source
        # self.executor.shutdown(wait=True, cancel_futures=False)
        self.executor = ThreadPoolExecutor(max_workers=3,thread_name_prefix='Capture-Save')
        self._start_camera()

        self.flag_capture = last_capture_flag
        print(self.flag_capture)

        with self.cv_capture:
            self.cv_capture.notify_all()

    def _start_camera(self):
        """
        Start the camera source based on the configuration.
        """
        self.realsense_ready = False
        if self.camera_source == 'realsense':
            if self.realsense_cam is None:
                self._setup_realsense()
            self.realsense_ready = True
        if self.camera_source == 'webcam':
            self.cap = cv2.VideoCapture(-1)
            # set metadata
            self.frame_metadata = self.get_metadata_from_config(self.config)
        elif self.camera_source == 'redis':
            pass # TODO handle dealt camera if necessary
        elif self.camera_source == 'redis':
            pass

    def _stop_camera(self):
        if self.realsense_ready and self.realsense_cam is not None:
            self.realsense_ready = False
            self.realsense_cam.stop_capture()
            # self.realsense_cam.shutdown()
            self.realsense_cam = None
        else:
            self.cap.release()

    def capture_frame(self, wait=True, align_depth_to_color=True):
        frame = None
        pc = None
        if self.camera_source == "realsense" and self.realsense_ready and self.realsense_cam is not None:
            # check if camera started

            frame = self.realsense_cam.capture_frame(wait=wait, align_depth_to_color=align_depth_to_color)
            
            # frames, color, depth  = self.realsense_cam.get_frames()
            # aligned_frames, aligned_color, aligned_depth = self.realsense_cam.get_aligned_frames(frames)
            # if aligned_color is None:
            #     return None
            
            # aligned_color = np.asanyarray(aligned_color.get_data())
            # frame = aligned_color
            # _, pc = self.realsense_cam.get_depth_pc(aligned_depth)
            # # reshape the point cloud
            # H, W = aligned_color.shape[:2]
            # pc = pc.reshape(H, W, 3)

        elif self.camera_source == 'webcam':
            # while True:
            ret, _frame = self.cap.read()
            if ret:
                frame = _frame
            #     break
            # if not wait:
            #     break

        elif self.camera_source == 'video':
            # loop video
            # while True:
            ret, _frame = self.cap.read()     
            if ret:
                frame = _frame
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
        elif self.camera_source == 'image':
            if self.frame_index == 0:
                ret, frame = self.cap.read()       
            else:
                frame = self.frame

        elif self.camera_source == 'redis':
            frame = self.frame_new # updated externally in the update method
            pc = self.point_cloud_new

        # with self.lock:
        # self.point_cloud = pc
        # self.frame = frame
        # self.frame_index +=1
        return (frame, pc)

    def set_new_frame(self, frame):
        """Set a new frame from an external source."""
        with self.lock:
            self.frame_new = frame


    def process_frame(self, frame: Union[open3d.t.geometry.Image, open3d.t.geometry.RGBDImage],
                      point_cloud: Union[open3d.t.geometry.PointCloud, np.ndarray]) -> tuple:
        error = 0
        colored_depth_frame = None
        pcd_o3d = None

        using_fake_depth = False
        invalid_depth_mask = None
        # parse frame into open3d RGBDImage or RGBImage
        if isinstance(frame, np.ndarray):
            if frame.shape[-1] == 3: # RGB image, fake depth
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                rgb =  open3d.t.geometry.Image(frame)
                fake_depth = np.ones(frame.shape[:2], dtype=np.float32) #+ \
                    # np.random.rand(*frame.shape[:2]).astype(np.float32)*0.001 #noies
                depth = open3d.t.geometry.Image(fake_depth)
                
            elif frame.ndim == 4: # RGBD image
                rgb =  open3d.t.geometry.Image(frame[:,:,:3])
                depth = open3d.t.geometry.Image(frame[:, :, -1])
            else: 
                raise ValueError(f"Unsupported frame shape: {frame.shape}, expected: RGB (H, W, 3) or  RGBD (H, W, 4)")
            frame = open3d.t.geometry.RGBDImage(rgb, depth)
            using_fake_depth = True

        elif isinstance(frame, open3d.t.geometry.Image): # RGB image, fake depth
            rgb = frame
            fake_depth = np.ones(frame.shape[:2], dtype=np.float32) #+ \
                # np.random.rand(*frame.shape[:2]).astype(np.float32) *0.001 # noise
            depth = open3d.t.geometry.Image(fake_depth)
            frame = open3d.t.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
            using_fake_depth = True
        elif isinstance(frame, open3d.t.geometry.RGBDImage):
            # ensure rgbd has all values greater than zeros 
            pass
            # depth_frame = frame.depth.as_tensor()
            # invalid_depth_mask = depth_frame == 0
            # depth_frame[invalid_depth_mask] = -1 # used to ensure open3d doesn't exclude these points
            # frame.depth = open3d.t.geometry.Image(depth_frame)
        elif frame is None:
            return None, None, None, None, 1
        else:
            raise ValueError(f"Unsupported frame type: {type(frame)}, expected: RGBImage, RGBDImage or numpy.ndarray") 
        # depth_frame = frame.depth.as_tensor()
        # color_frame = frame.color.as_tensor()
        # invalid_depth_mask = depth_frame == 0
        # depth_frame[invalid_depth_mask] = -1 # used to ensure open3d doesn't exclude these points
        
        # frame.depth = open3d.t.geometry.Image(depth_frame)
        # frame.color = open3d.t.geometry.Image(color_frame)
        # if point_cloud is None:
        #     # use intrinsics and extrinsics to compute point cloud
        #     point_cloud = depth_to_pointcloud(frame.depth.as_tensor().cpu().numpy()[:,:,0]/self.frame_metadata.depth_scale, 
        #                                       self.intrinsics.cpu().numpy(),
        #                                       self.extrinsics.cpu().numpy())
            
        # frame = frame.to(self.o3d_device)
        if point_cloud is None:
            # use intrinsics and extrinsics to compute point cloud

            # compute pointcloud from frame
            try:
                pcd_o3d = open3d.t.geometry.PointCloud.create_from_rgbd_image(
                        frame.cpu(), self.frame_metadata.intrinsics.intrinsic_matrix, self.extrinsics,
                        self.frame_metadata.depth_scale, self.depth_max,
                        self.pcd_stride, self.flag_normals)

            except RuntimeError:
                error = 1

            # pcd_o3d.point.positions = pcd_o3d.point.positions.contiguous()
            # point_cloud = np.zeros((color_frame.shape[0] * color_frame.shape[1], 3), dtype=np.float32)  
            # pc_np =  pcd_o3d.point.positions.cpu().numpy()

            # point_cloud[valid.cpu().numpy().ravel()] = pc_np
            # point_cloud = point_cloud.reshape(color_frame.shape[0], color_frame.shape[1], 3)
            # print(f"point cloud shape: {point_cloud.shape}")

        elif isinstance(point_cloud, np.ndarray):
            # convert numpy array to open3d point cloud
            pcd_o3d = open3d.t.geometry.PointCloud({
                'positions':
                    open3d.core.Tensor(point_cloud.reshape(-1, 3).astype(np.float32), 
                                       dtype=open3d.core.Dtype.Float32),
                'colors':
                    open3d.core.Tensor(frame.color.as_tensor().cpu().numpy().reshape(-1, 3).astype(np.float32)/255.0,
                                       dtype=open3d.core.Dtype.Float32),
                'normals':
                    open3d.core.Tensor(np.zeros_like(point_cloud).reshape(-1, 3).astype(np.float32),
                                       dtype=open3d.core.Dtype.Float32)
            })
        
        elif isinstance(point_cloud, open3d.t.geometry.PointCloud):
            # convert open3d point cloud to open3d point cloud and color it
            pcd_o3d = point_cloud.to(self.o3d_device)
            pcd_o3d.point.colors = open3d.core.Tensor(frame.color.as_tensor().cpu().numpy().reshape(-1, 3).astype(np.float32)/255.0,
                                       dtype=open3d.core.Dtype.Float32)
        colored_depth_frame = frame.depth.colorize_depth(
            self.frame_metadata.depth_scale, 0, self.depth_max)
        
        # if invalid_depth_mask is not None:
        # #     # set invalid depth to 0
        #     pcd_o3d.point.colors[invalid_depth_mask[:,:,0].T().flatten()] = [0,0,0]
        #     # pcd_o3d.point.colors = open3d.core.Tensor(frame_colors, device=self.o3d_device)


        return frame, point_cloud, pcd_o3d, colored_depth_frame, error
    
    def get_viewer_payload(self, *args, **kwargs):
        frame, point_cloud = self.capture_frame(wait=True, align_depth_to_color=True)
        frame, point_cloud, pcd_o3d, colored_depth_frame, error = self.process_frame(frame, point_cloud)

        viewer_payload = {
                'rgb_frame': frame.color if frame is not None else None,
                'depth_frame': frame.depth if frame is not None else None,
                'colored_depth_frame': colored_depth_frame,
                'point_cloud': point_cloud,
                'pcd_o3d': pcd_o3d,
                'status_message': self.status_message,
                'error': error,
                'frame_metadata': self.frame_metadata
            }
        return viewer_payload

    def run(self):
        """Run pipeline."""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        
        self.future_frame = self.executor.submit(self.get_viewer_payload, wait=True, align_depth_to_color=True)

        pcd_errors = 0
        while (not self.flag_exit):
            
            try:
                viewer_payload = self.future_frame.result(timeout=0.5)
            except (TimeoutError, CancelledError):
                # If the future is not done, continue to the next iteration
                self.future_frame = self.executor.submit(self.get_viewer_payload, wait=True, align_depth_to_color=True)
                continue
            self.future_frame = self.executor.submit(self.get_viewer_payload, wait=True, align_depth_to_color=True)
            
            if viewer_payload.get("rgb_frame") is None:
                self.logger.warning(f"Frame {frame_id} is None, skipping...")
                continue
            
            viewer_payload.update(frame_index=frame_id)
            # preprocess frame to extract point cloud if necessary
            # frame, point_cloud, pcd_o3d, colored_depth_frame, error = self.process_frame(frame, point_cloud)
            # pcd_errors+= error

            # if pcd_o3d.is_empty():
            #     self.logger.warning(f"No valid depth data in frame {frame_id})")
            #     continue

            # n_pts += pcd_o3d.point.positions.shape[0]
            # if frame_id % 60 == 0 and frame_id > 0:
            #     t0, t1 = t1, time.perf_counter()
            #     self.logger.debug(f"\nframe_id = {frame_id}, \t {(t1-t0)*1000./60:0.2f}"
            #               f"ms/frame \t {(t1-t0)*1e9/n_pts} ms/Mp\t")
            #     n_pts = 0

            
            # update the GUI with the new frame
            # viewer_payload = {
            #     'rgb_frame': frame.color,
            #     'depth': colored_depth_frame,
            #     'point_cloud': point_cloud,
            #     'pcd_o3d': pcd_o3d,
            #     'status_message': self.status_message,
            #     "frame_index": frame_id,
            # }
            # profiler = cProfile.Profile()
            # profiler.enable()
            self.update_view_func(viewer_payload)

            # profiler.disable()
            # pstats.Stats(profiler).sort_stats(pstats.SortKey.CUMULATIVE).print_stats(100)

            #handle flags
            if self.flag_save_rgbd:
                self.save_rgbd()
                self.flag_save_rgbd = False
            if self.flag_save_pcd:
                self.save_pcd()
                self.flag_save_pcd = False

            with self.cv_capture:  # Wait for capture to be enabled
                self.cv_capture.wait_for(
                    predicate=lambda: self.flag_capture or self.flag_exit)
            self.toggle_record()
            frame_id += 1

        self._stop_camera()
        self.executor.shutdown(cancel_futures=True)


    def _create_output_folder(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
    def save_pcd(self, pcd_o3d):
        """Save current point cloud."""
        self._create_output_folder()
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(self.output_folder, f"{self.frame_metadata.serial_number}_pcd_{now}.ply")

        # Convert colors to uint8 for compatibility
        pcd_o3d.point.colors = (pcd_o3d.point.colors * 255).to(
            open3d.core.Dtype.UInt8)
        self.executor.submit(open3d.t.io.write_point_cloud,
                             filename,
                             pcd_o3d,
                             write_ascii=False,
                             compressed=True,
                             print_progress=False)
        self.status_message = f"Saving point cloud to {filename}."

    def save_rgbd(self):
        """Save current RGBD image pair."""
        self._create_output_folder()
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = os.path.join(self.output_folder, f"{self.frame_metadata.serial_number}_color_{now}.jpg")
        self.executor.submit(open3d.t.io.write_image, filename,
                             self.rgbd_frame.color)
        filename = os.path.join(self.output_folder, f"{self.frame_metadata.serial_number}_depth_{now}.png")
        self.executor.submit(open3d.t.io.write_image, filename,
                             self.rgbd_frame.depth)
        self.status_message = (
            f"Saving RGBD images to {filename[:-3]}.{{jpg,png}}.")
        
    
    def _start_recording(self):
        """
        Start recording RGB frames and point clouds.
        """

        if self.camera_source == "realsense":
            if self.realsense_cam is not None:
                self.realsense_cam.resume_record()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.rgb_writer = cv2.VideoWriter(
                os.path.join(self.recording_dir, f"rgb_{timestamp}.avi"),
                cv2.VideoWriter_fourcc(*"XVID"),
                                30,  # Assuming 30 FPS
                                (640, 480)  # Assuming frame size
            )
            self.pc_writer = open(os.path.join(self.recording_dir, f"pointcloud_{timestamp}.pkl"), "wb")
        self.logger.info(f"[GUIInterface] Recording started. Files saved to {self.recording_dir}")

    def _stop_recording(self):
        """
        Stop recording and release resources.
        """
        if self.camera_source == "realsense":
            if  self.realsense_cam is not None:
                self.realsense_cam.stop_record()
        else:
            if self.rgb_writer:
                self.rgb_writer.release()
                self.rgb_writer = None
            if self.pc_writer:
                self.pc_writer.close()
                self.pc_writer = None
        self.logger.info("[GUIInterface] Recording stopped.")

    def toggle_record(self):
        if self.flag_record and not self.recording:
            self._start_recording()
            self.recording = True
        elif not self.flag_record and self.recording:
            self._stop_recording()
            self.recording = False


def pc_to_xyz_image(pc: open3d.core.Tensor,
                    depth: open3d.core.Tensor,
                    depth_scale: float = 1000.0,
                    depth_max: float  = np.inf) -> np.ndarray:
    """
    pc      : (N,3) tensor from create_from_rgbd_image(...)
    depth   : original depth image (H,W)
    returns : (H,W,3) ndarray, NaN where depth was invalid
    """
    h, w = depth.shape[:2]
    # valid pixels are the ones that produced a 3‑D point
    valid = (depth > 0)# & (depth < depth_max * depth_scale)   # (H,W) bool
    xyz_flat = np.full((h * w, 3), np.nan, dtype=np.float32)  # init with NaNs
    xyz_flat[valid.cpu().numpy().ravel()] = pc.cpu().numpy()  # scatter back
    return xyz_flat.reshape(h, w, 3)


class GUIInterface(BaseInterface):
    def __init__(self,config={}, models_config=None):
        super().__init__(config=config)
        self.logger = Logger.get_logger(type(self).__name__, config.get("logging", None))

        self.max_height = config.get("max_height", 600)
        self.camera_source = config.get('camera_source','webcam')
        self.output_dir = config.get('output_dir', os.path.join(os.path.dirname(__file__), '../../../output'))
        self.queries_list = []
        self.running = True
        self.new_info = True

        self.lock = threading.Lock()

        self.gui_camera_sources = ["webcam", "realsense", "video", "image"]
        self.external_sources = ["redis", "api"]
        self.camera_sources = self.gui_camera_sources + self.external_sources

        self._parse_models_config(models_config)

        self.locating_new_obj=False
        
        # Initialize RealSense pipeline if available
        self.realsense_cam = None

        self.default_image=np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(self.default_image, "No camera feed", (240, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # self.frame = None
        # self.point_cloud = None
        self.frame_new = None
        self.point_cloud_new = None

        # self.tracked_objects = []
        self.frame_index = 0
        self.frame_metadata = None
        self.depth_frame = None

        # Open3D visualization
        config_3d_viewer = config.get("3d_viewer", {})
        self.show_3d_viewer = config_3d_viewer.get("enabled", True)
        pc_limits_dict = config_3d_viewer.get("limits", {})
        self.pc_limits = np.array([
            [pc_limits_dict.get("x_min", -np.inf), pc_limits_dict.get("y_min", -np.inf), pc_limits_dict.get("z_min", -np.inf)],
            [pc_limits_dict.get("x_max", np.inf), pc_limits_dict.get("y_max", np.inf), pc_limits_dict.get("z_max", np.inf)]
        ])

        self.pc_res = config_3d_viewer.get("pointcloud_resolution", 1)
        self.update_every_n_frames = config_3d_viewer.get("update_every_n_frames", 10)
        self.show_origin = config_3d_viewer.get("show_origin", True)

        

        self.recording = False
        self.recording_dir = config.get("recording_dir", os.path.join(self.output_dir, "recordings"))
        os.makedirs(self.recording_dir, exist_ok=True)
        self.rgb_writer = None
        self.pc_writer = None



        self.camera_pipeline = CameraPipeline(self.update_viewer, config=self.config) 
        # self.viewer = Viewer2D(
        #     self.camera_sources, self.thresholds,
        #     close=self.close,
        #     switch_to_next_camera=self.switch_to_next_camera,
        #     set_update=self.set_update,
        #     add_query=self.add_query,
        #     start_recording=self.camera_pipeline._start_recording,
        #     stop_recording=self.camera_pipeline._stop_recording,
        #     save_elements=self.save_elements,
        #     trigger_reset=self._trigger_reset,
        #     set_camera_source=self.set_camera_source,
        # )

        self.vfov = self.camera_pipeline.vfov
        self.max_points = self.camera_pipeline.max_points
        self.viewer = Viewer3D(
            1.25 * self.vfov,
            self.max_points,
            on_window_close=self.on_window_close,
            on_toggle_capture=self.on_toggle_capture,
            on_save_pcd=self.on_save_pcd,
            on_save_rgbd=self.on_save_rgbd,
            on_toggle_record=self.on_toggle_record,
            on_toggle_normals=self.on_toggle_normals,
            on_camera_source_changed=self.on_camera_source_changed,
            on_add_query=self.on_add_query
        )

        self.running = False


    # =================== callback methods ===================

    def on_toggle_capture(self, is_enabled):
        """Callback to toggle capture."""
        self.camera_pipeline.flag_capture = is_enabled
        if not is_enabled:
            self.on_toggle_record(False)
            if self.viewer.toggle_record is not None:
                self.viewer.toggle_record.is_on = False
        else:
            with self.camera_pipeline.cv_capture:
                self.camera_pipeline.cv_capture.notify()

    def on_toggle_record(self, is_enabled):
        """Callback to toggle recording RGBD video."""
        self.camera_pipeline.flag_record = is_enabled

    def on_toggle_normals(self, is_enabled):
        """Callback to toggle display of normals"""
        self.camera_pipeline.flag_normals = is_enabled
        self.viewer.flag_normals = is_enabled
        self.viewer.flag_gui_init = False

    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.camera_pipeline.flag_exit = True
        with self.camera_pipeline.cv_capture:
            self.camera_pipeline.cv_capture.notify_all()
        return True  # OK to close window

    def on_save_pcd(self):
        """Callback to save current point cloud."""
        self.camera_pipeline.flag_save_pcd = True

    def on_save_rgbd(self):
        """Callback to save current RGBD image pair."""
        self.camera_pipeline.flag_save_rgbd = True

    def on_camera_source_changed(self, source, source_idx):
        """Callback to handle camera source selection."""
        # self.camera_source = self.camera_sources[source]
        self.set_camera_source(source)
        # self.camera_source = self.camera_sources[(self.camera_sources.index(self.camera_source)+1) % len(self.camera_sources)]
        self.logger.info(f"[GUIInterface] Switching camera to {self.camera_source}.")



    def on_add_query(self):
        """Callback to handle adding a new query."""
        query = self.viewer.text_input.text_value
        if query:
            self.queries_list.append(query)
            self.viewer.update_queries_list(self.queries_list)
            self.viewer.text_input.text_value = ""  # Clear the input box after adding the query
            self.logger.info(f"Added query: {query}")
            with self.lock:
                self.objects_to_add.append({
                    "class_name": query,
                    "compute_pose": True,
                    "compute_mask": True,
                    "track": True
                })
            print(self.objects_to_add)

    # =========================================================

    def _parse_models_config(self, model_config):
        self.thresholds = {}
        if not model_config:
            return
        
        all_models = model_config.get("detection", []) + model_config.get("segmentation", [])
        for m in all_models:
            model_name = m.get("name", "")
            if model_name:
                self.thresholds[model_name] = {}
                # add any keys with "threshold" in the name
                if isinstance(m, dict):
                    for k, v in m.items():
                        if "threshold" in k and isinstance(v, (int, float)):
                            self.thresholds[model_name][k] = float(v)


    def set_camera_source(self, source):
        # self.camera_pipeline._stop_camera()
        self.camera_source = source
        self.camera_pipeline.set_camera_source(source)
        
        input_source = source if source in self.external_sources else "gui"

        with self.lock:
            self.updates["config"]=self.updates.get("config",{})
            self.updates["config"]["pipeline"] ={"input": {"source": input_source}}
        # self.camera_pipeline._start_camera()

    def switch_to_next_camera(self):
        # if self.realsense_available:
        # self.camera_source = 'realsense' if self.camera_source == 'webcam' else 'webcam'
        self.camera_source = self.camera_sources[(self.camera_sources.index(self.camera_source)+1) % len(self.camera_sources)]
        self.logger.info(f"[GUIInterface] Switching camera to {self.camera_source}.")

        self.set_camera_source(self.camera_source)

    
    def update_viewer(self, viewer_payload):
        viewer_payload.update({
            "frame_index": viewer_payload.get("frame_index", self.frame_index),
            "camera_source": self.camera_source,
            "camera_sources": self.camera_sources,
            "detections": self.detections,
        })
        with self.lock:
            # print(self.frame_index)
            rgb = viewer_payload.get("rgb_frame")
            self.frame = np.asarray(rgb.cpu(), dtype=np.uint8) if rgb is not None else None

            pc = viewer_payload.get("pcd_o3d")
            pc_np = viewer_payload.get("point_cloud")
            if pc_np is not None and rgb is not None:
                if np.prod(pc_np.shape) == np.prod(self.frame.shape):
                    h, w, _ = self.frame.shape
                    self.point_cloud = pc_np.reshape((h, w, 3), order="C")
                else:
                    # print(np.prod(pc_np.shape), np.prod(self.frame.shape))
                    self.point_cloud = None
            else:
                self.point_cloud = None

            self.depth_frame = viewer_payload.get("depth_frame")
            self.frame_metadata = viewer_payload.get("frame_metadata")
            self.frame_index = viewer_payload.get("frame_index", self.frame_index)

        # self.viewer.update(viewer_payload)
        gui.Application.instance.post_to_main_thread(
            self.viewer.window,
            lambda: self.viewer.update(viewer_payload))

    def run(self):

        self.running = True
        
        self.logger.info("[GUIInterface] Starting GUI interface.")
        threading.Thread(name='CameraPipeline', target=self.camera_pipeline.run).start()

        # enable camera
        self.camera_pipeline.flag_capture = True
        with self.camera_pipeline.cv_capture:
            self.camera_pipeline.cv_capture.notify_all()
        
        self.viewer.run()
        # gui.Application.instance.run()
        # while self.running:
        #     time.sleep(1)
        #     print("running")
        self.close()

    
    def set_update(self, **updates):
        """
        Set the update for the GUI interface.
        """
        with self.lock:
            self.updates.update(updates)
            self.logger.info(f"[GUIInterface] Updates: {updates}")

    def add_query(self, query):
        """
        Add a query to the list of queries to be processed.
        """
        with self.lock:
            self.queries_list.append(query)
            self.locating_new_obj = True
            self.logger.info(f"[GUIInterface] Adding new query: {query}")
            self.objects_to_add.append({
                    "class_name": query,
                    "compute_pose": True,
                    "compute_mask": True,
                    "track": True
                })
    
    def start(self):
        """
        Run the Tkinter mainloop in a background thread.
        """
        # self.gui_thread = threading.Thread(target=self.run, daemon=True)
        # self.gui_thread.start()
        pass


    def save_elements(self):
        raise NotImplementedError("Saving elements is not implemented yet.")

    # ====================================== main interface methods ======================================
    def get_synced_frame_and_pc(self):
        """
        Retrieve an image/frame and point cloud from synchronized sensors.

        :return: A dict of tuple of {sensor_name: (frame, point_cloud) where:
                 - frame is a NumPy array (H x W x C) in BGR, RGB, or grayscale.
                 - point_cloud is a structured point cloud (e.g., Nx3 NumPy array).
                 Return None if no frame or point cloud is available or on error.
        """
        with self.lock:
            frame = self.frame
            pc = self.point_cloud
            index = self.frame_index
            frame_metadata = self.frame_metadata
            depth_frame = self.depth_frame
        
        return {"front": {"rgb":frame, "point_cloud": pc, "index": index, "frame_metadata": frame_metadata, "depth_frame": depth_frame}}
    
    def set_frame(self, frame):
        with self.lock:
            self.frame = frame


    def publish_results(self, detections: list[Element], id: int = 0, sensor_name:str = None):
        """
        Publish detection results to a remote service or communication channel. 

        :param detections: List of Element objects representing detected objects.
        :param id: Optional ID of the detection result (e.g., frame index).
        :param sensor_name: Optional name of the sensor that produced the detections.
        """
        with self.lock:
            self.locating_new_obj=False
            if detections != self.detections:
                self.new_info = True

            self.detections = detections
            self.queries_list = [d.class_name for d in detections]

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
        Release resources or shut down connections.
        """
        pass

    def update(self, updates=None):
        """
        Update the internal state or retrieve new data from the interface.
        """
        if updates:
            self.logger.info("[GUIInterface] Received update.")
            if isinstance(updates, dict):
                if "frames" in updates.keys() and len(updates["frames"].keys())>0:
                    
                    camera = list(updates["frames"].keys())[0]
                    self.logger.info(f"[GUIInterface] Received frame from camera {camera}")
                    with self.lock:
                        camera_source=updates.get("source", self.camera_source)
                        if camera_source != "gui":
                            frame_new = updates["frames"][camera].get("rgb")
                            self.frame_new = frame_new.copy() if frame_new is not None else None
                            self.camera_source = camera_source
                            point_cloud_new = updates["frames"][camera].get("point_cloud", None)
                            self.point_cloud_new = point_cloud_new.copy() if point_cloud_new is not None else None
                            # self.new_info = True

    def get_updates(self):
        
        # self._update_3d_viewer()
        return super().get_updates()

   

    def _record_frame_and_pc(self, frame, point_cloud):
        """
        Record the current frame and point cloud if recording is active.
        """
        if self.recording and frame is not None:
            self.rgb_writer.write(frame)
            if point_cloud is not None:
                pickle.dump(point_cloud, self.pc_writer, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    gui_interface = GUIInterface(config= {"camera_source": "webcam"})
    gui_interface.start()
    gui_interface.run()
    # while gui.running:
    #     pass
