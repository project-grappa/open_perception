import os
import pickle
import cv2
import threading
import numpy as np
from typing import List
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
from open_perception.communication.drag_drop_loader import DragDropImageLoader


class GUIInterface(BaseInterface):
    def __init__(self,config=None, models_config=None):
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
        self.external_sources = ["redis"]
        self.camera_sources = self.gui_camera_sources + self.external_sources

        self._parse_models_config(models_config)

        self.locating_new_obj=False
        
        # Initialize RealSense pipeline if available
        self.realsense_cam = None

        self.default_image=np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(self.default_image, "No camera feed", (240, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.frame = None
        self.point_cloud = None
        self.frame_new = None
        self.point_cloud_new = None

        # self.tracked_objects = []
        self.frame_index = 0
        self.user_input = ''

        # Open3D visualization
        config_3d_viewer = config.get("3d_viewer", {})
        self.show_3d_viewer = config_3d_viewer.get("enabled", True)
        pc_limits_dict = config_3d_viewer.get("limits", {})
        self.pc_limits = np.array([
            [pc_limits_dict.get("x_min", -np.inf), pc_limits_dict.get("y_min", -np.inf), pc_limits_dict.get("z_min", -np.inf)],
            [pc_limits_dict.get("x_max", np.inf), pc_limits_dict.get("y_max", np.inf), pc_limits_dict.get("z_max", np.inf)]
        ])

        self.pc_res = config_3d_viewer.get("pointcloud_resolution", 5)
        self.update_every_n_frames = config_3d_viewer.get("update_every_n_frames", 10)
        self.show_origin = config_3d_viewer.get("show_origin", True)

        self.first_point_cloud = True
        self.open3d_vis = None
        self.open3d_pcd = None
        self.open3d_bboxes = []
        self._start_camera()

        self.recording = False
        self.recording_dir = config.get("recording_dir", os.path.join(self.output_dir, "recordings"))
        os.makedirs(self.recording_dir, exist_ok=True)
        self.rgb_writer = None
        self.pc_writer = None

        # Initialize drag-and-drop loader
        self.drag_drop_loader = DragDropImageLoader(on_image_loaded=self._on_drag_drop_image_loaded)

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

    def _setup_realsense(self):
        """Configure the RealSense pipeline and start streaming."""

        from open_perception.communication.camera.realsense import RealSenseConnector
        try:
            self.realsense_cam = RealSenseConnector(
                resolution = (640, 480), 
                invert_xy= True,
                cam_serial = self.config.get('realsense_cam_serial', None) 
            )
            self.logger.info("[GUIInterface] RealSense camera setup complete.")
        except Exception as e:
            self.logger.error(f"[GUIInterface] Error setting up RealSense camera: {e}")
            self.realsense_cam = None

    def _start_camera(self):
        """
        Start the camera source based on the configuration.
        """
        self.use_realsense = False
        if self.camera_source == 'realsense':
            if self.realsense_cam is None:
                self._setup_realsense()
            self.use_realsense = True
        elif self.camera_source == 'webcam':
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.logger.error("[GUIInterface] Failed to open webcam!")
                self.cap = None
            else:
                # Give camera time to warm up
                import time
                time.sleep(0.5)
                # Try reading a few frames to ensure camera is ready
                for _ in range(5):
                    ret, _ = self.cap.read()
                    if ret:
                        break
                self.logger.info("[GUIInterface] Webcam opened successfully")
        elif self.camera_source == 'redis':
            pass

    def _stop_camera(self):
        if self.use_realsense and self.realsense_cam is not None:
            self.realsense_cam.shutdown()
        elif hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

    def _change_source(self, source):
        self._stop_camera()

        self.camera_source = source
        input_source = source if source in self.external_sources else "gui"

        with self.lock:
            self.updates["config"]=self.updates.get("config",{})
            self.updates["config"]["pipeline"] ={"input": {"source": input_source}}
        self._start_camera()

    def switch_camera(self):
        # if self.realsense_available:
        # self.camera_source = 'realsense' if self.camera_source == 'webcam' else 'webcam'
        self.camera_source = self.camera_sources[(self.camera_sources.index(self.camera_source)+1) % len(self.camera_sources)]
        self.logger.info(f"[GUIInterface] Switching camera to {self.camera_source}.")

        self._change_source(self.camera_source)

        

    def capture_frame(self):
        frame = None
        pc = None
        if self.camera_source == "realsense" and self.use_realsense and self.realsense_cam is not None:
            frames, color, depth  = self.realsense_cam.get_frames()
            aligned_frames, aligned_color, aligned_depth = self.realsense_cam.get_aligned_frames(frames)
            if aligned_color is None:
                return None
            
            aligned_color = np.asanyarray(aligned_color.get_data())
            frame = aligned_color
            _, pc = self.realsense_cam.get_depth_pc(aligned_depth)
            # reshape the point cloud
            H, W = aligned_color.shape[:2]
            pc = pc.reshape(H, W, 3)

        elif self.camera_source == 'webcam':
            if self.cap is None or not self.cap.isOpened():
                self.logger.warning("[GUIInterface] Webcam not available")
                return None
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("[GUIInterface] Failed to read frame from webcam")
                return None
            self.logger.debug(f"[GUIInterface] Captured frame: {frame.shape if frame is not None else 'None'}")
        elif self.camera_source == 'redis':
            frame = self.frame_new # updated externally in the update method
            pc = self.point_cloud_new

        elif self.camera_source == 'video':
            ret, frame = self.cap.read()       
            # loop video
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                return None
            
        elif self.camera_source == 'image':
            if self.frame_index == 0:
                ret, frame = self.cap.read()       
            else:
                frame = self.frame

        with self.lock:
            self.point_cloud = pc
            self.frame = frame
            self.frame_index +=1
        return frame

    def _draw_input_box(self, detections, width):
        with self.lock:
            queries_list = self.queries_list
        
        height = max(150, (len(queries_list)+self.locating_new_obj)*30+120)
        input_box = np.ones((height, width, 3), dtype=np.uint8) * 230
        
        # Add GUI Elements
        text_color = (0, 0, 0)
        cv2.putText(input_box, "Press ']' to switch camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        cv2.putText(input_box, "Press '\\' delete all objects", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        cv2.putText(input_box, "Press '[' load files", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        cv2.putText(input_box, "separate hierarchical searches using '.' e.g. person.face.eyes", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        cv2.putText(input_box, f"Input: {self.user_input}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

        # display camera sources as a vertical list on the top right with white background (alpha 0.5) and black text, the current source should be highlighted in green
        y_offset = 30
        for source in self.camera_sources:
            color = (0, 0, 0)
            if source == self.camera_source:
                color = (0, 200, 0)
            cv2.putText(input_box, source, (input_box.shape[1]-100, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20


        y_offset = 130
        detection_names = [d.class_name for d in detections]
        for text in queries_list:
            if text in detection_names:
                class_color = (0, 200, 0) 
            elif self.locating_new_obj:
                class_color = (0, 255, 255)
            else:
                class_color = (0, 0, 200)
            cv2.putText(input_box, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, class_color, 2)
            y_offset += 30
        if self.locating_new_obj:
            cv2.putText(input_box, "Computing...", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.input_box = input_box

    def run(self):
        cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)
        cv2.waitKey(1)  # Allow window to initialize

        # create trackbars for the thresholds
        for k,v  in self.thresholds.items():
            for key, value in v.items():
                cv2.createTrackbar(f"{k} {key}", "Camera Feed", int(value*100), 100, self._update_thresholds)

        self.user_input = ''
        self.logger.info("[GUIInterface] Starting GUI interface.")
        fps = 0
        last_frame_stamp = None
        while self.running:
            frame = self.capture_frame()

            if frame is None:
                frame_annotated=self.default_image.copy()
                # state the source of the frame
                cv2.putText(frame_annotated, f"Source: {self.camera_source}", (240, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                frame_annotated = frame.copy()
                # self.new_info = True
                if last_frame_stamp is not None:
                    time_diff = cv2.getTickCount() - last_frame_stamp
                    new_fps = cv2.getTickFrequency() / time_diff
                    fps = 0.9 * fps + 0.1 * new_fps if fps > 0 else new_fps # smooth the fps
                last_frame_stamp = cv2.getTickCount()
                cv2.putText(frame_annotated, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            
            # =========== Keyboard input ==============
            key = cv2.waitKey(1) & 0xFF
            self._handle_keyboard_inputs(key)

            # Update drag-and-drop loader if active
            if self.drag_drop_loader.active:
                self.drag_drop_loader.update()
        
            with self.lock:
                detections = self.detections

            cv2_window_width = cv2.getWindowImageRect("Camera Feed")[2]*2
            target_width = min(frame_annotated.shape[1], cv2_window_width)
            target_height = frame_annotated.shape[0] * target_width // frame_annotated.shape[1]

            # update interface in case frame width has changed
            if hasattr(self, 'input_box') and target_width != self.input_box.shape[1]:
                self.new_info = True
            # update the gui interface
            if self.new_info:
                self._draw_input_box(detections, width=target_width)
                self.new_info = False
                print(f"input: {self.user_input}")
            
            # self._update_3d_viewer()
            if self.show_3d_viewer and self.frame_index % self.update_every_n_frames == 0:
                self._display_point_cloud(self.point_cloud, self.frame)
                if self.open3d_vis is not None:
                    self._display_3d_detections(detections)


            # Draw on the frame
            frame_annotated = draw_elements_detections(frame_annotated, detections)
            frame_annotated = draw_segmentations(frame_annotated, detections)

            reshaped_frame_annotated = cv2.resize(frame_annotated, (target_width, target_height))
            # Combine the frame with the input box
            combined_frame = np.vstack((reshaped_frame_annotated, self.input_box))

            # resize frame if it is too large
            if combined_frame.shape[0] > self.max_height:
                scale = self.max_height/combined_frame.shape[0]
                combined_frame = cv2.resize(combined_frame, (0,0), fx=scale, fy=scale)

            cv2.imshow("Camera Feed", combined_frame.copy())

            with self.lock:
                point_cloud = self.point_cloud
            self._record_frame_and_pc(frame, point_cloud)
            
        if self.recording:
            self._stop_recording()
        self._stop_camera()
        cv2.destroyAllWindows()

    def _display_point_cloud(self, point_cloud, frame, render = True, detections = None):
        if point_cloud is not None:
            if self.open3d_vis is None:
                self.open3d_pcd = open3d.geometry.PointCloud()
                
            colors = frame.copy().reshape(-1, 3)
            pc = point_cloud.reshape(-1,3)

            #filter out the points
            if detections:
                for d in detections:
                    if d.points_indices is not None:
                        colors[d.points_indices] = d.color
                
            mask = np.all((pc > self.pc_limits[0]) & (pc < self.pc_limits[1]), axis=1)
            pc = pc[mask]
            colors = colors[mask]

            self.open3d_pcd.points = open3d.utility.Vector3dVector(pc[::self.pc_res])
            self.open3d_pcd.colors = open3d.utility.Vector3dVector(colors[::self.pc_res,::-1] / 255.0)
            

            if self.open3d_vis is None:
                # self.first_point_cloud = False
                self.open3d_vis = open3d.visualization.Visualizer()
                self.open3d_vis.create_window("3D Point Cloud")
                self.open3d_vis.add_geometry(self.open3d_pcd)
                axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                self.open3d_vis.add_geometry(axis)

                self.ctr = self.open3d_vis.get_view_control()
                self.ctr.set_up([0,0,1])
                self.ctr.set_front([1,0,0])
                pc_center = np.mean(pc, axis=0)
                self.ctr.set_lookat(pc_center)

            self.open3d_vis.update_geometry(self.open3d_pcd)
            if render:
                self.open3d_vis.poll_events()
                self.open3d_vis.update_renderer()

    def _display_3d_detections(self, elements, render = True):


        for i in range(len(elements), len(self.open3d_bboxes)):
            self.open3d_vis.remove_geometry(self.open3d_bboxes[i])
        self.open3d_bboxes = self.open3d_bboxes[:len(elements)]

        for i, element in enumerate(elements):
            if element.open3d_bbox is None:
                continue
            if i >= len(self.open3d_bboxes):
                new_bbox = True
                open3d_bbox= open3d.geometry.OrientedBoundingBox()
                pc = open3d.geometry.PointCloud()
            else:
                new_bbox = False
                open3d_bbox, pc = self.open3d_bboxes[i]
            
            open3d_bbox.center = element.open3d_bbox.center
            open3d_bbox.extent = element.open3d_bbox.extent
            open3d_bbox.R = element.open3d_bbox.R
            open3d_bbox.color = element.color/255.0
            # color = get_color(element.element_id)
            # open3d_bbox.color = color/255.0

            pc.points = open3d.utility.Vector3dVector(element.points)
            # pc.colors = open3d.utility.Vector3dVector(np.tile(color, (element.points.shape[0], 1))/255.0)
            
            # open3d_bbox = open3d_bbox.create_from_points(pc.points)
            # open3d_bbox.color = color/255.0
            # if self.open3d_pcd is not None and element.points_indices is not None:
            #     # recolor point cloud with the new bounding box colors for the point_indexes
            #     original_colors = self.open3d_pcd.colors
            #     original_colors = np.asarray(original_colors)
            #     original_colors[element.points_indices] = element.color / 255.0
            #     self.open3d_pcd.colors = open3d.utility.Vector3dVector(original_colors)

            if new_bbox:
                self.open3d_bboxes.append([open3d_bbox, pc])
                self.open3d_vis.add_geometry(open3d_bbox)
                # self.open3d_vis.add_geometry(pc)
            else:
                self.open3d_vis.update_geometry(open3d_bbox)
                # self.open3d_vis.update_geometry(pc)
                if element.points_indices is not None:
                    self.open3d_vis.update_geometry(self.open3d_pcd)

        if render:
            self.open3d_vis.poll_events()
            self.open3d_vis.update_renderer()

    def _update_3d_viewer(self):
        
        if hasattr(self, 'last_frame_index') and self.last_frame_index != self.frame_index:
            start_time = datetime.now()
            # pr = cProfile.Profile()
            # pr.enable()
            with self.lock:
                if self.show_3d_viewer and self.frame_index % self.update_every_n_frames == 0:
                    self._display_point_cloud(self.point_cloud, self.frame, render=(self.open3d_vis is None))
                    if self.open3d_vis is not None:
                        self._display_3d_detections(self.detections)
            # pr.disable()
            # pr.print_stats()

            dt = (datetime.now() - start_time)
            # convert to seconds
            dt = dt.total_seconds()
            if dt == 0:
                dt = 0.00001
            total_loop_time = (datetime.now() - self.last_time).total_seconds() if hasattr(self, 'last_time') else 1
            self.fps_3d_plot = 0.9*self.fps_3d_plot + 0.1*(1/dt) if hasattr(self, 'fps_3d_plot') and self.fps_3d_plot > 0 else 1/dt
            self.fps_main_loop = 0.9*self.fps_main_loop + 0.1*(1/total_loop_time) if hasattr(self, 'fps_main_loop') and self.fps_main_loop > 0 else 1/total_loop_time

            print(f"Time to update: {dt}\t total time {total_loop_time}\tfps {self.fps_main_loop}\t fps_3d_plot {self.fps_3d_plot}")

            # Check if Open3D visualizer is using GPU
            if open3d.core.cuda.is_available():
                print("[GUIInterface] Open3D is using GPU.")
            else:
                print("[GUIInterface] Open3D is using CPU.")
            self.last_time = start_time
        self.last_frame_index = self.frame_index


    def _handle_keyboard_inputs(self, key):
        """
        Handle keyboard inputs for the GUI interface.
        """
        valid_key = True
        if key == 27:  # ESC key
            self.running = False
            self.close()

        elif key == ord(']'):
            self._trigger_reset()
            self.switch_camera()

        elif key == 22:  # Ctrl+V key
            from PIL import ImageGrab

            img = ImageGrab.grabclipboard()
            # parse to np
            if img is not None:
                img = np.array(img)
                
            self.logger.info("[GUIInterface] Ctrl+V detected.")
        
        elif key == ord('='):
            # save all elements in a pickle file alongside the overall detection mask
            self.logger.info("[GUIInterface] Saving all elements.")
            with self.lock:
                detections = self.detections
                frame_shape = self.frame.shape[:2]
            if detections:
                mask = merge_segmentations(detections, frame_shape)
                
                # open a navigation window to select a folder
                Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
                
                folder_path = askdirectory(initialdir=self.output_dir)  # show an "Open" dialog box and return the path to the selected file
                if folder_path:
                    folder_path = os.path.join(folder_path, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
                    os.makedirs(folder_path, exist_ok=True)
                    self.logger.info(f"[GUIInterface] Selected folder: {folder_path}")
                    # save the elements
                    elements_file = os.path.join(folder_path, "elements.pkl")
                    with open(elements_file, 'wb') as f:
                        pickle.dump([el.to_dict() for el in detections], f, protocol=pickle.HIGHEST_PROTOCOL)
                        # pickle.dump(detections, f, protocol=pickle.HIGHEST_PROTOCOL)
                    # save the mask as a pickle file
                    mask_file = os.path.join(folder_path, "mask.pkl")
                    with open(mask_file, 'wb') as f:
                        pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)

                    #save mask as .npy
                    mask_npy_file = os.path.join(folder_path, "mask.npy")
                    np.save(mask_npy_file, mask)
                    
                    # save mask as a black and white png (overwrite if exists)
                    mask_png = mask.astype(np.uint8)
                    mask_png_file = os.path.join(folder_path, "mask.png")
                    cv2.imwrite(mask_png_file, mask_png)
                    self.logger.info(f"[GUIInterface] Saved elements to {elements_file}")

        elif key == ord('\\'):
            self.logger.info("[GUIInterface] Deleting all objects.")
            # with self.lock:
            #     self.objects_to_remove.extend(self.tracked_objects)
            self._trigger_reset()
            
            with self.lock:
                self.queries_list.clear()
            
        elif key == 13 and len(self.user_input) > 0:  # Enter key
            self.logger.info(f"[GUIInterface] Adding new object: {self.user_input}")
            
            with self.lock:
                self.queries_list.append(self.user_input)
                # self.tracked_objects.append(self.user_input)
                self.objects_to_add.append({
                    "class_name": self.user_input,
                    "compute_pose": True,
                    "compute_mask": True,
                    "track": True
                })
                self.locating_new_obj = True

            self.user_input = ''
        elif key == ord("'"):  # Start/stop recording
            if not self.recording:
                self._start_recording()
                self.recording = True
            else:
                self._stop_recording()
                self.recording = False
        elif key == ord('['):  # Toggle drag-and-drop window
            self.drag_drop_loader.toggle()
            status = "opened" if self.drag_drop_loader.active else "closed"
            self.logger.info(f"[GUIInterface] Drag & drop window {status}.")
        elif key == 8:  # Backspace key
            self.user_input = self.user_input[:-1]
        elif (key >= ord("a") and key <= ord("z")) or \
             (key >= ord("A") and key <= ord("Z")) or \
             (key>=ord("0") and key<=ord("9")) or \
             (key in [ord("."), ord(" "), ord("_")]):
            self.user_input += chr(key)
        elif key != 255:
            print("PRESSED:", key)
            valid_key = False
        else:
            valid_key = False
        if valid_key:
            self.new_info = True

    def _update_thresholds(self, value):
        # retrieve the trackbar values of all thresholds
        for k,v  in self.thresholds.items():
            for key, _ in v.items():
                if cv2.getTrackbarPos(f"{k} {key}", "Camera Feed") != -1:
                    self.thresholds[k][key] = float(cv2.getTrackbarPos(f"{k} {key}", "Camera Feed")/100)
        with self.lock:
            self.updates["thresholds"] = self.thresholds
    
    def start(self):
        """
        Start the GUI interface.
        Note: The actual run() method should be called from the main thread
        by the orchestrator for proper OpenCV GUI operation.
        This method is kept for standalone execution compatibility.
        """
        self.gui_thread = threading.Thread(target=self.run, daemon=True)
        self.gui_thread.start()

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
            frame = self.frame if self.frame is not None else None
            pc = self.point_cloud if self.point_cloud is not None else None
            index = self.frame_index
        
        return {"front": {"rgb":frame, "point_cloud": pc, "index": index}}
    
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

    def _start_recording(self):
        """
        Start recording RGB frames and point clouds.
        """
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
        if self.rgb_writer:
            self.rgb_writer.release()
            self.rgb_writer = None
        if self.pc_writer:
            self.pc_writer.close()
            self.pc_writer = None
        self.logger.info("[GUIInterface] Recording stopped.")

    def _record_frame_and_pc(self, frame, point_cloud):
        """
        Record the current frame and point cloud if recording is active.
        """
    
        if self.recording and frame is not None:
            self.rgb_writer.write(frame)
            if point_cloud is not None:
                pickle.dump(point_cloud, self.pc_writer, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _on_drag_drop_image_loaded(self, filepath: str, image: np.ndarray):
        """
        Callback when an image or video is loaded via drag-and-drop.
        
        Args:
            filepath: Path to the loaded image or video
            image: The loaded image as a numpy array, or None if it's a video
        """
        # Stop current camera source
        self._stop_camera()
        
        if image is None:
            # It's a video file
            self.logger.info(f"[GUIInterface] Loaded video via drag-and-drop: {filepath}")
            with self.lock:
                self.cap = cv2.VideoCapture(filepath)
                self.frame = None
                self.frame_index = 0
                self.camera_source = 'video'
        else:
            # It's an image file
            self.logger.info(f"[GUIInterface] Loaded image via drag-and-drop: {filepath}")
            # Set the loaded image as the current frame
            with self.lock:
                self.frame = image
                self.frame_index = 0
                self.camera_source = 'image'
                # Create a dummy video capture that just returns this image
                self.cap = type('obj', (object,), {
                    'read': lambda *args: (True, image),
                    'isOpened': lambda *args: True,
                    'set': lambda *args: None,
                    'release': lambda *args: None
                })()
        
        # Reset detections for new file
        self._trigger_reset()
        
if __name__ == "__main__":
    gui = GUIInterface({})
    gui.start()
    while gui.running:
        pass
