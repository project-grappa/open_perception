# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Online 3D depth video processing pipeline.

- Connects to a RGBD camera or RGBD video file (currently
  RealSense camera and bag file format are supported).
- Captures / reads color and depth frames. Allow recording from camera.
- Convert frames to point cloud, optionally with normals.
- Visualize point cloud video and results.
- Save point clouds and RGBD images for selected frames.

For this example, Open3D must be built with -DBUILD_LIBREALSENSE=ON
"""

import os
import json
import time
import logging as log
import argparse
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import platform
import sys
from open_perception.utils.visualization import draw_elements_detections, draw_segmentations, get_label_3d, get_color, merge_segmentations, oriented_bbox_to_line_mesh
import cv2


isMacOS = (platform.system() == "Darwin")


import os
import time
import cv2
import threading
import numpy as np

from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilename
from scipy.spatial.transform import Rotation as R
import open3d
from open_perception.logging.base_logger import Logger
from datetime import datetime

import open3d.visualization.gui as gui
from easydict import EasyDict


class Viewer2D:
    """
    Create a 2D viewer for displaying the detections
    """
    def __init__(self, camera_sources, thresholds, logger=None, **callbacks):
        # self.vfov = vfov
        # self.max_pcd_vertices = max_pcd_vertices
        self.callbacks = callbacks
        self.camera_sources = camera_sources
        self.queries_list = []
        self.user_input = ''
        self.locating_new_obj = False

        self.logger = logger if logger else Logger.get_logger(type(self).__name__)
        self.initialized = True

        # box to display infos 
        self.input_box = None
        self.max_height = 600
        self.new_info = True
        self.default_image=np.zeros((480,640,3), dtype=np.uint8)
        cv2.putText(self.default_image, "No camera feed", (240, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        self.thresholds = thresholds
        self.camera_source = None

        # 3d parameters TODO parse correctly
        self.show_3d_viewer = False
        self.first_point_cloud = True
        self.open3d_vis = None
        self.open3d_pcd = None
        self.open3d_bboxes = []
        self.pc_limits = np.array([[0, 0, 0], [1, 1, 1]])
        self.pc_res = 1
        self.update_every_n_frames = 1


    def _draw_input_box(self, detections, width):
        # with self.lock:
        # queries_list = self.queries_list
        
        height = max(150, (len(self.queries_list)+self.locating_new_obj)*30+120)
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
        for text in self.queries_list:
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

    def _update_3d_viewer(self, point_cloud, frame, detections):
        
        if hasattr(self, 'last_frame_index') and self.last_frame_index != self.frame_index:
            start_time = datetime.now()
            # pr = cProfile.Profile()
            # pr.enable()
            # with self.lock:
            if self.show_3d_viewer and self.frame_index % self.update_every_n_frames == 0:
                self._display_point_cloud(point_cloud, frame, render=(self.open3d_vis is None))
                if self.open3d_vis is not None:
                    self._display_3d_detections(detections, render=True)
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
        



    def update(self, viewer_payload):
        """
        Update the GUI with new frame, and detections
        """

        # for the first run:
        if self.initialized == False:
            cv2.namedWindow("Camera Feed")

            # create trackbars for the thresholds
            for k,v  in self.thresholds.items():
                for key, value in v.items():
                    cv2.createTrackbar(f"{k} {key}", "Camera Feed", int(value*100), 100, self._update_thresholds)
            self.initialized = True

        frame = viewer_payload.get("rgb_frame", None)
        point_cloud = viewer_payload.get("pcd_o3d", None)
        last_frame_stamp = viewer_payload.get("last_frame_stamp", None)
        self.frame_index = viewer_payload.get("frame_index", 0)
        fps = viewer_payload.get("fps", 0)
        self.camera_source = viewer_payload.get("camera_source", self.camera_source)
        detections = viewer_payload.get("detections", [])
        self.new_info = viewer_payload.get("new_info", self.new_info)

        # parse int numpy
        if isinstance(frame, open3d.t.geometry.Image):
            frame = np.asarray(frame)
        if isinstance(frame, open3d.t.geometry.RGBDImage):
            frame = np.asarray(frame.color)
        if isinstance(point_cloud, open3d.t.geometry.PointCloud):
            point_cloud = np.asarray(point_cloud.point.positions)

        # self.detections = detections
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

        # update interface in case frame width has changed
        if self.input_box is not None and frame_annotated.shape[1] != self.input_box.shape[1]:
            self.new_info = True
        # update the gui interface
        if self.new_info:
            self._draw_input_box(detections, width=frame_annotated.shape[1])
            self.new_info = False
            print("updated")
        
        # self._update_3d_viewer()
        if self.show_3d_viewer and self.frame_index % self.update_every_n_frames == 0:
            self._display_point_cloud(point_cloud, frame)
            if self.open3d_vis is not None:
                self._display_3d_detections(detections)


        # Draw on the frame
        frame_annotated = draw_elements_detections(frame_annotated, detections)
        frame_annotated = draw_segmentations(frame_annotated, detections)

        # Combine the frame with the input box
        combined_frame = np.vstack((frame_annotated, self.input_box))

        # resize frame if it is too large
        if combined_frame.shape[0] > self.max_height:
            scale = self.max_height/combined_frame.shape[0]
            combined_frame = cv2.resize(combined_frame, (0,0), fx=scale, fy=scale)

        cv2.imshow("Camera Feed", combined_frame)

        # self._record_frame_and_pc(frame, point_cloud)

    def _handle_keyboard_inputs(self, key):
        """
        Handle keyboard inputs for the GUI interface.
        """
        valid_key = True
        if key == 27:  # ESC key
            self.close()
            self.callbacks["close"]()

        elif key == ord(']'):
            self.callbacks["trigger_reset"]()
            self.callbacks["switch_to_next_camera"]()

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
            self.callbacks["save_elements"]()
            # with self.lock:
            #     detections = self.detections
            #     frame_shape = self.frame.shape[:2]
            # if detections:
            #     mask = merge_segmentations(detections, frame_shape)
                
            #     # open a navigation window to select a folder
            #     Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
                
            #     folder_path = askdirectory(initialdir=self.output_dir)  # show an "Open" dialog box and return the path to the selected file
            #     if folder_path:
            #         self.logger.info(f"[GUIInterface] Selected folder: {folder_path}")
            #         # save the elements
            #         elements_file = os.path.join(folder_path, "elements.pkl")
            #         with open(elements_file, 'wb') as f:
            #             pickle.dump([el.to_dict() for el in detections], f, protocol=pickle.HIGHEST_PROTOCOL)
            #             # pickle.dump(detections, f, protocol=pickle.HIGHEST_PROTOCOL)
            #         # save the mask as a pkl file
            #         mask_file = os.path.join(folder_path, "mask.pkl")
            #         with open(mask_file, 'wb') as f:
            #             pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)

            #         self.logger.info(f"[GUIInterface] Saved elements to {elements_file}")

        elif key == ord('['):
            # open a navigation window to select a video
            
            Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
            video_sufixes =  ["mp4", "avi", "mov"]
            images_sufixes = ["png", "jpg", "jpeg", "gif"]
            video_path = askopenfilename(filetypes=[("Videos and Images files", "*."+" *.".join(video_sufixes+images_sufixes))])  # show an "Open" dialog box and return the path to the selected file
            if video_path:
                self.logger.info(f"[GUIInterface] Selected video file: {video_path}")
                self.callbacks["stop_camera"]()
                self.cap = cv2.VideoCapture(video_path)
                # with self.lock:
                # self.frame = None
                self.frame_index = 0
                if video_path.endswith(tuple(video_sufixes)):
                    self.camera_source = 'video'
                elif video_path.endswith(tuple(images_sufixes)):
                    self.camera_source = 'image'
                else:
                    raise ValueError(f"Unsupported file format: {video_path}")
                self.callbacks["set_camera_source"](self.camera_source)
                self.callbacks["trigger_reset"]()

        elif key == ord('\\'):
            self.logger.info("[GUIInterface] Deleting all objects.")
            # with self.lock:
            #     self.objects_to_remove.extend(self.tracked_objects)
            self.callbacks["trigger_reset"]()
            
            self.queries_list.clear()
            
        elif key == 13 and len(self.user_input) > 0:  # Enter key
            self.logger.info(f"[GUIInterface] Adding new object: {self.user_input}")
            
            self.queries_list.append(self.user_input)
            self.locating_new_obj = True
            self.callbacks["add_query"]({
                    "class_name": self.user_input,
                    "compute_pose": True,
                    "compute_mask": True,
                    "track": True
                })
            # with self.lock:
            #     # self.tracked_objects.append(self.user_input)
            #     self.objects_to_add.append({
            #         "class_name": self.user_input,
            #         "compute_pose": True,
            #         "compute_mask": True,
            #         "track": True
            #     })
            #     self.locating_new_obj = True

            self.user_input = ''
        elif key == ord("'"):  # Start/stop recording
            if not self.recording:
                self.callbacks["start_recording"]()
                self.recording = True
            else:
                self.callbacks["stop_recording"]()
                self.recording = False
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
        self.callbacks["set_update"](thresholds = self.thresholds)
        # with self.lock:
        #     self.updates["thresholds"] = self.thresholds

    def close():
        cv2.destroyAllWindows()










class Viewer3D:
    """Controls display and user interface. All methods must run in the main thread."""

    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, **callbacks):
        """Initialize.

        Args:
            vfov (float): Vertical field of view for the 3D scene.
            max_pcd_vertices (int): Maximum point clud verties for which memory
                is allocated.
            callbacks (dict of kwargs): Callbacks provided by the controller
                for various operations.
        """

        self.vfov = vfov
        self.max_pcd_vertices = max_pcd_vertices

        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window(
            "Open3D || Online RGBD Video Processing", 1280, 1280)
        # Called on window layout (eg: resize)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])

        self.bbox_material = o3d.visualization.rendering.MaterialRecord()
        # self.bbox_material.shader = "defaultLit"

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        # self.pcd_material.shader = "defaultLit"
        # Set n_pixels displayed for each 3D point, accounting for HiDPI scaling
        self.pcd_material.point_size = int(4 * self.window.scaling)

        # 3D scene
        self.pcdview = gui.SceneWidget()
        self.window.add_child(self.pcdview)
        self.pcdview.enable_scene_caching(True)  # makes UI _much_ more responsive
        self.pcdview.scene = rendering.Open3DScene(self.window.renderer)
        self.pcdview.scene.set_background([1, 1, 1, 1])  # White background
        self.pcdview.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])
        # Point cloud bounds, depends on the sensor range
        self.pcd_bounds = o3d.geometry.AxisAlignedBoundingBox([-3, -3, 0],
                                                              [3, 3, 6])
        self.pcd_bounds_legacy = o3d.t.geometry.AxisAlignedBoundingBox.from_legacy(self.pcd_bounds)
        self.camera_view()  # Initially look from the camera
        
        em = self.window.theme.font_size
        separation_height = int(round(0.5 * em))

        # Options panel
        self.panel = gui.Vert(em, gui.Margins(em, em, em, em))
        self.panel.preferred_width = int(360 * self.window.scaling)
        self.window.add_child(self.panel)
        

        self.video_size = (int(240 * self.window.scaling),
                           int(320 * self.window.scaling), 3)
        self.show_color = gui.CollapsableVert("Color image")
        self.show_color.set_is_open(True)
        self.panel.add_child(self.show_color)
        self.color_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_color.add_child(self.color_video)

        self.show_depth = gui.CollapsableVert("Depth image")
        self.show_depth.set_is_open(False)
        self.panel.add_child(self.show_depth)
        self.depth_video = gui.ImageWidget(
            o3d.geometry.Image(np.zeros(self.video_size, dtype=np.uint8)))
        self.show_depth.add_child(self.depth_video)

        self.status_message = gui.Label("")
        self.panel.add_child(self.status_message)

    

        # Text input box for user input
        self.text_input = gui.TextEdit()
        self.text_input.placeholder_text = "Enter query..."
        self.panel.add_child(gui.Label("Add Query"))
        self.panel.add_child(self.text_input)

        self.add_query_button = gui.Button("Add Query")
        self.on_add_query = callbacks['on_add_query']
        self.add_query_button.set_on_clicked(self.on_add_query)
        self.panel.add_child(self.add_query_button)

        self.window.set_on_key(self._on_key_pressed)

        # Dropdown for queries list
        self.queries_dropdown = gui.Combobox()
        self.panel.add_child(gui.Label("Queries List"))
        self.panel.add_child(self.queries_dropdown)

        # Dropdown for camera source selection
        self.camera_source_dropdown = gui.Combobox()
        sources = ["webcam", "realsense", "video", "image", "Redis"]
        for source in sources:
            self.camera_source_dropdown.add_item(source)

        self.camera_source_dropdown.set_on_selection_changed(
            callbacks['on_camera_source_changed']
        )
        self.panel.add_child(gui.Label("Camera Source"))
        self.panel.add_child(self.camera_source_dropdown)

        # Views and controls:

        toggles = gui.Horiz(em)
        self.panel.add_child(toggles)

        toggle_capture = gui.ToggleSwitch("Capture / Play")
        toggle_capture.is_on = True
        toggle_capture.set_on_clicked(
            callbacks['on_toggle_capture'])  # callback
        toggles.add_child(toggle_capture)

        self.flag_normals = False
        self.toggle_normals = gui.ToggleSwitch("Colors / Normals")
        self.toggle_normals.is_on = False
        self.toggle_normals.set_on_clicked(
            callbacks['on_toggle_normals'])  # callback
        toggles.add_child(self.toggle_normals)

        view_buttons = gui.Horiz(em)
        self.panel.add_child(view_buttons)
        view_buttons.add_stretch()  # for centering
        camera_view = gui.Button("Camera view")
        camera_view.set_on_clicked(self.camera_view)  # callback
        view_buttons.add_child(camera_view)
        birds_eye_view = gui.Button("Bird's eye view")
        birds_eye_view.set_on_clicked(self.birds_eye_view)  # callback
        view_buttons.add_child(birds_eye_view)
        view_buttons.add_stretch()  # for centering

        save_toggle = gui.Horiz(em)
        self.panel.add_child(save_toggle)
        save_toggle.add_child(gui.Label("Record / Save"))
        self.toggle_record = None
        if callbacks['on_toggle_record'] is not None:
            save_toggle.add_fixed(1.5 * em)
            self.toggle_record = gui.ToggleSwitch("Video")
            self.toggle_record.is_on = False
            self.toggle_record.set_on_clicked(callbacks['on_toggle_record'])
            save_toggle.add_child(self.toggle_record)

        save_buttons = gui.Horiz(em)
        self.panel.add_child(save_buttons)
        # save_buttons.add_stretch()  # for centering
        save_pcd = gui.Button("Save Point cloud")
        save_pcd.set_on_clicked(callbacks['on_save_pcd'])
        save_buttons.add_child(save_pcd)
        save_rgbd = gui.Button("Save RGBD frame")
        save_rgbd.set_on_clicked(callbacks['on_save_rgbd'])
        save_buttons.add_child(save_rgbd)
        # save_buttons.add_stretch()  # for centering

        self.flag_exit = False
        self.flag_gui_init = False


        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", Viewer3D.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", Viewer3D.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", Viewer3D.MENU_OPEN)
            file_menu.add_item("Export Current Image...", Viewer3D.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", Viewer3D.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   Viewer3D.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(Viewer3D.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", Viewer3D.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu


        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        self.window.set_on_menu_item_activated(Viewer3D.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(Viewer3D.MENU_EXPORT,
                                     self._on_menu_export)
        self.window.set_on_menu_item_activated(Viewer3D.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(Viewer3D.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        self.window.set_on_menu_item_activated(Viewer3D.MENU_ABOUT, self._on_menu_about)
        # ----

        self.detections_geom_names = []

        self._apply_settings()

        self.render_option = o3d.visualization.RenderOption()
        self.render_option.show_coordinate_frame = True

        # Init thread to run while GUI is loading
        self.gui_loading = threading.Thread(
            target=self.gui_loading_thread, name="gui_loading_thread")
        self.gui_loading.start()



    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D GUI Example"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be on the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)


    def _on_menu_toggle_settings_panel(self):
        self.panel.visible = not self.panel.visible
        gui.Application.instance.menubar.set_checked(
            Viewer3D.MENU_SHOW_SETTINGS, self.panel.visible)
        
    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_export(self):
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".ply .stl .fbx .obj .off .gltf .glb",
            "Triangle mesh files (.ply, .stl, .fbx, .obj, .off, "
            ".gltf, .glb)")
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter(".ply", "Polygon files (.ply)")
        dlg.add_filter(".stl", "Stereolithography files (.stl)")
        dlg.add_filter(".fbx", "Autodesk Filmbox files (.fbx)")
        dlg.add_filter(".obj", "Wavefront OBJ files (.obj)")
        dlg.add_filter(".off", "Object file format (.off)")
        dlg.add_filter(".gltf", "OpenGL transfer files (.gltf)")
        dlg.add_filter(".glb", "OpenGL binary transfer files (.glb)")
        dlg.add_filter(".xyz", "ASCII point cloud files (.xyz)")
        dlg.add_filter(".xyzn", "ASCII point cloud with normals (.xyzn)")
        dlg.add_filter(".xyzrgb",
                       "ASCII point cloud files with colors (.xyzrgb)")
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.add_filter(".pts", "3D Points files (.pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _apply_settings(self):
        """Apply settings from the settings panel."""
        
        pass

    def gui_loading_thread(self):
        """Thread to run while the GUI is loading."""
        print("Loading GUI...", end="")
        def update_non_block():
            self.update({})
            self.app.run_one_tick()
        while self.flag_gui_init is False:
            time.sleep(0.1)
            print(".", end="")
            gui.Application.instance.post_to_main_thread(
                self.window,
                update_non_block
                # self.pcdview.force_redraw
            )
    
    def run(self):
        self.app.run()

    def update(self, viewer_payload):
        """Update visualization with point cloud and images. Must run in main
        thread since this makes GUI calls.

        Args:
            viewer_payload: dict {element_type: geometry element}.
                Dictionary of element types to geometry elements to be updated
                in the GUI:
                    'pcd_o3d': point cloud,
                    'rgb_frame': rgb image (3 channel, uint8),
                    'depth': depth image (uint8),
                    'status_message': message,
                    'detections': list of detections oriented bounding boxes
        """
        if not self.flag_gui_init:
            # Set dummy point cloud to allocate graphics memory
            
            if self.pcdview.scene.has_geometry('pcd_o3d'):
                self.pcdview.scene.remove_geometry('pcd_o3d')
            
            # self.pcd_material.shader = "normals" if self.flag_normals else "defaultLit"

            # coordinate frame
            if self.pcdview.scene.has_geometry('coordinate_frame'):
                self.pcdview.scene.remove_geometry('coordinate_frame')
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            self.pcdview.scene.add_geometry('coordinate_frame', coordinate_frame, self.bbox_material)


            bbox_lines = oriented_bbox_to_line_mesh(self.pcd_bounds)
            for j, l in enumerate(bbox_lines):
                # compute vertex normals
                # l.compute_vertex_normals()
                geom_name = f"scene_bbox_line{j}"
                if self.pcdview.scene.has_geometry(geom_name):
                    self.pcdview.scene.remove_geometry(geom_name)
                self.pcdview.scene.add_geometry(geom_name, l, self.bbox_material)
            self.flag_gui_init = True

        pcd = viewer_payload.get('pcd_o3d', None)
        self.display_pcd(pcd, 'pcd_o3d')

        # Display sphere to debug if pc is ordered
        # if pcd is not None:
        #     frame_index = viewer_payload.get('frame_index', 0)
        #     pos = pcd.point.positions[frame_index].numpy()
        #     # pos = pcd.select_by_index([frame_index]).point.positions.numpy()[0]
        #     # print(pos_old, "\t", pos)

        #     # Create a small sphere at the given position
        #     sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
        #     sphere.translate(pos)
        #     sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for visibility

        #     # Add the sphere to the scene
        #     if not self.pcdview.scene.has_geometry("highlight_sphere"):
        #         self.pcdview.scene.add_geometry("highlight_sphere", sphere, self.bbox_material)
        #     else:
        #         self.pcdview.scene.remove_geometry("highlight_sphere")
        #         self.pcdview.scene.add_geometry("highlight_sphere", sphere, self.bbox_material)
        
        detections = viewer_payload.get('detections', [])
        self._update_3d_detections(detections)

        # Draw on the frame
        if self.show_color.get_is_open() and 'rgb_frame' in viewer_payload:
            frame_annotated = viewer_payload.get('rgb_frame', None)
            # frame_annotated = np.asarray(viewer_payload['rgb_frame'].cpu(), dtype=np.uint8)
            # frame_annotated = draw_elements_detections(frame_annotated, detections)
            # frame_annotated = draw_segmentations(frame_annotated, detections)
            # frame_annotated = o3d.t.geometry.Image(frame_annotated.astype(np.uint8))
            
            if frame_annotated is not None:
                # TODO(ssheorey) Remove CPU transfer after we have CUDA -> OpenGL bridge
                sampling_ratio = self.video_size[1] / viewer_payload['rgb_frame'].columns
                self.color_video.update_image(
                    frame_annotated.resize(sampling_ratio))
                
        if self.show_depth.get_is_open() and 'colored_depth_frame' in viewer_payload:
            depth_frame = viewer_payload.get('colored_depth_frame', None)
            if depth_frame is not None:
                sampling_ratio = self.video_size[1] / depth_frame.columns
                self.depth_video.update_image(
                    depth_frame.resize(sampling_ratio).cpu())

        # if 'detections' in viewer_payload:
        self.pcdview.force_redraw()

    def display_pcd(self, pcd, geom_name):
        """Display point cloud in the 3D scene. 

        Args:
            pcd: point cloud to display.
            geom_name: name of the geometry to display.
        """
        if not self.pcdview.scene.has_geometry(geom_name):
            dummy_pcd = o3d.t.geometry.PointCloud({
                'positions':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'colors':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32),
                'normals':
                    o3d.core.Tensor.zeros((self.max_pcd_vertices, 3),
                                          o3d.core.Dtype.Float32)
            })
            self.pcdview.scene.add_geometry(geom_name, dummy_pcd, self.pcd_material)
            
        if os.name == 'nt':
            self.pcdview.scene.remove_geometry(geom_name)
            self.pcdview.scene.add_geometry(geom_name, pcd,
                                            self.pcd_material)
        else:
            update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                            rendering.Scene.UPDATE_COLORS_FLAG |
                            (rendering.Scene.UPDATE_NORMALS_FLAG
                             if self.flag_normals else 0))
            if pcd is not None:
                pcd = pcd.crop(self.pcd_bounds_legacy)
                self.pcdview.scene.scene.update_geometry(geom_name, pcd, update_flags)


    def _update_3d_detections(self, detections):

        update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                        rendering.Scene.UPDATE_COLORS_FLAG |
                        (rendering.Scene.UPDATE_NORMALS_FLAG
                            if self.flag_normals else 0))

        detections_geom_names = []
        # self.bbox_material.shader = "normals" if self.flag_normals else "defaultLit"

        # self.pcdview.clear_3d_labels()
        for i, det in enumerate(detections):
            geom_name = f"name-{det.class_name}_id-{det.element_id}"

            # add / update detection's points
            if det.points is not None:
                pcd_obj = o3d.t.geometry.PointCloud(det.points)
                pcd_obj.paint_uniform_color(det.color.astype(np.float32)/255)
                self.display_pcd(pcd_obj, f"{geom_name}_points")
                detections_geom_names.append(f"{geom_name}_points")
                
            # add / update the 3d bounding boxes
            if det.open3d_bbox is None:
                continue

            # add / update the detections 3d label
            if self.pcdview.scene.has_geometry(f"{geom_name}_label"):
                # self.pcdview.scene.remove_geometry(f"{geom_name}_label")
                pos = det.pose + det.open3d_bbox.extent / 2
                label_3d = get_label_3d(pos, det.class_name, det.color, 0.1)
                self.pcdview.scene.update_geometry(f"{geom_name}_label", label_3d, update_flags)
                detections_geom_names.append(f"{geom_name}_label")
                # self.pcdview.add_3d_label(pos, det.class_name)

            bbox_lines = oriented_bbox_to_line_mesh(det.open3d_bbox, 
                                                    radius=0.02,
                                                    color=det.color.astype(np.float32)/255.0)
            for j, l in enumerate(bbox_lines):
                name = f"{geom_name}_line{j}"

                if not self.pcdview.scene.has_geometry(name):
                    self.pcdview.scene.add_geometry(name, l, self.bbox_material)
                else:
                    self.pcdview.scene.remove_geometry(name)
                    self.pcdview.scene.add_geometry(name, l, self.bbox_material)
                    # self.pcdview.scene.scene.update_geometry(name, l, update_flags)

                detections_geom_names.append(name)

        # remove unused geometries
        for geom_name in self.detections_geom_names:
            if self.pcdview.scene.has_geometry(geom_name) and geom_name not in detections_geom_names:
                self.pcdview.scene.remove_geometry(geom_name)
        self.detections_geom_names = detections_geom_names

    def camera_view(self):
        """Callback to reset point cloud view to the camera"""
        center = self.pcd_bounds.get_center()
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 1])
        # Look at [0, 0, 1] from camera placed at [0, 0, 0] with Y axis
        # pointing at [0, -1, 0]
        self.pcdview.scene.camera.look_at(center, [0, 0, -1], [0, -1, 0])

    def birds_eye_view(self):
        """Callback to reset point cloud view to birds eye (overhead) view"""
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, [0, 0, 0])
        self.pcdview.scene.camera.look_at([0, 0, 1.5], [0, 3, 1.5], [0, -1, 0])

    def on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        """Callback on window initialize / resize"""
        frame = self.window.content_rect
        self.pcdview.frame = frame
        panel_size = self.panel.calc_preferred_size(layout_context,
                                                    self.panel.Constraints())
        width = panel_size.width #10 * layout_context.theme.font_size
        height = min(
            frame.height,
            self.panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self.panel.frame = gui.Rect(frame.get_right() - panel_size.width,
                                    frame.y, width, height)

    def update_queries_list(self, queries_list):
        """Update the dropdown with the current queries list."""
        self.queries_dropdown.clear_items()
        for query in queries_list:
            self.queries_dropdown.add_item(query)

    def _camera_source(self, camera_source):
        """Update the selected camera source in the dropdown."""
        for i in range(self.camera_source_dropdown.number_of_items):
            if self.camera_source_dropdown.get_item(i) == camera_source:
                self.camera_source_dropdown.selected_index = i
                break

    def _on_key_pressed(self, event):
        """Handle key press events in the text input."""
        # print("Key pressed:", event.key)
        if event.type == event.UP:
            # Handle up arrow key
            return
        
        if event.key == gui.KeyName.ENTER:  # Check if "Enter" key is pressed
            self.on_add_query()  # Simulate button click
        if int(event.key) >= int(gui.KeyName.A) and\
            int(event.key) <= int(gui.KeyName.Z) or\
            int(event.key) >= int(gui.KeyName.ZERO) and\
            int(event.key) <= int(gui.KeyName.NINE) or\
            int(event.key) == int(gui.KeyName.SPACE):
            # Append the pressed key to the text input
            self.text_input.text_value += chr(event.key)
        if event.key == gui.KeyName.BACKSPACE:
            # Remove the last character from the text input
            self.text_input.text_value = self.text_input.text_value[:-1]