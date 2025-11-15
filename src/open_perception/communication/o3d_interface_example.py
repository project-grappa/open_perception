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
from easydict import EasyDict

from open_perception.utils.visualization import draw_elements_detections, draw_segmentations, get_color, merge_segmentations


isMacOS = (platform.system() == "Darwin")


import cv2
class RGBCamera:
    """
    Collects RGB frames from available camera devices.
    """
    
    def __init__(self):
        self.metadata = {} 
        self.cap = None
        self.fake_depth = True
    
    def init_sensor(self, config=None, filename=None):
        """Initialize the camera sensor with configuration."""
        self.cap = cv2.VideoCapture(-1)
        self.filename = filename
        # showing values of the properties 
        print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))) 
        print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) 
        print("CAP_PROP_FPS : '{}'".format(self.cap.get(cv2.CAP_PROP_FPS))) 
        print("CAP_PROP_POS_MSEC : '{}'".format(self.cap.get(cv2.CAP_PROP_POS_MSEC))) 
        print("CAP_PROP_FRAME_COUNT  : '{}'".format(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) 
        print("CAP_PROP_BRIGHTNESS : '{}'".format(self.cap.get(cv2.CAP_PROP_BRIGHTNESS))) 
        print("CAP_PROP_CONTRAST : '{}'".format(self.cap.get(cv2.CAP_PROP_CONTRAST))) 
        print("CAP_PROP_SATURATION : '{}'".format(self.cap.get(cv2.CAP_PROP_SATURATION))) 
        print("CAP_PROP_HUE : '{}'".format(self.cap.get(cv2.CAP_PROP_HUE))) 
        print("CAP_PROP_GAIN  : '{}'".format(self.cap.get(cv2.CAP_PROP_GAIN))) 
        print("CAP_PROP_CONVERT_RGB : '{}'".format(self.cap.get(cv2.CAP_PROP_CONVERT_RGB))) 

        # get intrinsic matrix from config

        if config is None:
            config = {}
        width = config.get("width", self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = config.get("height", self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fov = config.get("fov", 60)
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

        self.metadata = EasyDict({
            "width": width,
            "height": height,
            "device_name": "CV2 Webcam",
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "intrinsics": {
                "intrinsic_matrix": intrinsic_matrix,
            },
            "serial_number": ""
        })

    def start_capture(self, start_record=False):
        """Start capturing frames from the camera."""
        if not self.cap.isOpened():
            log.error("Camera not initialized. Call init_sensor first.")
            return
        log.info("Started capturing frames.")
        self.recording = start_record
        if self.recording:
            self.video_writer = self._init_video_writer()

    def stop_capture(self):
        """Stop capturing frames from the camera."""
        if self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        log.info("Stopped capturing frames.")

    def pause_record(self):
        """Pause recording."""
        if self.recording:
            self.recording = False
            log.info("Recording paused.")

    def resume_record(self):
        """Resume recording."""
        if not self.recording:
            self.recording = True
            if not hasattr(self, 'video_writer') or self.video_writer is None:
                self.video_writer = self._init_video_writer()
            log.info("Recording resumed.")

    def capture_frame(self, **kwargs):
        """Capture a frame from the camera."""
        ret, frame = self.cap.read() 
        if not ret:
            log.error("Error reading frame from camera.")
            return None
        if self.recording and hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.write(frame)

        # Convert to Open3D image
        if self.fake_depth:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(frame), 
                o3d.geometry.Image(np.zeros_like(frame)), 
                convert_rgb_to_intensity=False
            )
        else:
            rgbd = o3d.geometry.Image(frame)
        return rgbd

    def _init_video_writer(self):
        """Initialize the video writer for recording."""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        filename = self.filename if self.filename else "output.avi"
        return cv2.VideoWriter(filename, fourcc, self.metadata["fps"],
                                (int(self.metadata["width"]), int(self.metadata["height"])))

    def get_metadata(self):
        """Get the camera metadata."""
        return self.metadata


# Camera and processing
class PipelineModel:
    """Controls IO (camera, video file, recording, saving frames). Methods run
    in worker threads."""

    def __init__(self,
                 update_view,
                 camera_config_file=None,
                 rgbd_video=None,
                 device=None):
        """Initialize.

        Args:
            update_view (callback): Callback to update display elements for a
                frame.
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.update_view = update_view
        if device:
            self.device = device.lower()
        else:
            self.device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
        self.o3d_device = o3d.core.Device(self.device)

        self.flag_change_camera = False
        self.gui_camera_sources = ["realsense","webcam", "video", "image"]
        self.external_sources = ["redis"]
        self.camera_sources = self.gui_camera_sources + self.external_sources
        self.camera_source = "realsense"


        self.video = None
        self.camera = None
        self.flag_capture = False
        self.cv_capture = threading.Condition()  # condition variable
        self.recording = False  # Are we currently recording
        self.flag_record = False  # Request to start/stop recording
        if rgbd_video:  # Video file
            self.video = o3d.t.io.RGBDVideoReader.create(rgbd_video)
            self.rgbd_metadata = self.video.metadata
            self.status_message = f"Video {rgbd_video} opened."

        else:  # RGBD camera
            now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"{now}.bag"
            self.camera = self._get_camera(0)
            if camera_config_file:
                with open(camera_config_file) as ccf:
                    self.camera.init_sensor(o3d.t.io.RealSenseSensorConfig(
                        json.load(ccf)),
                                            filename=filename)
            else:
                self.camera.init_sensor(filename=filename)
            self.camera.start_capture(start_record=False)
            self.rgbd_metadata = self.camera.get_metadata()
            self.status_message = f"Camera {self.rgbd_metadata.device_name} {self.rgbd_metadata.serial_number} opened."

        log.info(self.rgbd_metadata)

        # RGBD -> PCD
        self.extrinsics = o3d.core.Tensor.eye(4,
                                              dtype=o3d.core.Dtype.Float32,
                                              device=self.o3d_device)
        
        self.intrinsic_matrix = o3d.core.Tensor(
            self.rgbd_metadata.intrinsics.intrinsic_matrix,
            dtype=o3d.core.Dtype.Float32,
            device=self.o3d_device)
        
        self.depth_max = 3.0  # m
        self.pcd_stride = 2  # downsample point cloud, may increase frame rate
        self.flag_normals = False
        self.flag_save_rgbd = False
        self.flag_save_pcd = False
        self.flag_exit = False

        self.executor = ThreadPoolExecutor(max_workers=3,
                                           thread_name_prefix='Capture-Save')
        self.pcd_frame = None
        self.rgbd_frame = None

        
    
    def _get_camera(self, source_index):
        source = self.camera_sources[source_index]
        if source == "webcam":
            return RGBCamera()
        elif source == "realsense":
            return o3d.t.io.RealSenseSensor()

    @property
    def max_points(self):
        """Max points in one frame for the camera or RGBD video resolution."""
        return self.rgbd_metadata.width * self.rgbd_metadata.height

    @property
    def vfov(self):
        """Camera or RGBD video vertical field of view."""
        return np.rad2deg(2 * np.arctan(self.intrinsic_matrix[1, 2].item() /
                                        self.intrinsic_matrix[1, 1].item()))

    def run(self):
        """Run pipeline."""
        n_pts = 0
        frame_id = 0
        t1 = time.perf_counter()
        if self.video:
            self.rgbd_frame = self.video.next_frame()
        else:
            self.rgbd_frame = self.camera.capture_frame(
                wait=True, align_depth_to_color=True)

        pcd_errors = 0
        while (not self.flag_exit and
               (self.video is None or  # Camera
                (self.video and not self.video.is_eof()))):  # Video
            if self.video:
                future_rgbd_frame = self.executor.submit(self.video.next_frame)
            else:
                future_rgbd_frame = self.executor.submit(
                    self.camera.capture_frame,
                    wait=True,
                    align_depth_to_color=True)

            if self.flag_save_pcd:
                self.save_pcd()
                self.flag_save_pcd = False
            try:
                self.rgbd_frame = self.rgbd_frame.to(self.o3d_device)
                self.pcd_frame = o3d.t.geometry.PointCloud.create_from_rgbd_image(
                    self.rgbd_frame, self.intrinsic_matrix, self.extrinsics,
                    self.rgbd_metadata.depth_scale, self.depth_max,
                    self.pcd_stride, self.flag_normals)
                depth_in_color = self.rgbd_frame.depth.colorize_depth(
                    self.rgbd_metadata.depth_scale, 0, self.depth_max)
            except RuntimeError:
                pcd_errors += 1

            if self.pcd_frame.is_empty():
                log.warning(f"No valid depth data in frame {frame_id})")
                continue

            n_pts += self.pcd_frame.point.positions.shape[0]
            if frame_id % 60 == 0 and frame_id > 0:
                t0, t1 = t1, time.perf_counter()
                log.debug(f"\nframe_id = {frame_id}, \t {(t1-t0)*1000./60:0.2f}"
                          f"ms/frame \t {(t1-t0)*1e9/n_pts} ms/Mp\t")
                n_pts = 0
            frame_elements = {
                'color': self.rgbd_frame.color.cpu(),
                'depth': depth_in_color.cpu(),
                'pcd': self.pcd_frame.cpu(),
                'status_message': self.status_message,
                'detections': [] #self.detections.cpu() if self.detections else []
            }
            self.update_view(frame_elements)

            if self.flag_save_rgbd:
                self.save_rgbd()
                self.flag_save_rgbd = False
            self.rgbd_frame = future_rgbd_frame.result()
            with self.cv_capture:  # Wait for capture to be enabled
                self.cv_capture.wait_for(
                    predicate=lambda: self.flag_capture or self.flag_exit)
            self.toggle_record()
            frame_id += 1

            # update camera stream
            if self.flag_change_camera:
                self.camera.stop_capture()
                self.camera = self._get_camera(self.camera_source)
                self.camera.init_sensor()
                self.camera.start_capture()
                self.flag_change_camera = False
                self.flag_capture = True

        if self.camera:
            self.camera.stop_capture()
        else:
            self.video.close()
        self.executor.shutdown()
        log.debug(f"create_from_depth_image() errors = {pcd_errors}")

    def toggle_record(self):
        """Toggle recording state."""
        if self.camera is not None:
            if self.flag_record and not self.recording:
                self.camera.resume_record()
                self.recording = True
            elif not self.flag_record and self.recording:
                self.camera.pause_record()
                self.recording = False

    def save_pcd(self):
        """Save current point cloud."""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.rgbd_metadata.serial_number}_pcd_{now}.ply"
        # Convert colors to uint8 for compatibility
        self.pcd_frame.point.colors = (self.pcd_frame.point.colors * 255).to(
            o3d.core.Dtype.UInt8)
        self.executor.submit(o3d.t.io.write_point_cloud,
                             filename,
                             self.pcd_frame,
                             write_ascii=False,
                             compressed=True,
                             print_progress=False)
        self.status_message = f"Saving point cloud to {filename}."

    def save_rgbd(self):
        """Save current RGBD image pair."""
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"{self.rgbd_metadata.serial_number}_color_{now}.jpg"
        self.executor.submit(o3d.t.io.write_image, filename,
                             self.rgbd_frame.color)
        filename = f"{self.rgbd_metadata.serial_number}_depth_{now}.png"
        self.executor.submit(o3d.t.io.write_image, filename,
                             self.rgbd_frame.depth)
        self.status_message = (
            f"Saving RGBD images to {filename[:-3]}.{{jpg,png}}.")


class PipelineView:
    """Controls display and user interface. All methods must run in the main thread."""

    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    DEFAULT_IBL = "default"

    def __init__(self, vfov=60, max_pcd_vertices=1 << 20, camera_sources=["realsense", "webcam"], **callbacks):
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

        gui.Application.instance.initialize()
        self.window = gui.Application.instance.create_window(
            "Open3D || Online RGBD Video Processing", 1280, 960)
        # Called on window layout (eg: resize)
        self.window.set_on_layout(self.on_layout)
        self.window.set_on_close(callbacks['on_window_close'])

        self.pcd_material = o3d.visualization.rendering.MaterialRecord()
        self.pcd_material.shader = "defaultLit"
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
        self.show_color.set_is_open(False)
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

    
        # Dropdown for queries list
        self.queries_dropdown = gui.Combobox()
        self.panel.add_child(gui.Label("Queries List"))
        self.panel.add_child(self.queries_dropdown)

        # Text input box for user input
        self.text_input = gui.TextEdit()
        self.text_input.placeholder_text = "Enter query..."
        self.panel.add_child(gui.Label("Add Query"))
        self.panel.add_child(self.text_input)

        add_query_button = gui.Button("Add Query")
        add_query_button.set_on_clicked(callbacks['on_add_query'])
        self.panel.add_child(add_query_button)


        # Dropdown for camera source selection
        self.camera_source_dropdown = gui.Combobox()
        for source in camera_sources:
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
        toggle_capture.is_on = False
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
                app_menu.add_item("About", PipelineView.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", PipelineView.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", PipelineView.MENU_OPEN)
            file_menu.add_item("Export Current Image...", PipelineView.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", PipelineView.MENU_QUIT)
            settings_menu = gui.Menu()
            settings_menu.add_item("Lighting & Materials",
                                   PipelineView.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(PipelineView.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", PipelineView.MENU_ABOUT)

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
        self.window.set_on_menu_item_activated(PipelineView.MENU_OPEN, self._on_menu_open)
        self.window.set_on_menu_item_activated(PipelineView.MENU_EXPORT,
                                     self._on_menu_export)
        self.window.set_on_menu_item_activated(PipelineView.MENU_QUIT, self._on_menu_quit)
        self.window.set_on_menu_item_activated(PipelineView.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        self.window.set_on_menu_item_activated(PipelineView.MENU_ABOUT, self._on_menu_about)
        # ----

        self._apply_settings()

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
            PipelineView.MENU_SHOW_SETTINGS, self.panel.visible)
        
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

    def update(self, frame_elements):
        """Update visualization with point cloud and images. Must run in main
        thread since this makes GUI calls.

        Args:
            frame_elements: dict {element_type: geometry element}.
                Dictionary of element types to geometry elements to be updated
                in the GUI:
                    'pcd': point cloud,
                    'color': rgb image (3 channel, uint8),
                    'depth': depth image (uint8),
                    'status_message': message
        """
        if not self.flag_gui_init:
            # Set dummy point cloud to allocate graphics memory
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
            if self.pcdview.scene.has_geometry('pcd'):
                self.pcdview.scene.remove_geometry('pcd')

            self.pcd_material.shader = "normals" if self.flag_normals else "defaultLit"
            self.pcdview.scene.add_geometry('pcd', dummy_pcd, self.pcd_material)
            self.flag_gui_init = True

        # TODO(ssheorey) Switch to update_geometry() after #3452 is fixed
        if os.name == 'nt':
            self.pcdview.scene.remove_geometry('pcd')
            self.pcdview.scene.add_geometry('pcd', frame_elements['pcd'],
                                            self.pcd_material)
        else:
            update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                            rendering.Scene.UPDATE_COLORS_FLAG |
                            (rendering.Scene.UPDATE_NORMALS_FLAG
                             if self.flag_normals else 0))
            self.pcdview.scene.scene.update_geometry('pcd',
                                                     frame_elements['pcd'],
                                                     update_flags)

        # Update color and depth images
        frame_annotated = frame_elements['color']
        detections = frame_elements.get('detections', [])

        # if self.show_3d_viewer and self.frame_index % self.update_every_n_frames == 0:
        #         self._display_point_cloud(self.point_cloud, self.frame)
        #         if self.open3d_vis is not None:
        self._update_3d_detections(detections)

        # Draw on the frame
        frame_annotated = draw_elements_detections(frame_annotated, detections)
        frame_annotated = draw_segmentations(frame_annotated, detections)


        # TODO(ssheorey) Remove CPU transfer after we have CUDA -> OpenGL bridge
        if self.show_color.get_is_open() and 'color' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['color'].columns
            self.color_video.update_image(
                frame_annotated.resize(sampling_ratio).cpu())
        if self.show_depth.get_is_open() and 'depth' in frame_elements:
            sampling_ratio = self.video_size[1] / frame_elements['depth'].columns
            self.depth_video.update_image(
                frame_elements['depth'].resize(sampling_ratio).cpu())

        if 'status_message' in frame_elements:
            self.status_message.text = frame_elements["status_message"]

        self.pcdview.force_redraw()


    def _update_3d_detections(self, detections):

        update_flags = (rendering.Scene.UPDATE_POINTS_FLAG |
                        rendering.Scene.UPDATE_COLORS_FLAG |
                        (rendering.Scene.UPDATE_NORMALS_FLAG
                            if self.flag_normals else 0))

        for det in detections:
            geom_name = f"{det.class_name}_id:{det.element_id}"
            if self.pcdview.scene.has_geometry(geom_name):
                self.pcdview.scene.remove_geometry(geom_name)
            
            self.pcdview.scene.scene.update_geometry(geom_name,
                                                     det.bbox,
                                                     update_flags)


    def camera_view(self):
        """Callback to reset point cloud view to the camera"""
        center = self.pcd_bounds.get_center()
        self.pcdview.setup_camera(self.vfov, self.pcd_bounds, center)
        # Look at [0, 0, 1] from camera placed at [0, 0, 0] with Y axis
        # pointing at [0, -1, 0]
        self.pcdview.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, -1, 0])

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
        width = 17 * layout_context.theme.font_size
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

    def update_camera_source(self, camera_source):
        """Update the selected camera source in the dropdown."""
        for i in range(self.camera_source_dropdown.number_of_items):
            if self.camera_source_dropdown.get_item(i) == camera_source:
                self.camera_source_dropdown.selected_index = i
                break


class PipelineController:
    """Entry point for the app. Controls the PipelineModel object for IO and
    processing  and the PipelineView object for display and UI. All methods
    operate on the main thread.
    """

    def __init__(self, camera_config_file=None, rgbd_video=None, device=None):
        """Initialize.

        Args:
            camera_config_file (str): Camera configuration json file.
            rgbd_video (str): RS bag file containing the RGBD video. If this is
                provided, connected cameras are ignored.
            device (str): Compute device (e.g.: 'cpu:0' or 'cuda:0').
        """
        self.pipeline_model = PipelineModel(self.update_view,
                                            camera_config_file, rgbd_video,
                                            device)

        self.queries_list = []
        self.camera_source = "realsense"

        self.pipeline_view = PipelineView(
            1.25 * self.pipeline_model.vfov,
            self.pipeline_model.max_points,
            camera_sources=self.pipeline_model.camera_sources,
            on_window_close=self.on_window_close,
            on_toggle_capture=self.on_toggle_capture,
            on_save_pcd=self.on_save_pcd,
            on_save_rgbd=self.on_save_rgbd,
            on_toggle_record=self.on_toggle_record
            if rgbd_video is None else None,
            on_toggle_normals=self.on_toggle_normals,
            on_camera_source_changed=self.on_camera_source_changed,
            on_add_query=self.on_add_query
        )

        threading.Thread(name='PipelineModel',
                         target=self.pipeline_model.run).start()
        gui.Application.instance.run()

    def update_view(self, frame_elements):
        """Updates view with new data. May be called from any thread.

        Args:
            frame_elements (dict): Display elements (point cloud and images)
                from the new frame to be shown.
        """
        gui.Application.instance.post_to_main_thread(
            self.pipeline_view.window,
            lambda: self.pipeline_view.update(frame_elements))

    def on_toggle_capture(self, is_enabled):
        """Callback to toggle capture."""
        self.pipeline_model.flag_capture = is_enabled
        if not is_enabled:
            self.on_toggle_record(False)
            if self.pipeline_view.toggle_record is not None:
                self.pipeline_view.toggle_record.is_on = False
        else:
            with self.pipeline_model.cv_capture:
                self.pipeline_model.cv_capture.notify()

    def on_toggle_record(self, is_enabled):
        """Callback to toggle recording RGBD video."""
        self.pipeline_model.flag_record = is_enabled

    def on_toggle_normals(self, is_enabled):
        """Callback to toggle display of normals"""
        self.pipeline_model.flag_normals = is_enabled
        self.pipeline_view.flag_normals = is_enabled
        self.pipeline_view.flag_gui_init = False

    def on_window_close(self):
        """Callback when the user closes the application window."""
        self.pipeline_model.flag_exit = True
        with self.pipeline_model.cv_capture:
            self.pipeline_model.cv_capture.notify_all()
        return True  # OK to close window

    def on_save_pcd(self):
        """Callback to save current point cloud."""
        self.pipeline_model.flag_save_pcd = True

    def on_save_rgbd(self):
        """Callback to save current RGBD image pair."""
        self.pipeline_model.flag_save_rgbd = True

    def on_camera_source_changed(self, combobox, source):
        """Callback to handle camera source selection."""
        self.camera_source = source
        self.pipeline_model.flag_capture = False
        self.pipeline_model.flag_record = False
        self.pipeline_model.flag_change_camera = True
        self.pipeline_model.camera_source = source
        log.info(f"Camera source changed to: {source}")
        self.pipeline_view.update_camera_source(source)

    def on_add_query(self):
        """Callback to handle adding a new query."""
        query = self.pipeline_view.text_input.text_value
        if query:
            self.queries_list.append(query)
            self.pipeline_view.update_queries_list(self.queries_list)
            log.info(f"Added query: {query}")
            self.pipeline_view.text_input.text_value = ""  # Clear the input box after adding the query


if __name__ == "__main__":

    log.basicConfig(level=log.INFO)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--camera-config',
                        help='RGBD camera configuration JSON file')
    parser.add_argument('--rgbd-video', help='RGBD video file (RealSense bag)')
    parser.add_argument('--device',
                        help='Device to run computations. e.g. cpu:0 or cuda:0 '
                        'Default is CUDA GPU if available, else CPU.')

    args = parser.parse_args()

    if args.camera_config and args.rgbd_video:
        log.critical(
            "Please provide only one of --camera-config and --rgbd-video arguments"
        )
    else:
        PipelineController(args.camera_config, args.rgbd_video, args.device)