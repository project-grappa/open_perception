# -*- coding: utf-8 -*-
"""
visualization.py

Helper functions for visualizing detections and segmentations on images.
"""

import cv2
import open3d as o3d
from open_perception.pipeline.element import Element
import numpy as np
from typing import Union, List, Any


COLORS = [
    [255, 1, 1],    # Red
    [1, 255, 1],    # Green
    [1, 1, 255],    # Blue
    [255, 255, 1],  # Cyan
    [255, 1, 255],  # Magenta
    [1, 255, 255],  # Yellow
    [128, 1, 1],    # Maroon
    [1, 128, 1],    # Dark Green
    [1, 1, 128],    # Navy
    [128, 128, 1],  # Olive
    [128, 1, 128],  # Purple
    [1, 128, 128],  # Teal
    [192, 192, 192],# Silver
    [128, 128, 128],# Gray
    [1, 1, 1],      # Black
    [255, 255, 255] # White
]

COLOR_TO_TEXT = { # TODO: CLEAN THIS
    (255, 1, 1): "Red",
    (1, 255, 1): "Green",
    (1, 1, 255): "Blue",
    (255, 255, 1): "Cyan",
    (255, 1, 255): "Magenta",
    (1, 255, 255): "Yellow",
    (128, 1, 1): "Maroon",
    (1, 128, 1): "Dark Green",
    (1, 1, 128): "Navy",
    (128, 128, 1): "Olive",
    (128, 1, 128): "Purple",
    (1, 128, 128): "Teal",
    (192, 192, 192): "Silver",
    (128, 128, 128): "Gray",
    (1, 1, 1): "Black",
    (255, 255, 255): "White"
}

def get_color(index):
    """
    Get a color from the COLORS palette by index.
    :param index: Index of the color in the palette.
    :return: BGR color tuple.
    """
    return np.array(COLORS[index % len(COLORS)], dtype=np.uint8) 

def get_iou(bbox1: Union[list, np.ndarray], bbox2: Union[list, np.ndarray]) -> float:
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.
    :param bbox1: First bounding box [x1, y1, x2, y2].
    :param bbox2: Second bounding box [x1, y1, x2, y2].
    :return: IoU value.
    """
    if isinstance(bbox1, list):
        bbox1 = np.array(bbox1)
    if isinstance(bbox2, list):
        bbox2 = np.array(bbox2)

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = area_bbox1 + area_bbox2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0

# Placeholder functions if you don't have a visualization.py:
def draw_elements_detections(frame, detections: list[Element], big_labels = False, shift_if_necessay=True) -> np.ndarray:
    """
    Draw bounding boxes with labels and scores on the frame.
    :param frame: BGR image (numpy)
    :param detections: list of dict {bbox: [x1, y1, x2, y2], class_name: str, score: float}
    :return: modified frame with bounding box overlays
    """
    all_labels_coords = []
    for det in detections:
        # print(det.meta)
        if len(det.bbox) == 4:
            # Assume format [x1, y1, x2, y2]
            x1, y1, x2, y2 = det.bbox
        elif len(det.bbox) == 2:
            # Assume format [x1, y1], [x2, y2]
            x1, y1 = det.bbox[0]
            x2, y2 = det.bbox[1]
        else:
            continue
        if det.color is not None:
            color = det.color
        else:
            color = get_color(det.element_id)
            det.color = color
            det.color_name = COLOR_TO_TEXT.get(tuple(color), f"RGB={list(color)}" )

        x1 = int(x1); y1 = int(y1); x2 = int(x2); y2 = int(y2)
        label = f"{det.class_name} {det.detection_prob:.2f}" if det.detection_prob is not None else det.class_name
        cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if big_labels:
            text_color = (255, 255, 255)
            # Calculate text size
            font_scale = 1.5
            thickness = 4
        else:
            text_color = color.tolist()
            font_scale = 0.5
            thickness = 1

        text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_w, text_h = text_size

        text_x1 = x1
        text_y1 = y1 
        # Shift the label position to avoid overlap with the bounding box
        if shift_if_necessay:
            # Try to avoid overlap with previous labels using IoU
            max_attempts = 10
            attempt = 0
            while attempt < max_attempts:
                overlap_found = False
                for prev_x1, prev_y1, prev_w, prev_h in all_labels_coords:
                    # Define current and previous label bounding boxes
                    curr_bbox = [text_x1, text_y1 - text_h - 8, text_x1 + text_w, text_y1 - 2]
                    prev_bbox = [prev_x1, prev_y1, prev_x1 + prev_w, prev_y1 + prev_h]
                    iou = get_iou(curr_bbox, prev_bbox)
                    if iou > 0:
                        # Shift right if overlap detected
                        text_x1 = prev_x1 + prev_w + 1
                        overlap_found = True
                        break
                if not overlap_found:
                    break
                attempt += 1

            # update the text_y1 to be above the rectangle
            all_labels_coords.append((text_x1, text_y1 - text_h - 8, text_w, text_h + 8))        

        if big_labels:
            # Draw filled rectangle as background
            cv2.rectangle(frame, (text_x1, text_y1 - text_h - 8), (text_x1 + text_w, text_y1 - 2), color.tolist(), thickness=cv2.FILLED)
        # Draw the label text
        cv2.putText(frame, label, (text_x1, text_y1 - 5), font, font_scale, text_color, thickness)


    return frame

def merge_segmentations(segmentations: list[Element], frame_shape: tuple[int, int]) -> np.ndarray:
    """
    Merge segmentation masks into a single frame-sized mask.
    :param
    segmentations: list of detected Element
    frame_shape: (height, width) of the frame
    :return: merged mask
    """
    # Initialize an empty mask
    mask = np.zeros(frame_shape, dtype=np.uint8)
    for det in segmentations:
        if det.segmentation_mask is None:
            continue
        # Ensure mask is the same size as the frame or the region is mapped appropriately
        mask += det.segmentation_mask[:,:,0]
    return mask.astype(np.uint8)

def draw_segmentations(frame, segmentations: list[Element]) -> np.ndarray:
    """
    Overlay segmentation masks on the frame. Each mask can be a 2D array of 0s and 1s
    or 0-255. We'll randomly color the mask for demonstration.
    :param frame: BGR image (OpenCV)
    :param segmentations: list of detected Element
    :return: modified frame with segmentation overlays
    """
    # Example: For each mask, pick a random color and blend

    for det in segmentations:
        mask = det.segmentation_mask
        if mask is None or frame is None or mask.shape[:2] != frame.shape[:2]:
            continue
        # Ensure mask is the same size as the frame or the region is mapped appropriately

        if det.color is not None:
            color = det.color
        else:
            color = get_color(det.element_id) # Use a fixed color based on ID
            det.color = color
            det.color_name = COLOR_TO_TEXT.get(tuple(color), f"RGB={list(color)}" )

        # We assume mask is the same H,W as the frame; if not, you'd need to resize or place it.
        colored_mask = (mask * color).astype(np.uint8)
        frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.5, 0)
        seg_prob = det.segmentation_prob
        
        label = f"{det.class_name} {seg_prob:.2f}"

        # Find the centroid of the mask to place the label
        M = cv2.moments(mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(frame, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 1)

    return frame





def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle



def get_label_3d(position, text, color=(1, 1, 1), size=0.1):
    """
    Create a 3D label at the specified position with the given text and color.
    Args:
        position (tuple): The (x, y, z) position of the label.
        text (str): The text to display.
        color (tuple): The color of the label in RGB format.
        size (float): The size of the label.
    """

    open3d_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=0.1).to_legacy()
    open3d_mesh.paint_uniform_color(color)


    # Scale down since default mesh is quite big
    # Location
    # I am adding another location shift as an example.
    open3d_mesh.transform([[size, 0, 0, position[0]], [0, size, 0, position[1]], [0, 0, size, position[2]],
                                [0, 0, 0, 1]])

    return open3d_mesh

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    def normalize_points(self, a, axis=-1, order=2):
        """Normalizes a numpy array of points"""
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis), l2

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = self.normalize_points(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)




def custom_line_set(lines: np.ndarray, color: np.ndarray, width: float=0.01) -> List[Any]:
    """
    Create custom line geometries for visualization.

    Args:
        lines (np.ndarray): Array of line start and end points.
        color (np.ndarray): Color of the lines.
        width (float): Width of the lines.

    Returns:
        List[Any]: List of Open3D geometries for the cylinder lines.
    """
    points = np.array(lines).reshape(-1, 3)
    lines = np.array([[i, i + 1] for i in range(0, len(points), 2)])
    line_mesh = LineMesh(points, lines, colors=color, radius=width)
    line_mesh_geoms = line_mesh.cylinder_segments
    
    return line_mesh_geoms



def oriented_bbox_to_line_mesh(bbox, radius=0.02, color=[1, 0, 0]):
    
    points = bbox.get_box_points()
    lines = [[2, 0], [0, 1], [1, 7], [2, 7],
              [4, 6], [6, 3], [3, 5], [5, 4],
                [0, 3], [1, 6], [7, 4], [2, 5]]

    colors = [color for i in range(len(lines))]

    line_mesh = LineMesh(points, lines, colors, radius=radius)

    return line_mesh.cylinder_segments

