"""
orchestrator.py

Coordinates the flow of data between detection and segmentation models.
Demonstrates using a GroundingDINO-like detection model and SAM2 segmentation.

Assumes:
    - A detection model that inherits from BaseDetectionModel (e.g. GroundingModel).
    - A segmentation model that inherits from BaseSegmentationModel (e.g. SAM2Model).
    - Optional aggregator or tracker if needed.
    - Communication modules can be used to publish results.

Usage:
    orchestrator = Orchestrator(
        config=config,
        tracker=tracker,
        aggregator=aggregator,
        comm_modules=comm_modules
    )
    orchestrator.run()
"""

from typing import List, Dict, Any
from open_perception.communication.base_interface import BaseInterface
from open_perception.perception.base_models import (
    BaseDetectionModel,
    BaseSegmentationModel,
)
from open_perception.state_estimation.base_state_estimator import BaseStateEstimator
from open_perception.pipeline.element import Element
from open_perception.logging.base_logger import Logger
import numpy as np
from datetime import datetime
from open_perception.utils.common import detections_to_elements
from open_perception.utils.config_loader import merge_dicts
from open_perception.utils.visualization import get_color
import threading


class Orchestrator:
    def __init__(
        self,
        config: dict,
        tracker=None,
        aggregator=None,
        comm_modules: list[BaseInterface] = None,
    ):
        """
        :param config: Dictionary containing the pipeline configuration (from default.yaml, etc.).
        :param tracker: An optional object tracker.
        :param aggregator: An optional aggregator to fuse multiple detection results.
        :param comm_modules: Dictionary of communication interfaces (gui, redis, ros, api, etc.).
        """
        # Initialize logger
        self.logger = Logger.get_logger("orchestrator", config)

        self._parse_config(config)
        self.tracker = tracker
        self.aggregator = aggregator
        self.comm_modules = comm_modules if comm_modules else {}
        self.running = True

        self.frame_idx = 0
        self.all_elements: Dict[str, list[Element]] = {}
        self.class_info = {}

        self.perception_thread = None
        self.models_initialized = False
        self._initialize_models()

    def _reset_models(self):
        """
        Reset models to their initial state.
        """
        for model in self.detection_models:
            model.reset()
        for model in self.segmentation_models:
            model.reset()
        if self.state_estimator is not None:
            self.state_estimator.reset()

    def _initialize_models(self):
        """
        Initialize detection and segmentation models based on the config.
        We'll assume config has pipeline.perception.models[] with model info.
        """
        # The orchestrator could hold references to multiple models.
        self.detection_models: list[BaseDetectionModel] = []
        self.segmentation_models: list[BaseSegmentationModel] = []
        self.state_estimator: BaseStateEstimator = None

        # retrieve the perception config from the pipeline config
        perception_config = self.config.get("pipeline", {}).get("perception", {})
        models_config = perception_config.get("models", {})
        detection_models_config = models_config.get("detection", [])
        segmentation_models_config = models_config.get("segmentation", [])
        vlm_search_config = perception_config.get("vlm_search", {})

        # Load the detection models
        for m in detection_models_config:
            if not m.get("enabled", True):
                continue
            name = m.get("name", "").lower()

            if "dummy" in name:
                from open_perception.perception.detection_models import (
                    DummyDetectionModel,
                )

                detection_model = DummyDetectionModel(model_config=m)
                detection_model.load_model()
            # If the name indicates a GroundingDINO-like detection model
            if "grounding" in name:
                # Import or reference your actual detection model class
                from open_perception.perception.detection_models import (
                    GroundingDinoModelHF,
                )

                detection_model = GroundingDinoModelHF(model_config=m)
                detection_model.load_model()
            elif "yoloworld" in name:
                from open_perception.perception.detection_models import YoloWorldModel

                detection_model = YoloWorldModel(model_config=m)
                detection_model.load_model()
            elif "lookup" in name:
                from open_perception.perception.detection_models import (
                    LookUpDetectionModel,
                )

                detection_model = LookUpDetectionModel(model_config=m)
                detection_model.load_model()
            # Add more detection models here if needed
            else:
                self.logger.error(f"Model {name} not found.")
                continue

            # Apply wrapper to perform multi-granular detection using the model specified
            if m.get("multi_granular", False):
                self.logger.info("Multi-granular model detected.")
                from open_perception.perception.detection_models import (
                    MultigranularDetectionModel,
                )

                detection_model = MultigranularDetectionModel(
                    model_config=m, detection_model=detection_model
                )

            # Apply the VLM search wrapper if specified
            if m.get("vlm_search", False):
                from open_perception.perception.vlm_based_models import (
                    VLMDetectionWithSearch,
                )

                detection_model = VLMDetectionWithSearch(
                    vlm_config=vlm_search_config,
                    detection_model=detection_model,
                )

            self.detection_models.append(detection_model)

        # load the segmentation models
        for m in segmentation_models_config:
            if not m.get("enabled", True):
                continue
            name = m.get("name", "").lower()
            # If the name indicates an SAM2 segmentation model
            if "sam2" in name:
                from open_perception.perception.segmentation_models import SAM2Model

                seg_model = SAM2Model(model_config=m)
                seg_model.load_model()
                self.segmentation_models.append(seg_model)

        # Log a warning if no models were found
        if not self.detection_models:
            self.logger.warning("No detection model found or enabled in config.")
        if not self.segmentation_models:
            self.logger.warning("No segmentation model found or enabled in config.")

        # Initialize state estimator if needed
        state_estimator_config = perception_config.get("state_estimation", {})
        self.logger.debug(state_estimator_config)
        for state_estimator in state_estimator_config:
            if not state_estimator.get("enabled", True):
                continue
            self.logger.debug(state_estimator)
            name = state_estimator.get("name", "").lower()
            if "pca" in name:
                from open_perception.state_estimation.base_state_estimator import (
                    PCAStateEstimator,
                )

                self.state_estimator = PCAStateEstimator(config=state_estimator)
            # Add more state estimators here if needed
        self.models_initialized = True

    def _parse_config(self, config: Dict):
        """
        Parse the pipeline configuration and set up the orchestrator.
        """
        self.config = config

        # Retrieve input source info from config
        input_config = self.config.get("pipeline", {}).get("input", {})
        self.source = input_config.get("source", "gui")

        self.logger.info(f"Starting main loop with input source: {self.source}")


    def perception_loop(self):
        """
        Main loop of the orchestrator. Waits for frames from the input source, runs detection,
        then segmentation on each bounding box, optionally tracks or aggregates, and
        publishes results via comm_modules.
        """
        last_frame_idx = {}
        # Loop reading frames until exit condition
        while self.running:
            # 1. Get the latest frame from the input source
            frames_dict = self.comm_modules[self.source].get_synced_frame_and_pc()
            self._check_for_updates()

            if not frames_dict:  # No frames received.
                continue

            for sensor_name, frames_data in frames_dict.items():
                frame = frames_data["rgb"]
                frame_idx = frames_data["index"]
                point_cloud = frames_data["point_cloud"]
                depth_frame = frames_data.get("depth_frame")
                frame_metadata = frames_data.get("frame_metadata")

                if frame is None:
                    continue
                self.logger.debug(f"Processing frame {frame_idx} from {sensor_name}")

                # 2. check for updates on the list of classes to be tracked
                queries_updated = self._check_for_new_queries()
                if queries_updated:
                    # get of all elements bounding boxes
                    elements = self.detect_elements(frame, frame_idx)

                    # track the old elements in the new frame to update their masks
                    old_elements = self.all_elements.get(sensor_name, [])
                    if (
                        last_frame_idx.get(sensor_name, -1) != frame_idx
                        and len(old_elements) > 0
                    ):
                        self.all_elements[sensor_name] = self._track_elements(
                            frame, old_elements, frame_idx
                        )

                    # merge old detections with new ones
                    self.all_elements[sensor_name] = self.merge_elements(
                        self.all_elements.get(sensor_name, []), elements
                    )
                    # add new prompts to the segmentation model
                    self.update_segmentation_model(
                        frame, self.all_elements[sensor_name], frame_idx
                    )

                # track, compute pose and broadcast the results
                if last_frame_idx.get(sensor_name, -1) != frame_idx or queries_updated:
                    # self.logger.error(f"tracking elements in frame {frame_idx}")

                    # 3. track elements already detected in the frame
                    self.all_elements[sensor_name] = self._track_elements(
                        frame, self.all_elements.get(sensor_name, []), frame_idx
                    )

                    # 4. compute pose of the detections
                    self.all_elements[sensor_name] = self._compute_pose(
                        self.all_elements[sensor_name],
                        point_cloud,
                        depth_frame,
                        frame_metadata,
                        frame,
                    )

                    # 5. Publish the results via communication modules
                    self._publish_results(
                        self.all_elements[sensor_name], frame_idx, sensor_name
                    )
                last_frame_idx[sensor_name] = frame_idx

            self._update_comm_modules({"frames": frames_dict, "source": self.source})

        self.logger.info("Exiting main loop.")

    def run(self):
        """
        Start the orchestrator's main loop in a separate thread. And launches the gui interface (if enabled) in the main thread.
        """

        # Start the perception loop in a separate thread
        self.perception_thread = threading.Thread(
            target=self.perception_loop, daemon=True
        )
        self.perception_thread.start()
        # if "gui" in self.comm_modules and self.comm_modules["gui"].is_enabled():
        #     self.comm_modules["gui"].run()

        self.perception_thread.join()

        self.logger.info("Orchestrator run completed.")

    def _compute_pose(
        self,
        elements: list[Element],
        point_cloud=None,
        depth_frame=None,
        frame_metadata=None,
        frame=None,
    ):
        """
        Uses the state estimator to compute the pose of each element.
        """
        if self.state_estimator is not None:
            for element in elements:
                class_info = self.class_info.get(element.class_name, {})
                if class_info.get("compute_pose", True):
                    elements = self.state_estimator.estimate_state(
                        elements,
                        pointcloud=point_cloud,
                        depth_frame=depth_frame,
                        frame_metadata=frame_metadata,
                        frame=frame,
                    )

        return elements

    def _check_for_new_queries(self):
        """
        Check for updates on the list of classes to be tracked and update the models accordingly.
        """
        class_info_updated = self.class_info.copy()
        for module_name, comm_module in self.comm_modules.items():
            self.logger.info(f"Checking for new queries in {comm_module}")
            new_classes_info = comm_module.get_classes_to_add()
            self.logger.info(f"New classes to add: {new_classes_info}")
            for class_info in new_classes_info:
                class_name = class_info.get("class_name")
                if class_name in class_info_updated:
                    # Update existing class info
                    class_info_updated[class_name].update(
                        {
                            "compute_pose": class_info.get(
                                "compute_pose",
                                class_info_updated[class_name].get(
                                    "compute_pose", True
                                ),
                            ),
                            "compute_mask": class_info.get(
                                "compute_mask",
                                class_info_updated[class_name].get(
                                    "compute_mask", True
                                ),
                            ),
                            "track": class_info.get(
                                "track",
                                class_info_updated[class_name].get("track", True),
                            ),
                        }
                    )
                else:
                    # Add new class info
                    class_info_updated[class_name] = {
                        "compute_pose": class_info.get("compute_pose", True),
                        "compute_mask": class_info.get("compute_mask", True),
                        "track": class_info.get("track", True),
                    }

            classes_to_remove = comm_module.get_classes_to_remove()
            self.logger.info(f"Classes to remove: {classes_to_remove}")
            for class_to_remove in classes_to_remove:
                if class_to_remove in class_info_updated:
                    del class_info_updated[class_to_remove]
            for class_to_remove in classes_to_remove:  # TODO: Improve this
                for sensor_name, elements in self.all_elements.items():
                    for element in elements:
                        if element.class_name == class_to_remove:
                            self.all_elements[sensor_name].remove(element)

        queries_updated = False
        if self.class_info != class_info_updated:
            self.class_info = class_info_updated
            list_of_classes = list(self.class_info.keys())
            for model in self.detection_models:
                model.update_classes(list_of_classes)
            self.logger.info(f"Updated classes: {list_of_classes}")
            queries_updated = True
        self.logger.info(f"Classes being tracked: {list(self.class_info.keys())}")

        return queries_updated

    def reset(self):
        """
        Reset the orchestrator and all models.
        """
        self.all_elements = {}
        self.class_info = {}
        self._reset_models()
        self._publish_results([], frame_idx=0)

    def _check_for_updates(self):
        """
        Check for updates from communication modules (e.g., new classes to track).
        """
        for module_name, comm_module in self.comm_modules.items():
            updates = comm_module.get_updates()
            if updates:
                self.logger.info(f"Updates from {module_name}: {updates}")
                # Process updates here, e.g., update models, thresholds, etc.

                if "thresholds" in updates:
                    thresholds = updates["thresholds"]
                    # update model's attributes with new thresholds
                    for model in self.detection_models + self.segmentation_models:
                        if model is not None and hasattr(model, "update_thresholds"):
                            model.update_thresholds(thresholds)
                if "reset" in updates:
                    if updates["reset"]:
                        self.logger.info("Resetting models.")
                        self.reset()

                if (
                    "available_detections" in updates
                ):  # used by the LookUpDetectionModel
                    # list dicts containing detections performed by external models
                    available_detections = updates["available_detections"]
                    for model in self.detection_models:
                        if hasattr(model, "update_available_detections"):
                            model.update_available_detections(available_detections)

                if "config" in updates:
                    # update the self.config with the new config overrides
                    config = merge_dicts(self.config, updates["config"])
                    self.reset()
                    self._parse_config(config)
                    self.logger.info("Updated configuration.")
                    # update the models with the new config
                    # self._initialize_models()

    def merge_elements(self, old_elements: list[Element], new_elements: list[Element]):
        """
        Merge new elements with old elements and return the updated list.
        Checks if new detections bbox of the same class overlap with old detections, if so, reuse previous detection element.

        """
        IOU_THRESHOLD = 0.9

        def calculate_iou(bbox1, bbox2):
            (x1, y1), (x2, y2) = bbox1
            (x3, y3), (x4, y4) = bbox2
            x_overlap = max(0, min(x2, x4) - max(x1, x3))
            y_overlap = max(0, min(y2, y4) - max(y1, y3))
            intersection = x_overlap * y_overlap
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (x4 - x3) * (y4 - y3)
            union = area1 + area2 - intersection
            iou = intersection / union
            return iou

        old_names_to_elements = {}
        for old_element in old_elements:
            old_names_to_elements[old_element.class_name] = old_names_to_elements.get(
                old_element.class_name, []
            ) + [old_element]

        for new_element in new_elements:
            for old_element in old_names_to_elements.get(new_element.class_name, []):
                if new_element.class_name == old_element.class_name:
                    # check if the new bbox overlaps with the old bbox
                    new_bbox = new_element.bbox
                    old_bbox = old_element.bbox
                    # get intersection over union  of the two bounding boxes
                    iou = calculate_iou(new_bbox, old_bbox)
                    if iou > IOU_THRESHOLD:
                        # self.logger.debug(f"Reusing old detection for element {new_element.element_id}")
                        new_id = new_element.element_id
                        new_element = old_element
                        new_element.is_reused = True
                        new_element.element_id = new_id
                        break
        return new_elements

    def detect_elements(self, frame, frame_idx=0, sensor_name=None):
        """
        Detect elements in the frame.
        Returns a list of Element objects.
        """
        start_time = datetime.now()
        all_detections = []
        for model in self.detection_models:
            all_detections.append(model.run_inference(frame))

        # detections is a list of dicts like:
        # [
        #   {
        #       'bbox': [[x1, y1], [x2, y2]],
        #       'class': 'something',
        #       'prob': 0.90,
        #       'id' : 1
        #   },
        #   ...
        # ]

        # 2. For each detected bounding box, create an Element object
        # combine detections from multiple detection models
        detections = []
        for (
            model_detections
        ) in all_detections:  # TODO: aggregate detections from multiple models
            detections.extend(model_detections)
        end_time = datetime.now()

        elements = detections_to_elements(detections, frame, frame_idx, sensor_name)

        detection_classes = [element.class_name for element in elements]
        for i, class_name in enumerate(self.class_info.keys()):
            # assign a color to each class
            self.class_info[class_name]["color"] = self.class_info[class_name].get(
                "color", get_color(i)
            )

            # determine classes not found
            if class_name not in detection_classes:
                self.logger.debug(f"Class {class_name} not found in detections.")
                self.class_info[class_name]["lost_count"] = (
                    self.class_info[class_name].get("lost", 0) + 1
                )
            else:
                self.class_info[class_name]["lost_count"] = 0

        # update elements metadata
        for element in elements:
            element.meta["detection_time"] = (end_time - start_time).total_seconds()
        return elements

    def update_segmentation_model(self, frame, elements: list[Element], frame_idx):
        """
        Add new prompts to the segmentation model.
        """
        if not self.segmentation_models:
            return

        for element in elements:
            if element.is_parent:
                continue
            class_info = self.class_info.get(element.class_name, {})
            if not class_info.get("compute_mask", True):
                continue
            for seg_model in self.segmentation_models:
                if element.segmentation_mask is None:
                    self.logger.warning(
                        f"Adding segmentation prompt BBOX for element {element.element_id}"
                    )
                    seg_model.add_new_prompt(
                        frame,
                        ann_frame_idx=frame_idx,
                        ann_obj_id=element.element_id,
                        bbox=element.bbox,
                    )
                else:
                    self.logger.error(
                        f"Adding segmentation prompt mask for element {element.element_id}"
                    )
                    seg_model.add_new_prompt(
                        frame,
                        ann_frame_idx=frame_idx,
                        ann_obj_id=element.element_id,
                        mask=element.segmentation_mask,
                    )
        return

    def _track_elements(self, frame, elements: list[Element], frame_idx=0):
        """
        Track elements in the frame.
        """
        if not self.segmentation_models:
            return elements

        start_time = datetime.now()
        all_seg_results = []
        for seg_model in self.segmentation_models:
            seg_results = seg_model.run_inference(frame, frame_idx=frame_idx)
            all_seg_results.append(seg_results)
        end_time = datetime.now()

        # combine the segmentation results from all models
        seg_results = (
            all_seg_results[0] if all_seg_results else []
        )  # TODO: combine results from multiple models

        id_to_seg_results = {}
        for seg_result in seg_results:
            id_to_seg_results[seg_result["obj_id"]] = seg_result
        # update the elements with the segmentation results
        for element in elements:
            class_info = self.class_info.get(element.class_name, {})
            if not class_info.get("track", True):
                continue
            if element.element_id not in id_to_seg_results:
                self.logger.warning(
                    f"No segmentation results found for element '{element.class_name}' with ID {element.element_id}"
                )
                # element.segmentation_mask = None
                # element.segmentation_prob = 0
                # element.bbox = None
                element.lost_count += 1
            else:
                # for seg_result in seg_results:
                # if seg_result["obj_id"] == element.element_id:
                seg_result = id_to_seg_results[element.element_id]

                element.lost_count = 0
                element.segmentation_mask = seg_result["mask"]
                element.segmentation_prob = seg_result["prob"]

                y_indices, x_indices = np.where(seg_result["mask"][:, :, 0] > 0)

                if x_indices.size == 0 or y_indices.size == 0:
                    self.logger.warning(
                        f"No valid segmentation mask found for element {element.element_id}"
                    )
                    continue
                # Get the bounding box coordinates
                x_min, x_max = np.min(x_indices), np.max(x_indices)
                y_min, y_max = np.min(y_indices), np.max(y_indices)
                element.bbox = np.array([[x_min, y_min], [x_max, y_max]])

                # update the element's metadata
                element.meta["segmentation_time"] = (
                    end_time - start_time
                ).total_seconds()

        return elements

    def _parse_results(self, elements: list[Element]):
        """
        Summarize detected and segmented elements into a dict of results
        """
        results = []
        for element in elements:
            result = {
                "element_id": element.element_id,
                "class_name": element.class_name,
                "detection_prob": element.detection_prob,
                "bbox": element.bbox,
                "segmentation_mask": element.segmentation_mask,
                "segmentation_prob": element.segmentation_prob,
                "meta": element.meta,
            }
            results.append(result)
        return results

    def _update_comm_modules(self, updates=None):
        """
        Update communication modules (if any) based on the current state.
        """
        if not updates:
            return
        for module_name, comm_module in self.comm_modules.items():
            # print(updates)
            comm_module.update(updates=updates)

    def _publish_results(self, results, frame_idx=0, sensor_name=None):
        """
        Publishes results via communication modules (if any).
        Could publish to Redis, send via REST, or publish a ROS topic.
        """
        self.logger.info("Publishing Results")
        # if not results:
        #     self.logger.info("No results to publish.")
        #     return

        for module_name, comm_module in self.comm_modules.items():
            comm_module.publish_results(results, id=frame_idx, sensor_name=sensor_name)
