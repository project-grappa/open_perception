"""
detection_modules.py
This module contains example implementations of detection models for object detection tasks.

Classes:
    DummyDetectionModel: A dummy detection model that generates random bounding boxes and probabilities for given classes.
    GroundingDinoModel: A GroundingDINO-based detection model.
    YoloWorldModel: A YOLO-world based detection model.
    MultigranularDetectionModel: A multigranular detection model that detects classes by searching for them inside bounding boxes of possible parent classes.
    LookUpDetectionModel: A detection model that searches the list of object names to match a text input using semantic similarity.

Each detection model inherits from the BaseDetectionModel and implements methods for loading the model, running inference, updating classes, and resetting the model.
"""

import os
import numpy as np
from typing import List
from torchvision.ops import box_convert
from PIL import Image
from .base_models import BaseDetectionModel
from open_perception.logging.base_logger import Logger
import torch
import matplotlib.pyplot as plt
from open_perception.utils.visualization import draw_elements_detections
from open_perception.utils.common import detections_to_elements
from transformers import __version__ as _transformers_version
from packaging.version import Version
import io

# Suppress FutureWarnings from transformers and other libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
class DummyDetectionModel(BaseDetectionModel):
    """
    A dummy detection model that generates random bounding boxes and probabilities for given classes.
    """

    def load_model(self):
        self.model = "Dummy"
        self.classes = [] 

    def run_inference(self, image):
        dummy_detection = []
        for i in range(0, len(self.classes)):
            random_bbox = np.array(
                [
                    [
                        np.random.uniform(0, image.shape[1]),
                        np.random.uniform(0, image.shape[0]),
                    ],
                    [
                        np.random.uniform(0, image.shape[1]),
                        np.random.uniform(0, image.shape[0]),
                    ],
                ],
                dtype=np.int32,
            )
            # sort the bbox to make sure the first point is the top left corner and the second point is the bottom right corner
            random_bbox = np.sort(random_bbox, axis=0)

            random_prob = np.random.uniform(0.5, 1.0)
            detection = {
                "bbox": random_bbox,
                "class": self.classes[i],
                "prob": random_prob,
                "id": i,
            }
            dummy_detection.append(detection)
        return dummy_detection

    def update_classes(self, classes):
        self.classes = classes


class GroundingDinoModel(BaseDetectionModel):
    """
    detection model that uses the GroundingDINO model for object detection. 
    """

    def load_model(self):
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import load_model as load_dino_model
        from groundingdino.util.inference import predict as predict_dino

        from groundingdino import __file__ as gounding_dino_path
        from open_perception import __file__ as open_perception_path
        self.gorunding_dino_transform = T

        model_file = self.model_config.get(
            "model_file",
            os.path.dirname(gounding_dino_path) + "/config/GroundingDINO_SwinB_cfg.py",
        )
        ckpt_path = self.model_config.get(
            "checkpoint",
            os.path.dirname(open_perception_path)
            + "/../../checkpoints/groundingdino_swinb_cogcoor.pth",
        )
        self.logger = Logger.get_logger(type(self).__name__, self.model_config.get("logging", None))
        self.logger.info(f"[GroundingDinoModel] Loading model from {ckpt_path}...")
        self.logger.info(f"[GroundingDinoModel] Using config file {model_file}...")

        self.model = load_dino_model(model_file, ckpt_path)

        self.box_threshold = self.model_config.get("box_threshold", 0.35)
        self.text_threshold = self.model_config.get("text_threshold", 0.25)
        self.logger.info(
            f"[GroundingDinoModel] Box threshold: {self.box_threshold}, Text threshold: {self.text_threshold}"
        )

        self.predict = predict_dino

    def update_thresholds(self, thresholds):
        # self.model.set_thresholds(thresholds)

        self.logger.info("[GroundingDinoModel] Updating thresholds")
        self.box_threshold = thresholds.get(self.model_config["name"], {}).get(
            "box_threshold", self.box_threshold
        )
        self.text_threshold = thresholds.get(self.model_config["name"], {}).get(
            "text_threshold", self.text_threshold
        )

    def preprocess(self, image):
        transform = self.gorunding_dino_transform.Compose(
            [
                self.gorunding_dino_transform.RandomResize([800], max_size=1333),
                self.gorunding_dino_transform.ToTensor(),
                self.gorunding_dino_transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(image)
        # image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image_transformed

    def run_inference(self, image):
        """
        Runs the grounding-based detection.
        """

        H, W = image.shape[:2]
        if not self.classes:
            return []

        # 1. Preprocess
        image_new = self.preprocess(image)

        # 2. Forward pass
        boxes, logits, phrases = self.predict(
            model=self.model,
            image=image_new,
            caption=self.classes,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )
        # e.g.
        # detections = [
        #     {
        #         'bbox': [[50, 50], [150, 150]],
        #         'class': 'detected_object',
        #         'prob': 0.90,
        #         'id': 1
        #     }
        # ]
        # print(boxes, logits, phrases)

        detections = []
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for i in range(0, len(boxes)):
            bbox = xyxy[i]
            bbox = np.array(
                [
                    [float(bbox[0]) * W, float(bbox[1]) * H],
                    [float(bbox[2]) * W, float(bbox[3]) * H],
                ],
                dtype=np.int32,
            )
            detections.append(
                {
                    "bbox": bbox,
                    "class": phrases[i],
                    "prob": float(logits[i]),  # 1/ (1 + np.exp(-logits[i])),
                    "id": i + 1,
                }
            )

        # 3. Postprocess
        # detections = self.postprocess(detections)

        return detections

    def update_classes(self, classes: list[str]):
        # e.g.  "chair . person . dog ."
        if len(classes) > 0:
            self.classes = " . ".join(classes) + " ."
        else:
            self.classes = ""

    def reset(self):
        self.classes = ""
        # self.model.reset()


class GroundingDinoModelHF(BaseDetectionModel):
    """
    detection model that uses the GroundingDINO model for object detection. 
    """

    def load_model(self):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

        model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)

        
        self.logger = Logger.get_logger(type(self).__name__, self.model_config.get("logging", None))

        # self.model = load_dino_model(model_file, ckpt_path)

        self.box_threshold = self.model_config.get("box_threshold", 0.35)
        self.text_threshold = self.model_config.get("text_threshold", 0.25)
        
        self.logger.info(
            f"[GroundingDinoModel] Box threshold: {self.box_threshold}, Text threshold: {self.text_threshold}"
        )

        # self.predict = predict_dino

    def update_thresholds(self, thresholds):
        # self.model.set_thresholds(thresholds)

        self.logger.info("[GroundingDinoModel] Updating thresholds")
        self.box_threshold = thresholds.get(self.model_config["name"], {}).get(
            "box_threshold", self.box_threshold
        )
        self.text_threshold = thresholds.get(self.model_config["name"], {}).get(
            "text_threshold", self.text_threshold
        )

    def preprocess(self, image):
        image_source = Image.fromarray(image)
        inputs = self.processor(images=image_source, text=self.classes, return_tensors="pt").to(self.device)

        return inputs

    def run_inference(self, image):
        """
        Runs the grounding-based detection.
        """
        H, W = image.shape[:2]
        if not self.classes:
            self.logger.warning("[GroundingDinoModelHF] No classes provided for detection.")
            return []

        # 1. Preprocess
        inputs = self.preprocess(image)

        # 2. Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # deal with deprecated versions of transformers
            if Version(_transformers_version) < Version("4.51.0"):
                threshold_kwargs = {
                    "box_threshold": self.box_threshold,
                    "text_threshold": self.text_threshold,
                }
            else:
                threshold_kwargs = {
                    "threshold": self.box_threshold
                }
            
            results = self.processor.post_process_grounded_object_detection(
                                    outputs,
                                    inputs.input_ids,
                                    **threshold_kwargs,
                                    target_sizes=[(H, W)])[0]
            boxes = results["boxes"].cpu().numpy().astype(np.int32)
            probs = results["scores"].cpu().numpy()
            phrases = results["labels"]
        # e.g.
        # detections = [
        #     {
        #         'bbox': [[50, 50], [150, 150]],
        #         'class': 'detected_object',
        #         'prob': 0.90,
        #         'id': 1
        #     }
        # ]
        # print(boxes, logits, phrases)

        detections = []
        # xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        for i in range(0, len(boxes)):
            bbox = boxes[i].reshape(2, 2)
            detections.append(
                {
                    "bbox": bbox,
                    "class": phrases[i],
                    "prob": float(probs[i]),  # 1/ (1 + np.exp(-logits[i])),
                    "id": i + 1,
                }
            )

        # 3. Postprocess
        # detections = self.postprocess(detections)

        return detections

    def update_classes(self, classes: list[str]):
        # e.g.  "chair . person . dog ."
        if len(classes) > 0:
            self.classes = " . ".join(classes) + " ."
        else:
            self.classes = ""

    def reset(self):
        self.classes = ""
        # self.model.reset()


class YoloWorldModel(BaseDetectionModel):
    """
    detection model that uses the YOLO-world model for object detection.
    """

    def load_model(self):
        script_path = os.path.dirname(os.path.realpath(__file__))

        ckpt_path = self.model_config.get(
            "checkpoint",
            os.path.join(script_path, "../../../checkpoints/yolov8x-worldv2.pt"),
        )
        self.logger = Logger.get_logger(type(self).__name__, self.model_config.get("logging", None))
        self.logger.info(f"[YoloWorldModel] Loading model from {ckpt_path}...")

        try:
            from ultralytics import YOLOWorld
        except ImportError:
            self.logger.error("YOLOWorld not found. Please install the Ultralytics package.")
            raise ImportError

        self.classes = []
        self.model = YOLOWorld(ckpt_path)

    def update_classes(self, classes: list[str]):
        self.classes = classes
        if len(classes) > 0:
            self.model.set_classes(classes)

    def run_inference(self, image):
        # 1. Preprocess
        image = self.preprocess(image)

        # 2. Forward pass
        results = self.model.predict(image)

        # e.g.
        # detections = [
        #     {
        #         'bbox': [[30, 30], [200, 200]],
        #         'class': 'person',
        #         'prob': 0.95,
        #         'id': 1
        #     },
        #     {
        #         'bbox': [[55, 20], [300, 100]],
        #         'class': 'person',
        #         'prob': 0.70,
        #         'id': 1
        #     },
        #     {
        #         'bbox': [[250, 100], [320, 210]],
        #         'class': 'cat',
        #         'prob': 0.88,
        #         'id': 2
        #     }
        # ]
        detections = []
        for i, r in enumerate(results):
            bboxes = r.boxes.xyxy.to("cpu").numpy()
            probs= r.boxes.conf.to("cpu").numpy()
            clss = r.boxes.cls.to("cpu").numpy()
            for bbox, prob, cls in zip(bboxes, probs, clss):
                bbox = bbox.astype(int)
                bbox = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                detections.append(
                    {"bbox": bbox, "class": r.names[cls], "prob": float(prob), "id": i}
                )

        # 3. Postprocess
        detections = self.postprocess(detections)

        return detections

    def reset(self):
        self.classes = ""


class MultigranularDetectionModel(BaseDetectionModel):
    """
    Multigranular detection model, tries to detect classes by searching for them inside bounding boxes of possible parent classes
    """

    def __init__(self, model_config: dict=None, detection_model: BaseDetectionModel=None):
        super().__init__(model_config)
        self.detection_model = detection_model
        self.classes_hierarchy = {}

    def load_model(self):
        self.logger = Logger.get_logger(type(self).__name__, self.model_config.get("logging", None))
        self.detection_model.load_model()

    def _search_for_classes(self, image, classes_hierarchy):
        """
        Recursively search for classes with the detection model using cropped images of parent classes.

        Parameters
        ----------
        image : numpy.ndarray
            The image in which to search for classes.
        classes_hierarchy : dict
            A dictionary representing the hierarchy of classes to search for. 
            The keys are class names and the values are dictionaries representing sub-classes.

        Returns
        -------
        dict
            A dictionary where the keys are class names and the values are lists of tuples. 
            Each tuple contains a detection dictionary and a sub-classes detections dictionary.
            The detection dictionary contains information about the detected class, including the bounding box.
            If the class has sub-classes, the detection dictionary will also contain an "is_parent" key set to True.
            e.g. {"class_name": [(detection, detections_dict)]}
        """
        if len(classes_hierarchy) == 0:
            return {}

        classes = list(classes_hierarchy.keys())
        self.detection_model.update_classes(classes)

        detections = self.detection_model.run_inference(image)
        
        # debug
        if self.debug:
            if self.debug_info is None:
                self.debug_info = []

            elements = detections_to_elements(detections)
            overlay = draw_elements_detections(image.copy(), elements)
            plt.figure(figsize=(3*(len(elements)+1), 3))
            plt.subplot(1, len(elements)+1, 1)
            plt.imshow(overlay[:,:,::-1])
            plt.axis('off')
            plt.title(f"looking for {classes}")
            for d_idx, detection in enumerate(detections):
                bbox = detection["bbox"]
                cropped_image = image[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
                plt.subplot(1, len(elements)+1, d_idx+2)
                plt.imshow(cropped_image[:,:,::-1])
                plt.axis('off')
                plt.title(f"CROP of: {detection['class']} {detection['prob']:.2f}")

            # Update debugging info and display detections
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = np.array(Image.open(buf))[:,:,::-1] # Convert to RGB format
            self.debug_info.append({"role": "MultigranularDetectionModel", "image": img})
            if self.display_debug_info:
                plt.show()
            plt.close()

        detections_dict = {}
        for d_idx, detection in enumerate(detections):
            class_name = detection["class"]
            if class_name not in detections_dict:
                detections_dict[class_name] = []
            bbox = detection["bbox"]
            cropped_image = image[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
            
            # detect sub classes
            # TODO: handle bbox shared by multiple classes (e.g. class_name="cat cats")
            if class_name in classes_hierarchy:
                sub_classes_detections = self._search_for_classes(
                    cropped_image, classes_hierarchy[class_name]
                )
            else:
                sub_classes_detections = {}
            # add a tag to indicate if it is a parent class
            if classes_hierarchy.get(class_name,{}) != {}:
                detection["is_parent"] = True
            detections_dict[class_name].append((detection, sub_classes_detections))
        
        return detections_dict

    def nested_detections_to_list(self, nested_detections):
        detections = []
        for class_name, class_detections in nested_detections.items():
            for detection, sub_detections in class_detections:
                detections.append(detection)
                sub_detections_list = self.nested_detections_to_list(sub_detections)
                # update bbox coordinates of sub detections
                for sub_detection in sub_detections_list:
                    if "bbox" in sub_detection:
                        sub_detection["bbox"] = np.array(
                            [[
                                sub_detection["bbox"][0][0] + detection["bbox"][0][0],
                                sub_detection["bbox"][0][1] + detection["bbox"][0][1],
                            ],
                            [
                                sub_detection["bbox"][1][0] + detection["bbox"][0][0],
                                sub_detection["bbox"][1][1] + detection["bbox"][0][1],
                            ]]
                        )
                        
                detections += sub_detections_list
        return detections

    def run_inference(self, image):
        
        self.debug_info = None
        detections = self._search_for_classes(image, self.classes_hierarchy)

        detections = self.nested_detections_to_list(detections)
        # update ids
        for i, detection in enumerate(detections):
            detection["id"] = i
        return detections

    def _add_to_classes_hierarchy(
        self, sequence_of_classes: list[str], classes_dict: dict
    ):
        """
        recursively add classes to the classes hierarchy. building a tree structure of parent classes and their sub classes
        """
        if len(sequence_of_classes) == 0:
            return {}
        class_name = sequence_of_classes[0].strip()

        classes_dict[class_name] = self._add_to_classes_hierarchy(
            sequence_of_classes[1:], classes_dict.get(class_name, {})
        )

        return classes_dict

    def update_classes(self, classes: list[str]):
        self.classes_hierarchy = {}
        for class_sequence_str in classes:
            # sequence of classes that can encompass eachother e.g. "person.face.eyes"
            class_sequence = class_sequence_str.split(".")
            self.classes_hierarchy = self._add_to_classes_hierarchy(
                class_sequence, self.classes_hierarchy
            )

    def __getattr__(self, attr):
        """this method is a fallback option on any methods the original detection_model might support"""
        # using getattr ensures that both __getattribute__ and __getattr__ (fallback) get called
        # (see https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute)
        orig_attr = getattr(self.detection_model, attr)
        if callable(orig_attr):

            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if id(result) == id(self.detection_model):
                    return self
                return result

            return hooked
        else:
            return orig_attr




# =============================================================================
# Ground Truth Models
# =============================================================================


class LookUpDetectionModel(BaseDetectionModel):
    """
    A detection model that searches the list of object names to match a text input using semantic similarity.
    """

    def __init__(self, model_config=None):
        super().__init__(model_config)
        self.available_objects = []
        from sklearn.metrics.pairwise import cosine_similarity

        self.cosine_similarity = cosine_similarity
        self.similarity_threshold = self.model_config.get("similarity_threshold", 0.35)
        self.text_embedding_model = model_config.get("text_embedding_model", "clip-ViT-B-32")
        self.image_embedding_model = model_config.get("image_embedding_model", "clip-ViT-B-32")

        self.weight_text_similarity = self.model_config.get("weight_text_similarity", 0.5)
        self.weight_image_similarity = self.model_config.get("weight_image_similarity", 0.5)

        self.weight_text_similarity = self.weight_text_similarity / (self.weight_text_similarity + self.weight_image_similarity)
        self.weight_image_similarity = self.weight_image_similarity / (self.weight_text_similarity + self.weight_image_similarity)
    
    def load_model(self):
        from sentence_transformers import SentenceTransformer
        self.logger = Logger.get_logger(type(self).__name__, self.model_config.get("logging", None))
        self.text_model = SentenceTransformer(self.text_embedding_model)

        if self.image_embedding_model:
            self.image_model = SentenceTransformer(self.image_embedding_model)
        else:
            self.image_model = None

    def update_thresholds(self, thresholds):
        # self.model.set_thresholds(thresholds)

        self.logger.info("[LookupDetectionModel] Updating thresholds")
        self.similarity_threshold = thresholds.get(self.model_config["name"], {}).get(
            "similarity_threshold", self.similarity_threshold
        )

    def update_available_detections(self, available_detections):
        """
        Update the list of available detections and their states for similarity lookup.

        :param available_detections: List of dictionaries containing object names and states.
        """
        self.available_detections = available_detections

        

    def run_inference(self, image):
        """
        Perform similarity lookup to match text input with available detections.

        :param image: Input image (not used in this model).
        :return: List of detections with matched detections.
        """
        detections = []
        if not self.classes:
            return detections

        input_embeddings = self.text_model.encode(self.classes)

        self.object_text_embeddings = self.text_model.encode([obj['class'] for obj in self.available_detections])

        # compute the similarities between the input and the available objects
        text_similarities = self.cosine_similarity(input_embeddings, self.object_text_embeddings)
        if self.image_model:
            crops = []
            invalid_crops = False
            for obj in self.available_detections:
                if 'image' in obj:
                    crops.append(obj['image'])
                elif 'bbox' in obj:
                    # crop the image
                    bbox = obj['bbox']
                    crop = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
                    pil_crop = Image.fromarray(crop)
                    crops.append(pil_crop)
                else:
                    invalid_crops = True

            if not invalid_crops:
                # get the image embeddings
                image_embeddings = self.image_model.encode(crops)
                text_images_similarities = self.cosine_similarity(input_embeddings, image_embeddings)

                # combine the similarities from text and images
                similarities = text_similarities * self.weight_text_similarity + text_images_similarities * self.weight_image_similarity
            else:
                self.logger.warning("[LookUpDetectionModel] Some crops were invalid and will be ignored.")
                similarities = text_similarities
        else:
            similarities = text_similarities        

        # parse the results 
        for i, class_name in enumerate(self.classes):
            
            match_indices = np.where(similarities[i] > self.similarity_threshold)[0]
            for idx in match_indices:
                detection = {
                    "class": class_name,
                    "prob": similarities[i][idx],
                    "id": idx,
                    "bbox": self.available_detections[idx].get("bbox", None),
                }
                detections.append(detection)

        return detections

    def update_classes(self, classes: list[str]):
        self.classes = classes

    def reset(self):
        self.classes = []
        self.available_detections = []
        self.object_text_embeddings = []
