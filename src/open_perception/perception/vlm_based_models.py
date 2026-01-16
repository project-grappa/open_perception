"""
detection_modules.py
This module contains example implementations of detection models that leverage VLMs for object detection.

Classes:
    - VLMDetectionWithSearch: A class that uses a Vision Language Model (VLM) and a pre-trained object detection model to perform object detection.
    - VLMVerifier: A class that uses a Vision Language Model (VLM) to disambiguate between different objects detected in an image.

For Examples check examples/vlm_search.ipynb and examples/vlm_deambiguation.ipynb
"""

import os
import numpy as np
import json
from typing import List, Dict, Any, Optional
from open_perception.perception.base_models import BaseDetectionModel
from open_perception.logging.base_logger import Logger
import torch
import litellm
import base64
import cv2
import matplotlib.pyplot as plt

from open_perception.utils.common import combine_images_grid
from open_perception.utils.visualization import draw_elements_detections, COLOR_TO_TEXT
from open_perception.utils.common import detections_to_elements, recursive_filter_keys
from open_perception.utils.vlm import (
    encode_image,
    decode_image,
    parse_vlm_messages,
    cluster_by_similarity,
)


class VLMBaseDetectionModel(BaseDetectionModel):
    """
    VLMBaseDetectionModel is an abstract base class for detection models that use Vision Language Models (VLMs).
    It provides a common interface for VLM-based detection models.
    """

    detection_model: Optional[BaseDetectionModel] = None
    vlm_config: Optional[dict] = {}

    def load_model(self):
        self.logger = Logger.get_logger(__name__, self.vlm_config.get("logging", None))
        return self.detection_model.load_model()

    def update_last_messages(self, new_messages: List[dict] | dict):
        """
        Update the last messages list with new messages, and if debugging is enabled, parse the messages for debug information.
        """
        if not isinstance(new_messages, list):
            messages = [new_messages]
        else:
            messages = new_messages
        self.last_messages.extend(messages)

        if self.debug:
            parsed_msg = parse_vlm_messages(
                messages, display=self.display_debug_info, with_color=False
            )
            if self.debug_info is None:
                self.debug_info = []
            self.debug_info.extend(parsed_msg)
        self.update_debug_info()

    def update_debug_info(self):
        """
        Update the debug information by extending it with the detection model's debug info if available.
        """
        if self.debug and self.detection_model.debug_info is not None:
            if self.debug_info is None:
                self.debug_info = []
            # update the debug info with the detection model's debug info
            self.debug_info.extend(self.detection_model.debug_info)
            self.detection_model.debug_info = None

    def update_classes(self, classes: list[str]):
        """
        Update the classes for the detection model.
        Args:
            classes (list[str]): List of class names to update.
        """
        self.logger.info(f"Updating classes: {classes}")
        self.detection_model.update_classes(classes)
        self.classes = classes

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


class VLMDetectionWithSearch(VLMBaseDetectionModel):
    """
    VLMDetectionWithSearch is a class that uses a Vision Language Model (VLM) and a pre-trained object detection model to perform object detection.
    It allows for checking the presence of specified objects in an image and can be configured with various parameters.
    """

    tools = [
        {
            "type": "function",
            "function": {
                "name": "in_the_image",
                "description": "Check if the specified objects are in the image",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "The name of the object to be located in the image.",
                        },
                        "parent_object_name": {
                            "type": "string",
                            "description": "The parent object to which could encompass the object_name. If not specified, the object_name is considered a top-level object.",
                        },
                    },
                    "required": ["object_name"],
                },
            },
        }
    ]

    def __init__(
        self,
        vlm_config: dict,
        detection_model: Optional[BaseDetectionModel] = None,
    ):
        super().__init__(vlm_config)
        self.detection_model = detection_model
        self.vlm_config = vlm_config

        # Parse VLM search config
        self.max_attempts = vlm_config.get("max_attempts", 3)
        self.max_depth = vlm_config.get("max_depth", 2)
        self.use_image = vlm_config.get("use_image", True)
        self.max_image_size = vlm_config.get("max_image_size", [640, 640])
        self.use_only_on_failures = vlm_config.get("use_only_on_failures", False)

        self.system_prompt = vlm_config.get("prompts", {}).get("system_message", "")
        if not self.system_prompt:
            raise ValueError("System prompt is required in the VLM search config.")
        self.user_message = vlm_config.get("prompts", {}).get("user_message", "")
        if not self.user_message:
            raise ValueError("User message is required in the VLM search config.")
        self.generation_error_response = {
            "role": "assistant",
            "content": vlm_config.get("prompts", {}).get(
                "generation_error_response", ""
            ),
        }

        self.litellm_completion_kwargs = {
            "seed": 42,  # for reproducibility
            "model": vlm_config.get("model", "gpt-4o"),
            "max_tokens": vlm_config.get("max_tokens", 4096),
            "temperature": vlm_config.get("temperature", 0.0),
            "timeout": vlm_config.get("timeout", 120),
            "tools": self.tools,
            "tool_choice": "auto",
            "logprobs": True,
            # "response_format": { "type": "json_object" },
        }

        self.available_functions = {
            "in_the_image": self._in_the_image,
        }

        self.last_messages = []

    def _build_messages(self, image: np.ndarray, classes: list[str]) -> list[dict]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.user_message.format() + str(classes)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encode_image(image)}",
                        },
                    },
                ]
                if self.use_image
                else [],
            },
        ]
        return messages

    def _in_the_image(self, objects: list[str], image: np.ndarray) -> list[dict]:
        """
        Check if the specified objects are present in the image using the detection model.

        Args:
            objects (list[str]): List of objects to check for presence in the image.

        Returns:
            list[dict]: Results of the object detection.
        """
        self.logger.info(f"Checking if objects {objects} are in the image...")
        self.detection_model.update_classes(objects)
        detections = self.detection_model.run_inference(image)

        return detections

    def _call_functions(self, tool_calls: list[dict], image: np.ndarray) -> list[dict]:
        """
        Call the functions specified in the tool calls.

        Args:
            tool_calls (list[dict]): List of tool calls to execute.

        Returns:
            list[dict]: Results of the function calls.
        """
        functions_responses = []

        # combine the function calls to perform a single query to the detection model
        func_id_to_class_name = {}
        all_class_names = []

        for tool_call in tool_calls:
            function_name = tool_call.function.name

            if function_name in self.available_functions.keys():
                function_args = json.loads(tool_call.function.arguments)
                class_name = function_args.get("object_name", None)
                if class_name is None:
                    # register error in the format of the function call
                    functions_responses.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": f"ERROR: Function {function_name} requires an 'object' argument.",
                        }
                    )
                    continue
                if "parent_object_name" in function_args:
                    class_name = function_args["parent_object_name"] + "." + class_name

                func_id_to_class_name[tool_call.id] = class_name
                all_class_names.append(class_name)

            else:  # wrong function call
                self.logger.warning(
                    f"Function {function_name} not found in available functions."
                )
                functions_responses.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"ERROR: Function {function_name} not found in available functions. The available functions are: {list(self.available_functions.keys())}. Please update the function calls.",
                    }
                )

        self.logger.info(
            f"getting detections for {len(all_class_names)} classes: {all_class_names}"
        )

        # query the detection model with the class names
        detections = self._in_the_image(all_class_names, image)

        # debug: print the detections
        self.logger.debug(f"Detections: {detections}")
        self.logger.debug(f"Function Calls: {tool_calls}")

        # parse the detections to each function call
        for func_id, class_name in func_id_to_class_name.items():
            content = "Detection results: " + ", ".join(
                [f"{d['class']}: True" for d in detections]
            )
            # if class_name not in [d["class"] for d in detections]:
            #     if len(detections) > 0:
            #         content += ","
            #     content += f" {class_name}: False"

            functions_responses.append(
                {
                    "tool_call_id": func_id,
                    "role": "tool",
                    "name": "in_the_image",
                    "content": content,
                }
            )

        self.logger.info(
            f"Elements Found: {', '.join([d['class'] for d in detections])}"
        )

        return functions_responses, detections

    def _parse_response(self, response: str, detections: list[dict]) -> list[dict]:
        """
        Parses the response from the VLM and returns the detections.

        Args:
            response (str): The response from the VLM.
            detections (list[dict]): The list of detections to update.

        Returns:
            list[dict]: The updated list of detections.
        """
        try:  # parse the response as JSON
            # Crop the last segment of text that starts with ```json and ends with ```
            json_text = response
            start_idx = json_text.rfind("```json")
            end_idx = json_text.rfind("```")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = json_text[start_idx + 7 : end_idx].strip()
                json_text = (
                    json_text.replace("\n", "")
                    .replace("False", "false")
                    .replace("True", "true")
                )  # remove newlines and replace Python booleans with JSON booleans
            response_data = json.loads(json_text)  # TODO handle json

        except json.JSONDecodeError:
            self.logger.error(f"Failed to parse response as JSON: {response}")

        return detections
        # response_data = json.loads(response)

    def predict(self, image: np.ndarray) -> list[dict]:
        """
        Performs object detection and open-vocabulary perception on the input image.
        This method first uses a detection model to identify objects from a predefined set of classes in the provided image.
        If no classes are specified, a warning is logged and an empty list is returned.
        If objects are detected, it builds a message for a Vision-Language Model (VLM), sends the message for further analysis,
        and checks if the VLM requests any function calls for additional processing.
        Args:
            image (np.ndarray): The input image as a NumPy array.
        Returns:
            list[dict]: A list of detection results, where each result is a dictionary containing information about a detected object.
                        Returns an empty list if no classes are specified or no objects are detected.
        """

        self.logger.info("Running VLMDetectionWithSearch...")

        if not self.classes:
            self.logger.warning(
                "No classes specified for detection. Please update the classes before running inference."
            )
            return []

        all_detections = []

        # step 0: check if the objects can already be detected by the detection model
        if self.use_only_on_failures:
            self.detection_model.update_classes(self.classes)
            detections = self.detection_model.run_inference(image)
            all_detections.extend(detections)
            self.update_debug_info()

            if not detections:
                self.logger.warning("No objects detected in the image.")

            missing_classes = [
                c for c in self.classes if c not in [d["class"] for d in detections]
            ]
            if not missing_classes:
                self.logger.info(
                    "All classes are already detected by the detection model."
                )
                return detections
        else:
            # if use_only_on_failures is False, we assume that we want to use the VLM for all classes
            missing_classes = self.classes
            detections = []

        self.logger.info(f"Missing classes: {missing_classes}")

        # Step 1: build the messages for the VLM
        messages = self._build_messages(image, missing_classes)
        self.update_last_messages(messages)

        for attempt in range(self.max_attempts):
            self.logger.info(
                f"Attempt {attempt + 1} of {self.max_attempts} to call the VLM."
            )

            # Step 2: call the VLM with the messages
            response = litellm.completion(
                messages=self.last_messages, **self.litellm_completion_kwargs
            )["choices"][0]

            self.update_last_messages(response.message)

            tool_calls = response.message.tool_calls
            # Step 3: check if the model wanted to call a function
            if tool_calls:
                functions_responses, detections = self._call_functions(
                    tool_calls, image
                )

                all_detections.extend(detections)
                self.update_last_messages(functions_responses)

                # debug
                if self.debug:
                    elements = detections_to_elements(detections)
                    overlay = draw_elements_detections(image.copy(), elements)
                    plt.figure(figsize=(4, 4))
                    plt.imshow(overlay[:, :, ::-1])
                    plt.title(f"Detections - Attempt {attempt + 1} ")
                    plt.axis("off")
                    if self.display_debug_info:
                        plt.show()
                    else:
                        plt.close()

            elif response.message.get("content") is not None:
                # If no tool calls, but the model provided a response
                self.update_last_messages(
                    {"role": "assistant", "content": response.message["content"]}
                )
                message_content = response.message["content"]
                all_detections = self._parse_response(message_content, all_detections)
                break

            else:
                self.update_last_messages(self.generation_error_response)

        self.last_messages = messages
        return all_detections

    def run_inference(self, image) -> list[dict]:
        """
        Run inference on the input image using the VLM and the pre-trained object detection model.

        Args:
            image (np.ndarray): The input image for object detection.

        Returns:
            list[dict]: A list of detected objects with their bounding boxes and labels.
        """
        self.logger.info("Running inference...")
        return self.predict(image)


class VLMDescriptionModel(VLMBaseDetectionModel):
    """
    VLMDescriptionModel is an abstract base class for models that use Vision Language Models (VLMs) to generate descriptions of images.
    It provides a common interface for VLM-based description models.
    """

    def __init__(self, vlm_config: dict, detection_model: BaseDetectionModel):
        super().__init__(vlm_config)
        self.vlm_config = vlm_config
        self.detection_model = detection_model

        self.last_messages = []
        self.system_prompt = vlm_config.get("prompts", {}).get("system_message", "")
        if not self.system_prompt:
            raise ValueError("System prompt is required in the VLM config.")
        self.user_message = vlm_config.get("prompts", {}).get("user_message", "")
        if not self.user_message:
            raise ValueError("User message is required in the VLM config.")

        self.max_images_per_message = vlm_config.get("max_images_per_message", 10)
        self.max_image_size = vlm_config.get("max_image_size", [640, 640])

        self.litellm_completion_kwargs = {
            "seed": 42,  # for reproducibility
            "model": vlm_config.get("model", "gpt-4o"),
            "max_tokens": vlm_config.get("max_tokens", 4096),
            "temperature": vlm_config.get("temperature", 0.0),
            "timeout": vlm_config.get("timeout", 120),
            "logprobs": True,
            "response_format": {"type": "json_object"},
        }

    def _combine_crops_into_a_single_img(  # TODO: adapt to numpy only using cv2
        self, image_crops: list[np.ndarray]
    ) -> np.ndarray:
        """
        Combines a list of image crops into a single image by stacking them horizontally.
        Args:
            image_crops (list[np.ndarray]): List of cropped images as NumPy arrays.
        Returns:
            np.ndarray: A single image containing all the crops stacked horizontally.
        """
        assert len(image_crops) > 0, "No image crops provided to combine."

        grid_of_images = combine_images_grid(
            image_crops, ncols=self.max_images_per_message, max_size=self.max_image_size
        )

        return grid_of_images

    def _get_prompt_images_and_labels(
        self, image: np.ndarray, detections: list[dict]
    ) -> list[np.ndarray]:
        """
        Returns a list of images to be used in the prompt for the VLM. Each image containing crop of each detection.
        Args:
            detections (list[dict]): List of detections, each containing 'bbox', 'class', 'prob', and 'id'.
            image (np.ndarray): The original image as a NumPy array.
        Returns:
            list[np.ndarray]: A list of cropped images for each detection.
            list[str]: A list of index labels for each detection.
        """
        prompt_images = []
        prompt_labels = []
        for i, det in enumerate(detections):
            bbox = det.get("bbox", [])
            if bbox is not None:
                crop = image[bbox[0][1] : bbox[1][1], bbox[0][0] : bbox[1][0]]
                prompt_images.append(crop)
                prompt_labels.append(str(i))

        prompt_images = [np.array(self._combine_crops_into_a_single_img(prompt_images))]
        return prompt_images, prompt_labels

    def _build_messages(
        self, images: list[np.ndarray], classes: list[str]
    ) -> list[dict]:
        """
        Builds a message for the VLM with the provided image and classes.

        Args:
            image (np.ndarray): The input image as a NumPy array.

            classes (list[str]): List of classes to include in the message.

        Returns:
            list[dict]: A list of dictionaries representing the message for the VLM.
        """
        images_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(img)}",
                },
            }
            for img in images
        ]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": images_content
                + [
                    {
                        "type": "text",
                        "text": self.user_message.format(classes=classes),
                    },
                ],
            },
        ]
        return messages

    def _parse_response(self, response, detections: list[dict]) -> list[dict]:
        """
        Parses the response from the VLM and updates the detections description
        """
        parsed_response = json.loads(response)
        # Update the detections with the parsed response
        for k, v in parsed_response.items():
            idx = int(k)
            if idx < len(detections):
                # Update the detection with the new description
                detections[idx]["description"] = v
        return detections

    def run_inference(
        self, image: np.ndarray, precomputed_detections: Optional[list[dict]] = None
    ) -> str:
        # step 0: get the detections from the base detection model
        if precomputed_detections is None:
            detections = self.detection_model.run_inference(image)
            self.update_debug_info()
        else:
            detections = precomputed_detections

        if not detections:
            self.logger.info("No detections found in the image.")
            return []

        for i in range(0, len(detections), self.max_images_per_message):
            # slice the detections to fit in the message
            chunk_size = min(self.max_images_per_message, len(detections) - i)
            detections_slice = detections[i : i + chunk_size]
            if not detections_slice:
                continue

            # get the classes from the detections
            prompt_images, prompt_labels = self._get_prompt_images_and_labels(
                image, detections_slice
            )

            # step 1: build the messages for the VLM
            messages = self._build_messages(prompt_images, prompt_labels)
            self.update_last_messages(messages)

            # step 2: call the VLM with the messages
            response = litellm.completion(
                messages=messages, **self.litellm_completion_kwargs
            )["choices"][0]
            self.update_last_messages(response.message)

            # step 3: parse the response
            detections_slice = self._parse_response(
                response.message.get("content", ""), detections_slice
            )

            # update the detections with the parsed response
            detections[i : i + chunk_size] = detections_slice

        return detections


class VLMVerifier(VLMBaseDetectionModel):
    """
    VLMVerifier is a class that uses a Vision Language Model (VLM) to disambiguate between different objects in an image.
    It allows for checking the presence of specified objects in an image and can be configured with various parameters.
    """

    def __init__(
        self,
        vlm_config: dict,
        detection_model: Optional[BaseDetectionModel] = None,
    ):
        super().__init__()
        self.detection_model = detection_model
        self.vlm_config = vlm_config

        # Parse VLM config
        self.system_prompt = vlm_config.get("prompts", {}).get("system_message", "")
        if not self.system_prompt:
            raise ValueError("System prompt is required in the VLM config.")
        self.user_message = vlm_config.get("prompts", {}).get("user_message", "")
        if not self.user_message:
            raise ValueError("User message is required in the VLM config.")

        self.use_image = vlm_config.get("use_image", True)
        self.use_descriptions = vlm_config.get("use_descriptions", True)

        self.verify_single_detection = vlm_config.get("verify_single_detection", True)

        self.litellm_completion_kwargs = {
            "seed": 42,  # for reproducibility
            "model": vlm_config.get("model", "gpt-4o"),
            "max_tokens": vlm_config.get("max_tokens", 4096),
            "temperature": vlm_config.get("temperature", 0.0),
            "timeout": vlm_config.get("timeout", 120),
            "logprobs": True,
            "response_format": {"type": "json_object"},
        }
        self.last_messages = []

    def _build_messages(
        self,
        original_image: np.ndarray,
        masked_image: np.ndarray,
        classes: list[str],
        query: str,
        bboxes: list[list[int, int], list[int, int]],
        descriptions: Optional[list[str]] = None,
        colors: Optional[list[tuple[int, int, int]]] = None,
    ) -> list[dict]:
        """
        Builds a message for the VLM with the provided image and classes.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            classes (list[str]): List of classes to include in the message.
            query (str): The query string to include in the message.

        Returns:
            list[dict]: A list of dictionaries representing the message for the VLM.
        """

        classes_bboxes_colors = "\n".join(
            [f"{c}: bbox: {bbox}, color: {color}" for c, bbox, color in zip(classes, bboxes, colors)]
        )
        text_content = self.user_message.format(
            query=query,
            classes=classes,
            classes_bboxes_colors=classes_bboxes_colors,
        )
        if self.use_descriptions and descriptions:
            # if descriptions are provided, add them to the text content
            text_content += "\n".join(
                [f"{c}: {d}" for c, d in zip(classes, descriptions)]
            )

        image_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(original_image)}",
                },
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(masked_image)}",
                },
            },
        ]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": image_content + [
                    {
                        "type": "text",
                        "text": text_content,
                    },
                ],
            },
        ]
        return messages

    def _get_prompt_info(self, image: np.ndarray, detections: list[dict]) -> np.ndarray:
        """Returns a masked image with labels with the image anotation"""
        detection_elements = detections_to_elements(detections)

        # rename labels to be unique
        detection_labels = []
        bboxes = []
        descriptions = []
        colors = [COLOR_TO_TEXT.get(tuple(el.color), f"RGB={tuple(el.color)}") for el in detection_elements]
        for i, el in enumerate(detection_elements):
            el.class_name = f"{i}"
            el.detection_prob = None
            detection_labels.append(el.class_name)
            bboxes.append(el.bbox)
            descriptions.append(el.description)

        masked_image = draw_elements_detections(image.copy(), detection_elements, big_labels=True)


        return masked_image, detection_labels, bboxes, descriptions, colors

    def _parse_vlm_resonse(self, response, detection_list):
        """
        Parses the response from the VLM and returns the selected detections.

        Args:
            response: The response from the VLM.
            detection_list: The list of detections to select from.

        Returns:
            list[dict]: The selected detections based on the VLM response.
        """
        selected_detections = []
        for item in response.get("choices", []):
            if item.get("finish_reason") == "stop":
                # convert text to list
                response = json.loads(item.get("message", {}).get("content", "[]"))
                selected_detections.extend(
                    [detection_list[int(i)] for i in response.get("detections", [])]
                )
        return selected_detections

    def _get_detection_clusters(self, detections: list[dict]) -> Dict[str, list[dict]]:
        """
        Clusters detections by their class names.

        Args:
            detections (list[dict]): List of detections to cluster.

        Returns:
            Dict[str, list[dict]]: A dictionary where keys are class names and values are lists of detections.
        """
        if len(detections) == 0:
            return {c for c in self.classes}

        if len(self.classes) == 1:
            # if there is only one detection, we can return it as a single cluster
            return {detections[0]["class"]: detections}

        # cluster detections by similarity with the classes names and the detection names
        clustered_entries, similarities, clustered_indices = cluster_by_similarity(
            entries=[d["class"] for d in detections], clusters=self.classes
        )
        detections_clustered = {
            k: [detections[i] for i in indices]
            for k, indices in clustered_indices.items()
        }

        return detections_clustered

    def run_inference(
        self, image, precomputed_detections: Optional[list[dict]] = None
    ) -> list[dict]:
        self.last_messages = []

        # step 0: get the detections from the detection model
        if precomputed_detections is None:
            detections = self.detection_model.run_inference(image)
            self.update_debug_info()
        else:
            detections = precomputed_detections

        if not detections:
            self.logger.info("No detections found in the image.")
            return []

        detections_clustered = self._get_detection_clusters(detections)
        final_detections = []

        for original_label, detection_list in detections_clustered.items():
            # for query in self.classes:
            if len(detection_list) == 1 and not self.verify_single_detection:
                # if there is only one detection of this label, no deambiguation is needed. We can skip the VLM
                final_detections.extend(detection_list)
                continue

            # step 2: mask the image with the detections
            masked_image, detection_labels, bboxes, descriptions, colors = self._get_prompt_info(
                image, detection_list
            )

            # step 3: call the VLM with the messages
            messages = self._build_messages(
                original_image=image,
                masked_image=masked_image,
                classes=detection_labels,
                query=str(self.classes[0]), # TODO: fix this for multiple classes
                bboxes=bboxes,
                descriptions=descriptions,
                colors=colors
            )
            response = litellm.completion(
                messages=messages, **self.litellm_completion_kwargs
            )

            # filter use the vlm response to filter the detections
            selected_detections = self._parse_vlm_resonse(response, detection_list)
            final_detections.extend(selected_detections)

            # update last messages for debugging purposes
            self.update_last_messages(messages)
            self.update_last_messages(
                {
                    "role": "assistant",
                    "content": response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", ""),
                }
            )
        return final_detections


if __name__ == "__main__":
    # test image encoding and decoding
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    encoded_image = encode_image(test_image)
    decoded_image = decode_image(encoded_image)
    assert np.array_equal(test_image, decoded_image), (
        "Encoded and decoded images do not match!"
    )
    print("Image encoding and decoding works correctly!")

    # test if image is encoded correctly for the llm by loading a known image and querying the model for a description
    test_image = cv2.imread(
        "~/Desktop/CMU/research/motorcortex/motor_cortex/motor_cortex/perception/open_vocab_perception_pipeline/tests/cat_and_dog.png"
    )
    if test_image is None:
        raise ValueError("Test image not found. Please provide a valid image path.")
    encoded_image = encode_image(test_image)
    response = litellm.completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that describes images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the image."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}",
                        },
                    },
                ],
            },
        ],
        model="gpt-4o",
        max_tokens=100,
        temperature=0.0,
    )
    print(response)
    print("Image encoding for LLM works correctly!")
