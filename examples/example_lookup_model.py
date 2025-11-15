"""
This example shows how to use the redis interface to forward detections made by a fixed vocabulary model to the redis interface and look for the detections that match a given open vocabulary query.
"""


from matplotlib import pyplot as plt
import cv2
import numpy as np
import time

from open_perception.pipeline.element import Element
from open_perception.communication.redis_client import RedisClient
from open_perception.utils.config_loader import load_config
from ultralytics import YOLO

# Load a detection model with fixed vocabulary
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model
image = cv2.imread("bus.jpg")
result = model(image)[0]  # return a list of Results objects

# result.show()  # display to screen
# result.save_dir = None

classes = [result.names[int(c)] for c in result.boxes.cls.cpu()]
probs = result.boxes.conf.cpu().numpy()
bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
detections = [{"id": id, "bbox": [[b[0], b[1]], [b[2], b[3]]], "class": n, "prob": p} for id, (b, n, p) in enumerate(zip(bboxes, classes, probs))]

from open_perception.utils.visualization import draw_elements_detections
from open_perception.utils.common import detections_to_elements


original_detections = draw_elements_detections(image.copy(), detections_to_elements(detections, image))
plt.imshow(original_detections[:,:,::-1]); plt.axis('off'); plt.show(block=False)

# Forward the detections made by the fixed vocabulary model to the redis interface and look for the detections that match a given open vocabulary query
meta = {"detections": detections, "stamp": time.time(), "id": 1}

# load redis config from default config file
config = load_config("config/redis_interface_lookup.yaml") # loads overwrites to the config/default.yaml
redis_config = config["communication"]["redis"]

client = RedisClient(config=redis_config)
client.connect()
# client.reset()

# send frame and available detections to redis
client.send_frame("front", rgb=image, fake_pc=True, wait=True, meta=meta)
elements = client.locate(["vehicle", "white jacket"], wait=True)

masked_image = client.generate_masked_frame(elements, image)
client.disconnect()
plt.imshow(masked_image[:,:,::-1]); plt.axis('off'); plt.show()
