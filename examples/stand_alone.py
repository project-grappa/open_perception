from matplotlib import pyplot as plt
import cv2

from open_perception.perception.detection_models import (
    GroundingDinoModelHF,
    YoloWorldModel,
)
from open_perception.utils.visualization import draw_elements_detections
from open_perception.utils.common import detections_to_elements

image_file = "examples/images/cat_and_dog.png"
# load image as np array
image = cv2.imread(image_file)
# plt.imshow(image[:,:,::-1]); plt.axis('off'); plt.show()


model = GroundingDinoModelHF(model_config={})
model.load_model()
model.update_classes(["dog"])
detections = model.run_inference(image)


elements = detections_to_elements(detections)
image = draw_elements_detections(image, elements)
plt.imshow(image[:,:,::-1]); plt.axis('off'); plt.show()
