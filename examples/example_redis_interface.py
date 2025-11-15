from matplotlib import pyplot as plt
import cv2
import numpy as np
from open_perception.communication.redis_client import RedisClient
from open_perception.utils.config_loader import load_config

# load redis config from default config file
config = load_config("config/redis_interface.yaml") # loads form config/default.yaml
redis_config = config["communication"]["redis"]

client = RedisClient(config=redis_config)
client.connect()
# client.reset()


image_file = "examples/images/cat_and_dog.png"
# load image as np array
image = cv2.imread(image_file)
# plt.imshow(image[:,:,::-1]); plt.axis('off'); plt.show()


client.send_frame("front",rgb=image, fake_pc=True, wait=True)
elements = client.locate(["dog.eyes", "cat"], wait=True)
# elements = client.get_detection_elements()

masked_image = client.generate_masked_frame(elements, image)
plt.imshow(masked_image[:,:,::-1]); plt.axis('off'); plt.show()
client.disconnect()
