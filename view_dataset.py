import io
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset

def visualize_sample(item: dict, block: bool = True):
    # Unpack the item
    prompt = item.get('prompt', None)
    bbox = item.get('bbox', None)
    if bbox is not None:
        bbox = bbox.numpy()
    else:
        bbox = np.array([0, 0, 0, 0])  # Default bbox if not provided
    rgb = item['rgb']
    # Visualize the RGB image with the bbox and 
    rgb_np = rgb.numpy()
    # breakpoint()  # For debugging, you can inspect the rgb_np here
     # Format bgr to rgb
    rgb_np = rgb_np[:, :, ::-1]  # Convert BGR to RGB if needed
    # Show image with bounding box
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_np)
    if bbox is not None and bbox.shape == (4,):
        x_min, y_min, width, height = bbox
        plt.gca().add_patch(plt.Rectangle((x_min, y_min), width, height, fill=False, edgecolor='red', linewidth=2))
    plt.title(f"Prompt: {prompt}")
    plt.axis('off')
    plt.show(block=block)

class RefCOCODataset:
    def __init__(self, split="val"):
        self.ds = load_dataset("lmms-lab/RefCOCO", split=split)
        self.length = len(self.ds)
        
        # Seed
        self.seed = 42
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = self.ds[idx]
        # Load image from URL or bytes
        if "image" in sample:
            # HuggingFace datasets may store images as PIL.Image, bytes, or URLs
            img = sample["image"]
            if isinstance(img, Image.Image):
                image_np = np.asarray(img).astype(np.int16)  # Convert PIL Image to numpy array
                try:
                    image_np = image_np[:, :, ::-1].copy()  # Convert BGR if needed
                except:
                    return None # If conversion fails, return None
          
        else:
            raise KeyError("No 'image' key in dataset sample")
        prompt = sample.get("answer", [])
        # Choose a random prompt if multiple are available
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt = np.random.choice(prompt)
            prompt = str(prompt)
        elif isinstance(prompt, str):
            pass
        bbox = np.array(sample["bbox"], dtype=np.int16)  # [x1, y1, x2, y2]
        segmentation = sample.get("segmentation", None)
        # breakpoint()

        return {
            "rgb": torch.from_numpy(image_np),
            "prompt": prompt,
            "bbox": torch.from_numpy(bbox),
            "segmentation": segmentation,  # This can be None or a segmentation mask
        }


if __name__ == "__main__":
    # Example usage of the dataset and visualization
    split = "testB"  # Change to "train", "val", or "testB" as needed
    ds = RefCOCODataset(split=split)
    print(f"Loaded RefCOCO split {split} with {len(ds)} samples.")
    # sample = ds[0]
    # visualize_sample(sample)
    
    # If you want to visualize multiple samples, you can loop through them
    for i in range(len(ds)):  # Visualize first 5 samples
        sample = ds[i]
        # visualize_sample(sample)