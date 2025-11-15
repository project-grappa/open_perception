import os
import numpy as np
import json
from typing import List, Dict, Any, Optional
import base64
import cv2
import matplotlib.pyplot as plt


# Function to encode the image
def encode_image(image: np.ndarray) -> str:
    encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    _, buffer = cv2.imencode(".png", image, encode_params)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64


def decode_image(b64: str) -> np.ndarray:
    if ";base64," in b64:
        b64 = b64.split(";base64,")[-1]
    decoded = base64.b64decode(b64)
    nparr = np.frombuffer(decoded, np.uint8)
    bgr_img = cv2.imdecode(nparr, flags=cv2.IMREAD_COLOR)
    return bgr_img


# ====================== VLM ======================

from open_perception.logging.pdf_logger import PDFReport


def vlm_messages_to_pdf(
    messages: list[dict], pdf_file_path: str = None, title: Optional[str] = None
) -> PDFReport:
    parsed_messages = parse_vlm_messages(messages, display=False, with_color=False)
    pdf = debug_info_to_pdf(
        parsed_messages, pdf_file_path=pdf_file_path, title=title
    )
    return pdf

def debug_info_to_pdf(
    parsed_messages: List[Dict[str, Any]],
    pdf_file_path: str = None,
    title: Optional[str] = None,
) -> PDFReport:
    """
    Convert debug_info messages to a PDF report. This function iterates through the messages and adds them to a PDF report.
    It handles images encoded in base64 format, displaying them inline.

    Args:
        messages (list[dict]): List of messages from the VLM.
        pdf_file_path (str): Path to save the generated PDF report.
        title (str, optional): Title of the PDF report. If not provided, a default title will be used.

    Returns:
        PDFReport
    """

    pdf = PDFReport(title)
    pdf.add_page()

    for message in parsed_messages:
        content = message.get("text", "")
        image = message.get("image", None)
        color = message.get("color", None)

        if image is not None:
            # Display image
            pdf.add_image_from_array(image)
        elif content:
            # Display text
            if color:
                pdf.add_text(content, color=color)
            else:
                pdf.add_text(content)

    # save the PDF report to the specified file path
    if pdf_file_path is not None:
        pdf.output(pdf_file_path)
        print(f"PDF report saved to {pdf_file_path}")
    return pdf


def parse_vlm_messages(messages: list[dict], display=True, with_color=True) -> None:
    """
    Display VLM messages in a readable format. display images from base64 using plt interleaved with text messages.
    This function iterates through the messages and prints them with appropriate colors based on the role.
    It also handles images encoded in base64 format, displaying them inline.

    Args:
        messages (list[dict]): List of messages from the VLM.
    """
    role_colors = {
        "system": (64, 64, 64),  # Dark gray
        "user": (0, 102, 204),  # Blue
        "assistant": (0, 100, 0),  # Dark green
        "tool": (204, 153, 0),  # Dark yellow
        "function": (128, 0, 128),  # Purple
        "function_output": (255, 140, 0),  # Orange
    }
    parsed_msgs = []

    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role not in role_colors:
            role_color = None
        else:
            role_color = role_colors[role]

        # handle content types it could be list[dicts], list[str], str. if dict filter all types="image_url" from the print and instead decode them and display them as images
        if not isinstance(content, list):
            content = [content]

        for item in content:
            text = ""
            if isinstance(item, dict):
                if item.get("type") == "image_url":
                    # Decode base64 image

                    img_encoded = item.get("image_url", {}).get("url", "")
                    if img_encoded:
                        img = decode_image(img_encoded)
                        if display:
                            plt.figure(figsize=(4, 4))
                            plt.imshow(img[:, :, ::-1])
                            plt.axis("off")
                            plt.show()
                        parsed_msgs.append(
                            {"role": role, "image": img, "color": role_color}
                        )

                elif item.get("type") == "text":
                    text = item.get("text", "")
            elif isinstance(item, str):
                text = item
            else:
                text = str(item)

            if text:
                parsed_msgs.append(
                    {"role": role, "text": f"{role}: {text}", "color": role_color}
                )
                # Colorize text output using ANSI escape codes
                if with_color and role_color is not None:
                    r, g, b = role_color
                    text = f"\033[38;2;{r};{g};{b}m{text}\033[0m"
                if display:
                    print(f"{role}: {text}")

        # diplay tool calls
        if message.get("tool_calls"):
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name", "unknown")
                tool_args = tool_call.get("function", {}).get("arguments", "{}")
                tool_call_parsed = (
                    f"CALLING THE TOOL: {tool_name} \t with args: {tool_args}"
                )
                text = f"{tool_call_parsed}"
                parsed_msgs.append(
                    {"role": role, "text": f"{role}: {text}", "color": role_color}
                )
                if with_color and role_color is not None:
                    r, g, b = role_color
                    text = f"\033[38;2;{r};{g};{b}m{text}\033[0m"
                if display:
                    print(f"{role}: {text}")

    if not display:
        return parsed_msgs


from PIL import Image
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import torch


def cluster_by_similarity(
    entries: list[str | np.ndarray | Image.Image],
    clusters: list[str | np.ndarray | Image.Image],
    encoder: str = "openai/clip-vit-base-patch32",
) -> dict[str, list[str]]:
    """
    Cluster input texts based on their similarity to predefined clusters.

    Args:
        entries (list[str|np.ndarray|Image.Image]): List of input texts or images to be clustered.
        clusters (list[str]): List of predefined cluster names.
        encoder (Any, optional): Encoder function to compute similarity scores. If None, a clip text and image encoder is used.

    Returns:
        dict[str, list[str]]: Dictionary mapping cluster names to lists of input texts.
    """
    clustered_entries = {cluster: [] for cluster in clusters}
    clustered_indices = {cluster: [] for cluster in clusters}

    input_type = type(entries[0])
    assert all(isinstance(entry, input_type) for entry in entries), (
        "All entries must be of the same type"
    )

    cluster_type = type(clusters[0])
    assert all(isinstance(cluster, cluster_type) for cluster in clusters), (
        "All clusters must be of the same type"
    )

    model = CLIPModel.from_pretrained(encoder)
    tokenizer = AutoTokenizer.from_pretrained(encoder)

    processor = AutoProcessor.from_pretrained(encoder)

    # encode entries
    if input_type == str:
        # If entries are strings, tokenize them
        entries_inputs = tokenizer(entries, padding=True, return_tensors="pt")
        entries_features = model.get_text_features(**entries_inputs)
    elif input_type in [np.ndarray, Image.Image]:
        # If entries are images, process them
        if isinstance(entries[0], Image.Image):
            entries = [np.array(entry) for entry in entries]
        images = [
            Image.fromarray(entry) if isinstance(entry, np.ndarray) else entry
            for entry in entries
        ]
        entries_inputs = processor(images=images, return_tensors="pt")
        entries_features = model.get_image_features(**entries_inputs)
    else:
        raise ValueError(
            "Unsupported input type. Only str, np.ndarray, and Image.Image are supported."
        )

    with torch.no_grad():
        # encode clusters
        if cluster_type == str:
            # If clusters are strings, tokenize them
            clusters_inputs = tokenizer(clusters, padding=True, return_tensors="pt")
            clusters_features = model.get_text_features(**clusters_inputs).detach()
        elif cluster_type in [np.ndarray, Image.Image]:
            # If clusters are images, process them
            if cluster_type == Image.Image:
                images = clusters
            elif cluster_type == np.ndarray:
                images = [Image.fromarray(cluster) for cluster in clusters]

            clusters_inputs = processor(images=images, return_tensors="pt")
            clusters_features = model.get_image_features(**clusters_inputs).detach()
        else:
            raise ValueError(
                "Unsupported cluster type. Only str, np.ndarray, and Image.Image are supported."
            )

        # Compute similarity scores
        similarity_scores = entries_features @ clusters_features.T
        # normalize similarity scores
        similarity_scores = (
            similarity_scores
            / (
                entries_features.norm(dim=1, keepdim=True)
                @ clusters_features.norm(dim=1, keepdim=True).T
            )
            .cpu()
            .numpy()
        )

    clustered_indices = {}
    # Cluster entries based on the higher similarity scores with clusters
    for i, entry in enumerate(entries):
        # Find the index of the cluster with the highest similarity score
        best_cluster_index = np.argmax(similarity_scores[i])
        best_cluster = clusters[best_cluster_index]
        clustered_entries[best_cluster].append(entry)
        clustered_indices.setdefault(best_cluster, []).append(i)

    return clustered_entries, similarity_scores, clustered_indices


if __name__ == "__main__":
    # Example usage
    entries = ["dog", "cat", "fish"]
    clusters = ["animals", "pets"]
    clustered_entries, similarity_scores, clustered_indices = cluster_by_similarity(
        entries, clusters
    )
    print("Clustered Entries:", clustered_entries)
    print("Similarity Scores:", similarity_scores)
    print("Clustered Indices:", clustered_indices)
