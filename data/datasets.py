import torch
import json
from PIL import Image, ImageFile
from torch.utils.data import Dataset

import models.config as cfg

import pycocotools.mask as mask_utils


from data.processors import get_image_processor, get_tokenizer

import numpy as np

import random

from pathlib import Path

np.random.seed(0)
torch.random.seed = 0
random.seed(0)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ICCVTestDataset(Dataset):

    def __init__(self, vlm_cfg, path_to_data, path_to_json):
        super().__init__()

        self.path_to_data = path_to_data
        self.path_to_json = path_to_json

        with open(path_to_json) as f:
            self.data = json.load(f)

        self.vlm_cfg = vlm_cfg

        self.image_processor = get_image_processor(vlm_cfg.vit_img_size)
        self.tokenizer = get_tokenizer(
            vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        masks = torch.from_numpy(mask_utils.decode(item["rle"])).float()

        image = Image.open(self.path_to_data / "images" / item["image"]).convert("RGB")
        depth = Image.open(
            self.path_to_data / "depths" / item["image"].replace(".png", "_depth.png")
        ).convert("RGB")

        processed_images = [self.image_processor(image)]
        processed_depth = [self.image_processor(depth)]

        messages = [
            {
                "role": "user",
                "content": self.tokenizer.image_token
                * self.vlm_cfg.mp_image_token_length
                + item["conversations"][0]["value"]
                .replace("<image>", "")
                .replace("<mask>", "<mask_rgb><mask_depth>"),
            },
            {"role": "assistant", "content": ""},
        ]

        return {
            "images": processed_images,
            "masks": masks,
            "depths": processed_depth,
            "text_data": messages,
            "id": item["id"],
            "image": item["image"],
        }


class RGPTDataset(Dataset):  # Visual Question Answering Dataset
    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        depth_processor,
        mp_image_token_length,
        path_to_dataset,
        is_test=False,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.depth_processor = depth_processor
        self.mp_image_token_length = mp_image_token_length

        self.is_test = is_test

        self.path_to_dataset = Path(path_to_dataset)

    def __len__(self):
        return len(self.dataset)

    def _process_mm(self, item, mm_type, idx):
        if mm_type == "image":
            data_name = item["rgb_image"]
        else:
            data_name = item["depth_image"]

        if item["split"] == None:
            data_path = (
                self.path_to_dataset / item["dataset_name"] / f"{mm_type}s" / data_name
            )

        else:
            data_path = (
                self.path_to_dataset
                / item["dataset_name"]
                / item["split"]
                / f"{mm_type}s"
                / data_name
            )

        data = Image.open(data_path)

        if not isinstance(data, list):
            data = [data]

        # Process the images
        processed_images = []
        for image in data:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")

                if mm_type == "image":
                    processed_image = self.image_processor(image)
                else:
                    processed_image = self.depth_processor(image)

                processed_images.append(processed_image)
            else:
                raise ValueError(f"Error processing {mm_type} at index {idx}")

        return processed_images

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if "split" not in item:
            item["split"] = None

        if item["rle"] != None:
            masks = torch.from_numpy(mask_utils.decode(item["rle"])).float()
        else:
            masks = None

        processed_images = self._process_mm(item, "image", idx)
        processed_depths = self._process_mm(item, "depth", idx)

        if self.is_test == False:
            messages = [
                {
                    "role": "user",
                    "content": item["texts"]["user"],
                },
                {"role": "assistant", "content": item["texts"]["assistant"]},
            ]

        else:
            messages = [
                {
                    "role": "user",
                    "content": item["texts"]["user"],
                },
            ]

        messages[0]["content"] = (
            self.tokenizer.image_token
            * len(processed_images)
            * self.mp_image_token_length
            + messages[0]["content"]
        )

        if self.is_test == False:
            return {
                "images": processed_images,
                "masks": masks,
                "depths": processed_depths,
                "text_data": messages,
            }
        else:
            return {
                "id": item["id"],
                "images": processed_images,
                "masks": masks,
                "depths": processed_depths,
                "text_data": messages,
                "category": item["category"],
                "answer": item["texts"]["assistant"],
                "normalized_answer": item["normalized_answer"],
            }


class VQADataset(Dataset):  # Visual Question Answering Dataset
    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        mp_image_token_length,
        path_to_dataset,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.path_to_dataset = Path(path_to_dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle images (should be a list)
        images_data = item["rgb_image"]

        data_path = (
            self.path_to_dataset / item["dataset_name"] / f"images" / images_data
        )

        images_data = Image.open(data_path)

        if not isinstance(images_data, list):
            images_data = [images_data]

        # Now process the images
        processed_images = []
        for image in images_data:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_image = self.image_processor(image)
                processed_images.append(processed_image)
            else:
                raise ValueError(f"Error processing image at index {idx}")

        # Process text (should be a list)
        text_data = item["texts"]
        if not isinstance(text_data, list):
            text_data = [text_data]

        messages = []

        for text in text_data:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        messages[0]["content"] = (
            self.tokenizer.image_token
            * len(processed_images)
            * self.mp_image_token_length
            + messages[0]["content"]
        )

        return {
            "images": processed_images,
            "text_data": messages,
        }
