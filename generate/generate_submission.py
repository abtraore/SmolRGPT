import argparse
import torch
from torch.utils.data import DataLoader

from pathlib import Path

import models.config as config

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.datasets import ICCVTestDataset
from data.collators import ICCVTestCollator
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer

from tqdm import tqdm

import os

import pandas as pd

from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=120,
        help="Maximum number of tokens per output",
    )
    return parser.parse_args()


def main():

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    dataset_path = Path(f"{train_cfg.dataset_path}/warehouse-rgbd-smolRGPT")
    data_path = dataset_path / "test"
    annotation_path = dataset_path / "test.json"

    test_dataset = ICCVTestDataset(vlm_cfg, data_path, annotation_path)

    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    collator = ICCVTestCollator(tokenizer, max_length=vlm_cfg.vit_hidden_dim)

    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collator,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2,
    )

    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    answers = {"id": [], "image_id": [], "output": []}

    for batch in tqdm(test_dataloader):

        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        masks = batch["mask"]
        masks = [m.to(device) for m in masks]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ids = batch["id"]
        image_ids = batch["image_id"]

        # FOR DEBUGGING
        # print(f"Image shape: {image.shape}")
        # print(f"Depth shape: {depth.shape}")
        # print(f"Input IDs shape: {input_ids.shape}")
        # print(f"Attention mask shape: {attention_mask.shape}")
        # print(f"Number of masks: {len(masks)}")
        # print(f"Total masks: {sum([m.shape[-1] for m in masks])}")
        # if masks:
        #     print(f"First mask shape: {masks[0].shape}")

        with torch.no_grad():

            gen = model.generate(
                input_ids,
                image,
                depth,
                masks,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                greedy=True,
            )
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)

            answers["output"].extend([m.strip() for m in model_output])
            answers["id"].extend(ids)
            answers["image_id"].extend(image_ids)

    df = pd.DataFrame.from_dict(answers)
    df.to_csv("raw_output.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
