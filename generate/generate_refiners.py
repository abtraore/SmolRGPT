import argparse

from datasets import load_dataset
from datasets import Features, Value

import torch

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

import models.config as config
from data.datasets import RGPTDataset
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_depth_processor


# Otherwise, the tokenizer will through a warning
import os

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

    args = parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Loading weights from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    data_files = train_cfg.dataset_path + "/" + "osd-110k-smolRGPT" + "/data/*.parquet"

    features = Features(
        {
            "id_db": Value("string"),
            "id": Value("string"),
            "rgb_image": Value("string"),
            "depth_image": Value("string"),
            "rle": [{"size": [Value("int32")], "counts": Value("string")}],
            "texts": {"user": Value("string"), "assistant": Value("string")},
            "category": Value("string"),
            "normalized_answer": Value("string"),
            "dataset_name": Value("string"),
        }
    )

    ds = load_dataset(
        "parquet",
        data_files=data_files,
        features=features,
    )

    train_ds = ds["train"]

    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    depth_processor = get_depth_processor(vlm_cfg.vit_img_size)

    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    train_ds = RGPTDataset(
        dataset=train_ds,
        tokenizer=tokenizer,
        image_processor=image_processor,
        depth_processor=depth_processor,
        mp_image_token_length=vlm_cfg.mp_image_token_length,
        path_to_dataset=train_cfg.dataset_path,
    )

    iterator = iter(train_ds)
    for _ in range(len(train_ds)):

        batch = next(iterator)

        images = batch["images"]
        text_data = batch["text_data"]
        masks = [batch["masks"]]
        depths = batch["depths"]

        encoded_prompt = tokenizer.apply_chat_template(
            [text_data[0]], tokenize=True, add_generation_prompt=True
        )

        with torch.no_grad():
            tokens = torch.tensor(encoded_prompt).unsqueeze(0).to(device)
            image = images[0].unsqueeze(0).to(device)
            depth = depths[0].unsqueeze(0).to(device)
            masks = [m.to(device) for m in masks]

            gen = model.generate(
                tokens,
                image,
                depth,
                masks,
                max_new_tokens=args.max_new_tokens,
                greedy=True,
            )
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]

            print(f"Question: {text_data[0]['content'].replace('<|image|>',"")}")
            print(f"Predicted: {model_output}")
            print(f"Answer: {text_data[1]['content']}")
            print()


if __name__ == "__main__":
    main()
