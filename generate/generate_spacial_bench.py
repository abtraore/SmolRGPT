import argparse
import pandas as pd
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader

import models.config as config

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.datasets import RGPTDataset
from data.collators import ICCVTestCollator
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_depth_processor


from datasets import load_dataset
from datasets import Features, Value


# Otherwise, the tokenizer will through a warning
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Inference batch size",
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

    dataset_name = "spacial-rgpt-bench-smolRGPT"

    data_files = train_cfg.dataset_path + "/" + dataset_name + "/data/*.parquet"

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
    )["train"]

    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    depth_processor = get_depth_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    val_dataset = RGPTDataset(
        ds,
        tokenizer,
        image_processor,
        depth_processor,
        vlm_cfg.mp_image_token_length,
        path_to_dataset=train_cfg.dataset_path,
        is_test=True,
    )

    collator = ICCVTestCollator(
        tokenizer, max_length=vlm_cfg.vit_hidden_dim, use_image_id=False
    )

    test_dataloader = DataLoader(
        val_dataset,
        collate_fn=collator,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2,
    )

    answers = {"id": [], "output": []}

    for batch in tqdm(test_dataloader):

        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        masks = batch["mask"]
        masks = [m.to(device) for m in masks]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ids = batch["id"]

        with torch.no_grad():

            gen = model.generate(
                input_ids,
                image,
                depth,
                masks,
                attention_mask=attention_mask,
                max_new_tokens=120,
                greedy=True,
            )
            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)

            answers["output"].extend([m.strip() for m in model_output])
            answers["id"].extend(ids)

    df = pd.DataFrame.from_dict(answers)
    df.to_csv("answer_spacial_bench.csv", sep=";", index=False)


if __name__ == "__main__":
    main()
