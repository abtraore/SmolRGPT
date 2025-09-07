import os
import argparse

from datasets import load_dataset
from datasets import Features, Value

import torch

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

import models.config as config
from data.datasets import VQADataset
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor


os.environ["TOKENIZERS_PARALLELISM"] = (
    "false"  # Otherwise, the tokenizer will through a warning
)


def parse_args():
    parser = argparse.ArgumentParser(description="RGB only generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF.",
        required=True,
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

    data_files = (
        train_cfg.dataset_path + "/" + "llava-cc3m-smolRGPT" + "/data/*.parquet"
    )

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

    ds = ds["train"]
    ds = ds.shuffle(seed=0)

    if train_cfg.data_cutoff_idx is None:
        total_samples = len(ds)  # Use the entire dataset
    else:
        total_samples = min(len(ds), train_cfg.data_cutoff_idx)

    val_size = int(total_samples * train_cfg.val_ratio)
    train_size = total_samples - val_size

    train_ds = ds.select(range(train_size))
    val_ds = ds.select(range(train_size, total_samples))

    image_processor = get_image_processor(vlm_cfg.vit_img_size)

    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    val_ds = VQADataset(
        train_ds.select(range(val_size)),
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        path_to_dataset=train_cfg.dataset_path,
    )

    iterator = iter(val_ds)
    for _ in range(len(val_ds)):

        batch = next(iterator)

        images = batch["images"]
        text_data = batch["text_data"][:2]

        encoded_prompt = tokenizer.apply_chat_template(
            [text_data[0]], tokenize=True, add_generation_prompt=True
        )

        with torch.no_grad():
            tokens = torch.tensor(encoded_prompt).unsqueeze(0).to(device)
            image = images[0].unsqueeze(0).to(device)
            depth = None
            masks = None

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
            print("=" * 25)


if __name__ == "__main__":
    main()
