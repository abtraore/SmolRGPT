import os

import argparse

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from accelerate import Accelerator, DistributedDataParallelKwargs

from models.config import TrainConfig
from classifier.utils import ICCVRouterDataset, compute_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Classifier trainer")
    parser.add_argument("--task", type=str, choices=["question", "answer"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_step", type=int, default=250)
    parser.add_argument("--eval_step", type=int, default=250)
    parser.add_argument("--save_step", type=int, default=250)
    parser.add_argument("--report_to", type=str, default="tensorboard")

    args = parser.parse_args()

    train_cfg = TrainConfig()

    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(static_graph=True)]
    )

    PATH_TO_TRAIN_ANNOTATION = (
        f"{train_cfg.dataset_path}/warehouse-rgbd-smolRGPT/train.json"
    )
    PATH_TO_VAL_ANNOTATION = (
        f"{train_cfg.dataset_path}/warehouse-rgbd-smolRGPT/val.json"
    )

    DEVICE = accelerator.device if torch.cuda.is_available() else "cpu"

    model_id = "allenai/longformer-base-4096"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=4,
        device_map=DEVICE,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    train_dataset = ICCVRouterDataset(PATH_TO_TRAIN_ANNOTATION, tokenizer, args.task)
    val_dataset = ICCVRouterDataset(PATH_TO_VAL_ANNOTATION, tokenizer, args.task)

    training_args = TrainingArguments(
        output_dir=f"checkpoints/classifier_checkpoints/task-classifier-iccv-question",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        dataloader_prefetch_factor=2,
        dataloader_num_workers=4,
        gradient_checkpointing=True,
        max_steps=args.max_step,
        save_total_limit=1,
        weight_decay=0.01,
        bf16=True,
        optim="adamw_torch",
        report_to=args.report_to,
        eval_strategy="steps",
        eval_steps=args.eval_step,
        save_steps=args.save_step,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Wait for all processes to finish before ending
    accelerator.wait_for_everyone()

    # Only save on main process
    if accelerator.is_main_process:
        trainer.save_model()
        print("Training completed successfully!")
