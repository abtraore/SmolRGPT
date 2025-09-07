import math
import time
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from datasets import Features, Value

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from data.collators import (
    VQACollator,
    ICCVCollator,
)
from data.datasets import VQADataset, RGPTDataset
from data.processors import get_image_processor, get_depth_processor, get_tokenizer
from models.vision_language_model import VisionLanguageModel
import models.config as config
import models.utils as utils


from extractor.ollama_utils import llm_extract

# Added

from pathlib import Path


import os

os.environ["HF_HOME"] = "/tank/hf_tank"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "2"


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def init_dist():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def destroy_dist():
    dist.destroy_process_group()


def is_dist():
    return dist.is_available() and dist.is_initialized()


def is_master():
    return dist.get_rank() == 0 if is_dist() else True


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def dist_gather(o):
    o_all = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(o_all, o)
    return o_all


def wrap_model(model):
    return DistributedDataParallel(
        model, device_ids=[dist.get_rank()], find_unused_parameters=False
    )


def get_run_name(train_cfg, vlm_cfg):
    dataset_size = (
        "full_ds"
        if train_cfg.data_cutoff_idx is None
        else f"{train_cfg.data_cutoff_idx}samples"
    )
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    epochs = f"ep{train_cfg.epochs}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{epochs}_{learning_rate}_{date}"


def get_dataloaders(train_cfg, vlm_cfg):
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.vit_img_size)
    depth_processor = get_depth_processor(vlm_cfg.vit_img_size)
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    combined_train_data = []
    combined_val_data = []
    for dataset_name in train_cfg.dataset_name:

        dataset_root = Path(train_cfg.dataset_path + "/" + dataset_name)

        if Path.exists(dataset_root / "train") == False:
            data_files = train_cfg.dataset_path + "/" + dataset_name + "/data/*.parquet"

        else:

            data_files = {
                "train": train_cfg.dataset_path
                + "/"
                + dataset_name
                + "/data/train-*.parquet",
                "validation": train_cfg.dataset_path
                + "/"
                + dataset_name
                + "/data/val*.parquet",
            }

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
            cache_dir="/tank/smolrgpt-cache",
        )

        splits = list(ds.keys())

        # For alignment pretraining.
        if len(splits) == 1 and splits[0] == "train":
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

        # For Refiner warmump and SFT.
        else:
            if train_cfg.data_cutoff_idx is None:
                train_ds = ds["train"]
                val_ds = ds["validation"]
            else:
                train_ds = ds["train"].select(range(train_cfg.data_cutoff_idx))
                val_ds = ds["validation"].select(range(train_cfg.data_cutoff_idx))

            train_ds = train_ds.map(lambda x: {"split": "train"})
            val_ds = val_ds.map(lambda x: {"split": "val"})

        combined_train_data.append(train_ds)
        combined_val_data.append(val_ds)

    train_ds = concatenate_datasets(combined_train_data)
    val_ds = concatenate_datasets(combined_val_data)

    if vlm_cfg.stage == "connector_alignment":

        # Apply cutoff if specified
        if train_cfg.data_cutoff_idx is None:
            total_samples = len(train_ds)  # Use the entire dataset
        else:
            total_samples = min(len(train_ds), train_cfg.data_cutoff_idx)

        val_size = int(total_samples * train_cfg.val_ratio)
        train_size = total_samples - val_size

        train_dataset = VQADataset(
            train_ds.select(range(train_size)),
            tokenizer,
            image_processor,
            vlm_cfg.mp_image_token_length,
            path_to_dataset=train_cfg.dataset_path,
        )

        val_dataset = VQADataset(
            train_ds.select(range(val_size)),
            tokenizer,
            image_processor,
            vlm_cfg.mp_image_token_length,
            path_to_dataset=train_cfg.dataset_path,
        )

    else:

        train_dataset = RGPTDataset(
            dataset=train_ds,
            tokenizer=tokenizer,
            image_processor=image_processor,
            depth_processor=depth_processor,
            mp_image_token_length=vlm_cfg.mp_image_token_length,
            path_to_dataset=train_cfg.dataset_path,
        )

        val_dataset = RGPTDataset(
            dataset=val_ds,
            tokenizer=tokenizer,
            image_processor=image_processor,
            depth_processor=depth_processor,
            mp_image_token_length=vlm_cfg.mp_image_token_length,
            path_to_dataset=train_cfg.dataset_path,
        )

    # Create collators
    if vlm_cfg.stage == "connector_alignment":
        train_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    else:
        train_collator = ICCVCollator(tokenizer, vlm_cfg.lm_max_length)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders
    train_sampler = DistributedSampler(
        train_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        sampler=train_sampler,
        collate_fn=train_collator,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False,  # Usually False for validation
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=train_cfg.batch_size,
        sampler=val_sampler,
        collate_fn=train_collator,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader


# Not used...
def compute_accuracy(model, tokenizer, val_loader, device):
    acc = 0
    acc_left_right = 0
    acc_distance = 0
    acc_mcq = 0
    acc_count = 0

    total = 0
    for batch in val_loader:

        image = batch["image"].to(device)
        depth = batch["depth"].to(device)
        masks = batch["mask"]
        masks = [m.to(device) for m in masks]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer = batch["normalized_answer"]
        category = batch["category"]

        with torch.no_grad():

            gen = model.generate(
                input_ids,
                image,
                depth,
                masks,
                attention_mask=attention_mask,
                max_new_tokens=128,
            )

            model_output = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for i in range(len(model_output)):
                total += 1

                mo = model_output[i]
                ans = answer[i]
                cat = category[i]

                normalized_str = llm_extract(mo, cat)

                if cat == "left_right":
                    if ans == normalized_str:
                        acc_left_right += 1

                if cat == "count":
                    if ans == normalized_str:
                        acc_count += 1

                if cat == "distance":
                    acc_distance += torch.sqrt(
                        torch.nn.functional.mse_loss(
                            torch.tensor(float(ans)),
                            torch.tensor(float(normalized_str)),
                        )
                    )

                if cat == "mcq":
                    acc_mcq += 1

    acc_mcq /= total
    acc_distance /= total
    acc_count /= total
    acc_left_right /= total
    acc /= total

    return acc, acc_left_right, acc_count, acc_distance, acc_mcq


# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(train_cfg, vlm_cfg):
    train_loader, val_loader = get_dataloaders(train_cfg, vlm_cfg)
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template
    )

    total_dataset_size = len(train_loader.dataset)
    if train_cfg.log_wandb == True and is_master():
        run_name = get_run_name(train_cfg, vlm_cfg)
        if train_cfg.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")

        if train_cfg.log_wandb == True:
            run = wandb.init(
                entity=train_cfg.wandb_entity,
                project="nanoVLM",
                config={"VLMConfig": asdict(vlm_cfg), "TrainConfig": asdict(train_cfg)},
                name=run_name,
            )

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
    else:
        model = VisionLanguageModel(
            vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights
        )

    if is_master():
        print(
            f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
        )
        print(
            f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            print(
                f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}"
            )
        print(
            f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader)*get_world_size())} batches/epoch, batch size {int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            print(
                f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}"
            )

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = [
        {
            "params": list(model.MP.parameters()),
            "lr": train_cfg.lr_mp,
        },
        {
            "params": list(model.MP_depth.parameters()),
            "lr": train_cfg.lr_mp_depth,
        },
        {
            "params": list(model.rgb_refiner.parameters()),
            "lr": train_cfg.lr_rgb_refiner,
        },
        {
            "params": list(model.depth_refiner.parameters()),
            "lr": train_cfg.lr_depth_refiner,
        },
        {
            "params": list(model.decoder.parameters())
            + list(model.vision_encoder.parameters()),
            "lr": train_cfg.lr_backbones,
        },
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (
            torch.device("mps")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()

    print(f"Using device: {device}")
    model.to(device)

    if train_cfg.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    epoch_times = []
    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(train_cfg.epochs):
        epoch_start_time = time.time()
        model.train()
        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            is_update_step = (
                i + 1
            ) % train_cfg.gradient_accumulation_steps == 0 or i + 1 == len(train_loader)
            batch_start_time = time.time()
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Added
            if vlm_cfg.stage == "connector_alignment":
                depths = None
                masks = None

            else:
                depths = batch["depth"].to(device)
                masks = batch["mask"]
                masks = [m.to(device) for m in masks]

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (
                is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step
            ):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=(
                    torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16
                ),
            )
            with autocast_context:

                with context:
                    _, loss = model(
                        input_ids,
                        images,
                        depths,
                        masks,
                        attention_mask=attention_mask,
                        targets=labels,
                    )

            if train_cfg.gradient_accumulation_steps > 1:
                loss = loss / train_cfg.gradient_accumulation_steps

            loss.backward()

            if is_update_step:
                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, max_norm=train_cfg.max_grad_norm
                    )

                adj_lr_backbones = get_lr(
                    global_step,
                    train_cfg.lr_backbones,
                    len(train_loader)
                    * train_cfg.epochs
                    // train_cfg.gradient_accumulation_steps,
                )

                adj_lr_mp = get_lr(
                    global_step,
                    train_cfg.lr_mp,
                    len(train_loader)
                    * train_cfg.epochs
                    // train_cfg.gradient_accumulation_steps,
                )

                adj_lr_mp_depth = get_lr(
                    global_step,
                    train_cfg.lr_mp_depth,
                    len(train_loader)
                    * train_cfg.epochs
                    // train_cfg.gradient_accumulation_steps,
                )

                adj_lr_rgb_refiner = get_lr(
                    global_step,
                    train_cfg.lr_rgb_refiner,
                    len(train_loader)
                    * train_cfg.epochs
                    // train_cfg.gradient_accumulation_steps,
                )

                adj_lr_depth_refiner = get_lr(
                    global_step,
                    train_cfg.lr_depth_refiner,
                    len(train_loader)
                    * train_cfg.epochs
                    // train_cfg.gradient_accumulation_steps,
                )

                # Update LRs
                optimizer.param_groups[0]["lr"] = adj_lr_mp
                optimizer.param_groups[1]["lr"] = adj_lr_mp_depth
                optimizer.param_groups[2]["lr"] = adj_lr_rgb_refiner
                optimizer.param_groups[3]["lr"] = adj_lr_depth_refiner
                optimizer.param_groups[4]["lr"] = adj_lr_backbones

                optimizer.step()
                optimizer.zero_grad()

            if is_master() and train_cfg.log_wandb == True:

                run.log({"lr_mp": optimizer.param_groups[0]["lr"]}, step=global_step)
                run.log(
                    {"lr_mp_depth": optimizer.param_groups[1]["lr"]}, step=global_step
                )
                run.log(
                    {"lr_rgb_refiner": optimizer.param_groups[2]["lr"]},
                    step=global_step,
                )
                run.log(
                    {"lr_depth_refiner": optimizer.param_groups[3]["lr"]},
                    step=global_step,
                )
                run.log(
                    {"lr_backbone": optimizer.param_groups[4]["lr"]}, step=global_step
                )

            batch_loss = loss.item()
            if train_cfg.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_cfg.gradient_accumulation_steps
            total_train_loss += batch_loss

            num_tokens = torch.sum(
                attention_mask
            ).item()  # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = num_tokens / batch_duration

            # gather loss and t/s from all ranks if DDP
            batch_loss = mean(dist_gather(batch_loss)) if is_dist() else batch_loss
            tokens_per_second = (
                sum(dist_gather(tokens_per_second)) if is_dist() else tokens_per_second
            )

            if (
                train_cfg.eval_in_epochs
                and global_step % train_cfg.eval_interval == 0
                and is_update_step
            ):
                model.eval()
                if device == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    save = False
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["image"].to(device)
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        if vlm_cfg.stage == "connector_alignment":
                            depths = None
                            masks = None
                        else:
                            depths = batch["depth"].to(device)
                            masks = batch["mask"]
                            masks = [m.to(device) for m in masks]

                        with autocast_context:
                            _, loss = model(
                                input_ids,
                                images,
                                depths,
                                masks,
                                attention_mask=attention_mask,
                                targets=labels,
                            )

                        total_val_loss += loss.item()
                    avg_val_loss = total_val_loss / len(val_loader)
                    avg_val_loss = (
                        mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    )
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        save = True
                    if train_cfg.log_wandb == True and is_master():
                        run.log({"val_loss": avg_val_loss}, step=global_step)

                    if is_master() and global_step % (train_cfg.eval_interval * 2) == 0:
                        eval_model = (
                            model.module if is_dist() else model
                        )  # unwrap the model for eval if DDP

                        if save:
                            eval_model.save_pretrained(
                                save_directory=os.path.join(
                                    vlm_cfg.vlm_checkpoint_path, run_name
                                )
                            )

                        print(
                            f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}"
                        )
                    elif (
                        is_master()
                        and not global_step % (train_cfg.eval_interval * 4) == 0
                    ):
                        print(
                            f"Step: {global_step}, Loss: {batch_loss:.4f}, Tokens/s: {tokens_per_second:.2f}"
                        )

                model.train()

            if train_cfg.log_wandb == True and is_master():

                run.log(
                    {
                        "batch_loss": batch_loss,
                        "tokens_per_second": tokens_per_second,
                        **(
                            {"grad_norm": grad_norm}
                            if train_cfg.max_grad_norm is not None and is_update_step
                            else {}
                        ),
                    },
                    step=global_step,
                )

            if is_update_step:
                global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)
        # gather average batch loss from all ranks if DDP
        avg_train_loss = (
            mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss
        )

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = (
            sum(dist_gather(total_tokens_processed))
            if is_dist()
            else total_tokens_processed
        )
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb == True:
                run.log(
                    {
                        "epoch_loss": avg_train_loss,
                        "epoch_duration": epoch_duration,
                        "epoch_tokens_per_second": epoch_tokens_per_second,
                    }
                )

            print(
                f"Epoch {epoch+1}/{train_cfg.epochs}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}"
            )

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        total_samples_processed = len(train_loader.dataset) * train_cfg.epochs
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        # Push the best model to the hub (Please set your user name in the config!)
        if vlm_cfg.hf_repo_name is not None:
            print("Training complete. Pushing model to Hugging Face Hub...")

            hf_model = VisionLanguageModel.from_pretrained(
                os.path.join(vlm_cfg.vlm_checkpoint_path, run_name)
            )
            hf_model.push_to_hub(vlm_cfg.hf_repo_name)

        if train_cfg.log_wandb == True:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--last_stage_checkpoint", type=str, help="Last stage checkpoint"
    )
    parser.add_argument(
        "--stage",
        type=str,
        help="Training stage.",
        choices=["connector_alignment", "refiner_alignment", "sft", "mixed"],
        required=True,
    )
    parser.add_argument(
        "--vlm_checkpoint_path",
        type=str,
        help="Path to the VLM checkpoint for loading or saving",
    )
    parser.add_argument(
        "--compile", type=bool, help="Use torch.compile to optimize the model"
    )
    parser.add_argument(
        "--resume_from_vlm_checkpoint",
        type=bool,
        default=False,
        help="Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)",
    )
    parser.add_argument("--log_wandb", type=bool, default=True, help="Log to wandb")

    args = parser.parse_args()
    

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()

    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.log_wandb is not None:
        train_cfg.log_wandb = args.log_wandb

    # Added
    if args.stage is not None:
        vlm_cfg.stage = args.stage

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    # Dataset setting
    if vlm_cfg.stage == "connector_alignment":
        train_cfg.lr_mp = 1e-4
        train_cfg.dataset_name = ("llava-cc3m-smolRGPT",)

    elif vlm_cfg.stage == "refiner_alignment":

        assert (
            args.last_stage_checkpoint != None
        ), "Last stage checkpoint needed to continue training."

        train_cfg.resume_from_vlm_checkpoint = True
        vlm_cfg.vlm_checkpoint_path = args.last_stage_checkpoint

        train_cfg.lr_rgb_refiner = 1e-4
        train_cfg.lr_depth_refiner = 1e-4
        train_cfg.dataset_name = ("osd-110k-smolRGPT",)

    elif vlm_cfg.stage == "sft":

        assert (
            args.last_stage_checkpoint != None
        ), "Last stage checkpoint needed to continue training."

        train_cfg.resume_from_vlm_checkpoint = True
        vlm_cfg.vlm_checkpoint_path = args.last_stage_checkpoint

        train_cfg.lr_mp = 5e-5
        train_cfg.lr_mp_depth = 5e-5
        train_cfg.lr_rgb_refiner = 5e-5
        train_cfg.lr_depth_refiner = 5e-5
        train_cfg.lr_backbones = 5e-5
        train_cfg.dataset_name = ("warehouse-rgbd-smolRGPT",)

    elif vlm_cfg.stage == "mixed":

        assert (
            args.last_stage_checkpoint != None
        ), "Last stage checkpoint needed to continue training."

        train_cfg.resume_from_vlm_checkpoint = True
        vlm_cfg.vlm_checkpoint_path = args.last_stage_checkpoint

        train_cfg.lr_mp = 5e-5
        train_cfg.lr_mp_depth = 5e-5
        train_cfg.lr_rgb_refiner = 5e-5
        train_cfg.lr_depth_refiner = 5e-5
        train_cfg.lr_backbones = 5e-5
        train_cfg.dataset_name = ("warehouse-rgbd-smolRGPT", "osd-smolRGPT")

    print(f"STAGE: {vlm_cfg.stage}\n{ train_cfg.dataset_name}")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        print("--- VLM Config ---")
        print(vlm_cfg)
        print("--- Train Config ---")
        print(train_cfg)

    train(train_cfg, vlm_cfg)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()
