from dataclasses import dataclass, field


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 4 * vit_hidden_dim
    vit_patch_size: int = 16
    vit_img_size: int = 256
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip2-base-patch16-256"

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 8192
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = (
        3  # Number of extra tokens for the VLM (image start, image end, image token)
    )
    lm_vocab_size: int = (
        lm_base_vocab_size + extra_token_amount
    )  # Not a great way to do this, but it works for now (vlm_extra_tokens cannot be a dict, since this is mutable, and a Field has no len() function)
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_max_length: int = 512
    lm_use_tokens: bool = (
        False  # Decide if the LM expects tokens or embeddings as input (if using as a backbone for the VLM, set to False)
    )
    lm_tie_weights: bool = (
        True  # Decide if you want to tie the LM Head weight to the token embedding weights
    )
    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_chat_template: str = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    lm_eos_token_id: int = 0

    mp_pixel_shuffle_factor: int = 2
    mp_image_token_length: int = 64

    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "image_token": "<|image|>",
            "mask_rgb_token": "<mask_rgb>",
            "mask_depth_token": "<mask_depth>",
        }
    )  # , "boi_token": "<|image_start|>", "eoi_token": "<|image_end|>"})
    vlm_load_backbone_weights: bool = True
    vlm_restore_previous_state: bool = True
    vlm_checkpoint_path: str = "checkpoints/vlm_checkpoints"
    hf_repo_name: str = None  # "nanoVLM"

    # added:
    stage: str = (
        "connector_alignment",  # "connector_alignment" | "refiner_alignment" | "sft" | mixed
    )


@dataclass
class TrainConfig:
    lr_mp: float = 5e-5
    lr_mp_depth: float = 5e-5
    lr_rgb_refiner: float = 5e-5
    lr_depth_refiner: float = 5e-5
    lr_backbones: float = 5e-5

    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    # val_ratio: float = 0.0001
    batch_size: int = 14
    gradient_accumulation_steps: int = 4  # 2
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = 3600
    epochs: int = 20
    compile: bool = False
    resume_from_vlm_checkpoint: bool = (
        False  # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch
    )
    dataset_path: str = "/home/atraore/embia/NanoRGPT/datasets"  # Need to be changed
    dataset_name: tuple[str, ...] = (
        "warehouse-rgbd-smolRGPT",
        "osd-110k-smolRGPT",
        "osd-smolRGPT",
        "llava-cc3m-smolRGPT",
    )

    wandb_entity: str = "atraore"  # Indicate the entity to log to in wandb
    log_wandb: bool = False
