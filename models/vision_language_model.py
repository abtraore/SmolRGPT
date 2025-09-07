import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.config import VLMConfig
from models.region_utils import FeatureRefiner, MaskPooling

from data.processors import get_tokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model


class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)

        self.MP = ModalityProjector(cfg)
        self.MP_depth = ModalityProjector(cfg)

        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(
            cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template
        )

        self.rgb_refiner = FeatureRefiner(960, 3)
        self.depth_refiner = FeatureRefiner(960, 3)
        self.mask_pooling = MaskPooling()

        # Freeze vision encoder (always)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        if cfg.stage == "connector_alignment":
            self.freeze_for_rgb_connector_alignement()
        elif cfg.stage == "refiner_alignment":
            self.freeze_for_refiner_alignement()
        else:
            self.unfreeze_sft()

    def freeze_for_refiner_alignement(self):

        self.freeze_llm()

        for param in self.MP.parameters():
            param.requires_grad = True

        for param in self.MP_depth.parameters():
            param.requires_grad = True

        for param in self.depth_refiner.parameters():
            param.requires_grad = True

        for param in self.rgb_refiner.parameters():
            param.requires_grad = True

        for param in self.mask_pooling.parameters():
            param.requires_grad = True

        print("Freezing for rgb refiner aligmnent...")

    def unfreeze_sft(self):

        self.unfreeze_llm()

        for param in self.MP.parameters():
            param.requires_grad = True

        for param in self.MP_depth.parameters():
            param.requires_grad = True

        for param in self.depth_refiner.parameters():
            param.requires_grad = True

        for param in self.rgb_refiner.parameters():
            param.requires_grad = True

        for param in self.mask_pooling.parameters():
            param.requires_grad = True

        print("Unfreezing for SFT...")

    def freeze_for_rgb_connector_alignement(self):

        self.freeze_llm()

        for param in self.rgb_refiner.parameters():
            param.requires_grad = False

        for param in self.depth_refiner.parameters():
            param.requires_grad = False

        for param in self.mask_pooling.parameters():
            param.requires_grad = False

        for param in self.MP_depth.parameters():
            param.requires_grad = False

        print("Freezing for rgb connector aligmnent...")

    def freeze_llm(self):
        # Freeze language model
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.is_llm_frozen = True

        print("LLM frozen...")

    def unfreeze_llm(self):
        # Freeze language model
        for param in self.decoder.parameters():
            param.requires_grad = True

        self.is_llm_frozen = False

        print("LLM unfrozen...")

    def _replace_img_tokens_with_embd(
        self, input_ids, token_embd, image_embd, pooled_rgb_embd, pooled_depth_embd
    ):
        """
        Replace every image-token and mask-token placeholder in `input_ids` with the corresponding
        embeddings while preserving gradients.
        """
        batch_size, seq_len = input_ids.shape
        embed_dim = token_embd.shape[-1]

        # Start with token embeddings
        updated_token_embd = token_embd.clone()

        # Handle image token replacement (existing code)
        image_mask = input_ids == self.tokenizer.image_token_id
        if image_mask.any():
            # Flatten and replace
            updated_token_embd[image_mask] = image_embd.view(-1, embed_dim).to(
                updated_token_embd.dtype
            )

        if pooled_rgb_embd != None:

            # Handle mask token replacement with gradient-preserving method
            mask_token_mask = input_ids == self.tokenizer.mask_rgb_token_id

            if mask_token_mask.any() and pooled_rgb_embd is not None:
                # Method 1: Using masked_scatter_
                # First, we need to ensure mask_embbed is the right shape
                # Count total mask tokens
                total_mask_tokens = mask_token_mask.sum().item()

                if total_mask_tokens != pooled_rgb_embd.shape[0]:
                    print(
                        f"Warning: Expected {total_mask_tokens} mask embeddings, got {pooled_rgb_embd.shape[0]}"
                    )

                # Create a flat view of positions where mask tokens are
                flat_updated = updated_token_embd.view(-1, embed_dim)
                flat_mask = mask_token_mask.view(-1)

                # Replace using index_copy_ which preserves gradients
                mask_indices = flat_mask.nonzero(as_tuple=True)[0]

                # Ensure we don't exceed available mask embeddings
                num_replacements = min(len(mask_indices), pooled_rgb_embd.shape[0])
                if num_replacements < len(mask_indices):
                    print(
                        f"Warning: Only {num_replacements} mask embeddings available for {len(mask_indices)} mask tokens"
                    )
                    mask_indices = mask_indices[:num_replacements]

                # Create a new tensor with scatter to preserve gradients
                scattered = torch.zeros_like(flat_updated)
                scattered[mask_indices] = pooled_rgb_embd[:num_replacements].to(
                    scattered.dtype
                )

                # Combine with original embeddings
                flat_updated = torch.where(
                    flat_mask.unsqueeze(-1), scattered, flat_updated
                )

                # Reshape back
                updated_token_embd = flat_updated.view(batch_size, seq_len, embed_dim)

        if pooled_depth_embd != None:
            ###### DEPTH
            # Handle mask token replacement with gradient-preserving method
            mask_token_mask = input_ids == self.tokenizer.mask_depth_token_id

            if mask_token_mask.any() and pooled_depth_embd is not None:
                # Method 1: Using masked_scatter_
                # First, we need to ensure mask_embbed is the right shape
                # Count total mask tokens
                total_mask_tokens = mask_token_mask.sum().item()

                if total_mask_tokens != pooled_depth_embd.shape[0]:
                    print(
                        f"Warning: Expected {total_mask_tokens} mask embeddings, got {pooled_depth_embd.shape[0]}"
                    )

                # Create a flat view of positions where mask tokens are
                flat_updated = updated_token_embd.view(-1, embed_dim)
                flat_mask = mask_token_mask.view(-1)

                # Replace using index_copy_ which preserves gradients
                mask_indices = flat_mask.nonzero(as_tuple=True)[0]

                # Ensure we don't exceed available mask embeddings
                num_replacements = min(len(mask_indices), pooled_depth_embd.shape[0])
                if num_replacements < len(mask_indices):
                    print(
                        f"Warning: Only {num_replacements} mask embeddings available for {len(mask_indices)} mask tokens"
                    )
                    mask_indices = mask_indices[:num_replacements]

                # Create a new tensor with scatter to preserve gradients
                scattered = torch.zeros_like(flat_updated)
                scattered[mask_indices] = pooled_depth_embd[:num_replacements].to(
                    scattered.dtype
                )

                # Combine with original embeddings
                flat_updated = torch.where(
                    flat_mask.unsqueeze(-1), scattered, flat_updated
                )

                # Reshape back
                updated_token_embd = flat_updated.view(batch_size, seq_len, embed_dim)

        return updated_token_embd

    def _partial_forwad(self, input_ids, image, depth, mask):

        image_embd = self.vision_encoder(image)

        if self.cfg.stage == "refiner_alignment":
            image_embd = image_embd.detach().requires_grad_(True)

        image_embd = self.MP(image_embd)  # [num_images, mp_image_token_length, D_lm]

        rgb_refined = self.rgb_refiner(image_embd)

        if mask != None:
            pooled_rgb_embd = self.mask_pooling(rgb_refined, mask)
        else:
            pooled_rgb_embd = None

        if mask != None and depth != None:
            depth_embd = self.vision_encoder(depth)
            depth_embd = self.MP_depth(
                depth_embd
            )  # [num_images, mp_image_token_length, D_lm]
            depth_refined = self.depth_refiner(depth_embd)
            pooled_depth_embd = self.mask_pooling(depth_refined, mask)

        else:
            pooled_depth_embd = None

        token_embd = self.decoder.token_embedding(input_ids)  # [B, T_sequence, D_lm]

        return image_embd, pooled_rgb_embd, pooled_depth_embd, token_embd

    def forward(
        self,
        input_ids,
        image,
        depth,
        mask,
        attention_mask=None,
        targets=None,
    ):

        image_embd, pooled_rgb_embd, pooled_depth_embd, token_embd = (
            self._partial_forwad(input_ids, image, depth, mask)
        )

        updated_token_embd = self._replace_img_tokens_with_embd(
            input_ids, token_embd, image_embd, pooled_rgb_embd, pooled_depth_embd
        )

        # The updated_token_embd is now the token_embd with image parts replaced.
        # The attention_mask comes from the collator and should already cover the full sequence.
        logits, _ = self.decoder(updated_token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)  # Apply LM head
            # Loss is calculated over all tokens, but `targets` (labels) will have -100 for non-answer tokens.
            # No need to slice logits based on image embedding size here, as the target mask handles it.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        image,
        depth,
        mask,
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False,
    ):

        image_embd, pooled_rgb_embd, pooled_depth_embd, token_embd = (
            self._partial_forwad(input_ids, image, depth, mask)
        )

        # 3. Combine image and text embeddings
        initial_combined_embeds = self._replace_img_tokens_with_embd(
            input_ids, token_embd, image_embd, pooled_rgb_embd, pooled_depth_embd
        )

        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = input_ids.size(0)

        # --- Multimodal Prefill Phase ---
        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0,
        )

        last_token_output_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # --- Decode Phase by sampling tokens autoregressively using the kv-cache ---
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(
                    current_logits, top_k=top_k, top_p=top_p
                )
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)

            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(
                next_token_id
            )  # [B, 1, D_lm]

            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            device=attention_mask.device,
                            dtype=attention_mask.dtype,
                        ),
                    ),
                    dim=1,
                )

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )

            last_token_output = decode_step_output[:, -1, :]

            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty(
                (batch_size, 0), dtype=torch.long, device=input_ids.device
            )

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if (
            self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0
        ):  # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = (
                generated_ids == self.tokenizer.eos_token_id
            )  # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(
                seq_len, device=device
            )  # Create column indices [0, 1, ..., seq_len-1]

            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(
                eos_mask,
                col_indices_for_min.unsqueeze(0).expand_as(generated_ids),
                seq_len + 1,
            )

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(
                first_eos_indices_values, max=seq_len
            )

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = (
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand_as(generated_ids)
            )

            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = (
                col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            )

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        print(config_path)
        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(
            model, weights_path, strict=False
        )  # have change this for compatibility issue

        # Done!
        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.

        Args:
            save_directory (str): The directory to save the model and config.
        """
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.

        Args:
            repo_id (str): The repo ID on the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        # Create repo
        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            # Save to tmp directory
            self.save_pretrained(save_path)

            # Save model card
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            # Upload
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
