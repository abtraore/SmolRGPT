import torch
import torch.nn as nn


class RegionModalityProjector(nn.Module):
    def __init__(self, vision_hidden_dim, llm_hidden_dim, droput=0.1):
        super().__init__()

        self.proj = nn.Linear(vision_hidden_dim, llm_hidden_dim, bias=False)
        self.dropout = nn.Dropout(droput)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor):
        return self.dropout(self.proj(x))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MaskPooling(nn.Module):
    def __init__(self, mask_threshold=0.5):
        super().__init__()
        self.mask_threshold = mask_threshold

    def forward(self, x: torch.Tensor, mask_list, return_list=False, return_mask=False):
        """
        Args:
            x: [B, (HW), C]
            mask_list: List( tensor[M, IH, IW] )
        """
        batch_size = x.size(0)
        if mask_list is None:
            mask_list = [None for i in range(batch_size)]

        output = []
        attn_mask_list = []
        for i in range(batch_size):
            x_len = x.size(1)

            mask = mask_list[i].permute((2, 0, 1))

            if mask is None:
                output.append(None)
                attn_mask_list.append(None)
            else:
                # resize mask from image shape to feature map shape

                size = int(x_len**0.5)

                mask = mask.detach()
                mask = mask.float()[None, ...]
                mask = nn.functional.interpolate(
                    mask, size=(size, size), mode="bilinear"
                )
                mask = mask.to(x.dtype)
                mask = mask[0]
                feature = x[i]

                denorm = mask.sum(dim=(-1, -2)) + 1e-8  # M
                denorm = denorm.unsqueeze(-1)  # M, 1

                mask = mask.flatten(start_dim=1)  # M, H, W -> M, HW

                attn_mask_list.append(
                    (mask > self.mask_threshold).to(mask.dtype)
                )  # M, HW

                mask_pooled_x = torch.einsum(
                    "lc,ml->mc",
                    feature,
                    mask / denorm,
                )
                # mc output
                output.append(mask_pooled_x)

        if return_list:
            if return_mask:
                return output, attn_mask_list
            return output
        else:
            # FIXME: Not support Nonetype
            output = torch.cat(output)
            return output


class FeatureRefiner(nn.Module):
    def __init__(self, vision_hidden_size, upscale_factor):
        super().__init__()

        self.refiner = self._get_feature_refinement_module(
            vision_hidden_size, upscale_factor
        )

    def _get_feature_refinement_module(self, vision_hidden_size, upscale_factor=2):

        deconv_depth = upscale_factor
        modules = []
        for _ in range(deconv_depth - 1):
            modules.append(
                nn.ConvTranspose2d(
                    vision_hidden_size, vision_hidden_size, kernel_size=2, stride=2
                )
            )
            modules.append(LayerNorm2d(vision_hidden_size))
            modules.append(nn.GELU())
        modules.append(
            nn.ConvTranspose2d(
                vision_hidden_size, vision_hidden_size, kernel_size=2, stride=2
            )
        )
        modules.append(nn.GELU())

        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor):

        B, S, D = x.shape

        P_S = int(S**0.5)

        feat = x.permute(0, 2, 1).reshape(B, D, P_S, P_S)

        hres_feat = self.refiner(feat)

        hres_feat_flatten = hres_feat.reshape(B, D, -1).permute(0, 2, 1)

        return hres_feat_flatten
