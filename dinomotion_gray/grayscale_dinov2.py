from pathlib import Path
import os

import torch
from torch import nn

from dinomotion_gray.lora import inject_lora_qv


class GrayscaleDINOv2(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        train_patch_embed: bool = True,
        torch_home: str | None = None,
    ):
        super().__init__()
        if torch_home is not None:
            os.environ.setdefault("TORCH_HOME", str(Path(torch_home).resolve()))

        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.model_name = model_name
        self.patch_size = self._get_patch_size()
        self.embed_dim = getattr(self.model, "embed_dim", None) or self.model.norm.normalized_shape[0]

        self._convert_patch_embed_to_grayscale(train_patch_embed=train_patch_embed)
        inject_lora_qv(self.model, rank=lora_rank, alpha=lora_alpha)
        self._freeze_backbone(train_patch_embed=train_patch_embed)

    def _get_patch_size(self):
        patch_size = getattr(self.model.patch_embed, "patch_size", 14)
        if isinstance(patch_size, tuple):
            return int(patch_size[0])
        return int(patch_size)

    def _convert_patch_embed_to_grayscale(self, train_patch_embed: bool):
        old_proj = self.model.patch_embed.proj
        new_proj = nn.Conv2d(
            1,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        with torch.no_grad():
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        for param in new_proj.parameters():
            param.requires_grad = train_patch_embed
        self.model.patch_embed.proj = new_proj

    def _freeze_backbone(self, train_patch_embed: bool):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            if ".q_a." in name or ".q_b." in name or ".v_a." in name or ".v_b." in name:
                param.requires_grad = True

        for param in self.model.patch_embed.proj.parameters():
            param.requires_grad = train_patch_embed

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected grayscale BCHW input, got {tuple(x.shape)}")

        out = self.model.forward_features(x)
        if not isinstance(out, dict):
            raise TypeError("Expected DINOv2 forward_features to return a dict.")

        patch_tokens = out.get("x_norm_patchtokens")
        if patch_tokens is None:
            raise KeyError("DINOv2 forward_features did not provide 'x_norm_patchtokens'.")

        batch, _, height, width = x.shape
        feat_h = height // self.patch_size
        feat_w = width // self.patch_size
        feature_map = patch_tokens.reshape(batch, feat_h, feat_w, -1).permute(0, 3, 1, 2).contiguous()
        return feature_map
