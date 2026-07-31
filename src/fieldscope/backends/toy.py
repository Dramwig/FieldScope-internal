"""Small deterministic vector field used for local and CI verification."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class ToyFieldBackend:
    """Analytic spatially coupled field with no learned weights."""

    def __init__(self, device: str = "cpu", dtype: str = "float32", image_size: int = 128):
        self._device = torch.device(device)
        self._dtype = _DTYPES[dtype]
        self.image_size = image_size
        self._query_calls = 0
        self._evaluated_states = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def model_id(self) -> str:
        return "fieldscope/toy-coupled-field-v1"

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        images = images.to(device=self.device, dtype=self.dtype).clamp(0, 1)
        target = max(8, self.image_size // 8)
        images = F.interpolate(images, size=(target, target), mode="bilinear", align_corners=False)
        centered = images * 2.0 - 1.0
        luminance = (
            0.299 * centered[:, 0:1]
            + 0.587 * centered[:, 1:2]
            + 0.114 * centered[:, 2:3]
        )
        return torch.cat([centered, luminance], dim=1)

    def query_velocity(self, latents: torch.Tensor, clean_time: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)
        self._query_calls += 1
        self._evaluated_states += latents.shape[0]
        if clean_time.ndim == 0:
            clean_time = clean_time.expand(latents.shape[0])
        time = clean_time.to(device=self.device, dtype=self.dtype).reshape(-1, 1, 1, 1)
        local = F.avg_pool2d(latents, kernel_size=3, stride=1, padding=1)
        broad = F.avg_pool2d(latents, kernel_size=7, stride=1, padding=3)
        global_context = latents.mean(dim=(-2, -1), keepdim=True)
        horizontal = 0.5 * (torch.roll(latents, 1, -1) + torch.roll(latents, -1, -1))
        vertical = 0.5 * (torch.roll(latents, 1, -2) + torch.roll(latents, -1, -2))
        coupled = (
            0.65 * latents
            + 0.25 * local
            + 0.15 * broad
            + 0.10 * global_context
            + 0.08 * horizontal
            - 0.06 * vertical
        )
        return torch.tanh(coupled + (time - 0.5) * 0.4) + 0.05 * latents.square()

    def query_velocity_features(
        self,
        latents: torch.Tensor,
        clean_time: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Expose analytic hidden/Q/K maps for baseline-contract tests."""

        velocity = self.query_velocity(latents, clean_time)
        hidden = torch.cat([latents.to(velocity), velocity], dim=1)
        normalized = F.normalize(hidden.float(), dim=1).to(dtype=hidden.dtype)
        return velocity, {
            "dit_hidden": hidden,
            "dit_attention_q": normalized.unsqueeze(1),
            "dit_attention_k": normalized.unsqueeze(1),
        }

    def runtime_stats(self) -> dict[str, int]:
        return {
            "transformer_query_calls": self._query_calls,
            "transformer_evaluated_states": self._evaluated_states,
        }

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "toy",
            "model_id": self.model_id,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "time_convention": "clean_time: 0=noise, 1=image",
        }
