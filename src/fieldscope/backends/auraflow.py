"""AuraFlow adapter with explicit time and velocity convention conversion."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from fieldscope.config import BackendConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class AuraFlowBackend:
    """Frozen `fal/AuraFlow-v0.3` field queried in clean-time coordinates.

    AuraFlow uses model time 1 for noise and 0 for image. FieldScope exposes the
    opposite clean-time coordinate and therefore returns the negated native
    transformer output.
    """

    def __init__(self, config: BackendConfig):
        try:
            from diffusers import AuraFlowPipeline
        except ImportError as error:
            raise ImportError(
                "AuraFlow requires the `backbone` optional dependencies: "
                "pip install -e '.[backbone]'"
            ) from error

        self.config = config
        self._device = torch.device(config.device)
        self._dtype = _DTYPES[config.dtype]
        self.pipe = AuraFlowPipeline.from_pretrained(
            config.model_path,
            torch_dtype=self._dtype,
            variant=config.variant,
            use_safetensors=True,
            local_files_only=config.local_files_only,
        )
        self.pipe.to(self._device)
        self.pipe.transformer.eval().requires_grad_(False)
        self.pipe.vae.eval().requires_grad_(False)
        self.pipe.text_encoder.eval().requires_grad_(False)
        self._prompt_embeds = self._encode_prompt(config.prompt)
        if config.offload_text_encoder:
            self.pipe.text_encoder.to("cpu")
            if self._device.type == "cuda":
                torch.cuda.empty_cache()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def model_id(self) -> str:
        return str(self.config.model_path)

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        prompt_embeds, _, _, _ = self.pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            device=self.device,
            max_sequence_length=self.config.max_sequence_length,
        )
        return prompt_embeds.detach()

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        size = self.config.image_size
        images = F.interpolate(
            images.to(device=self.device, dtype=torch.float32),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        images = images.clamp(0, 1) * 2.0 - 1.0
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        posterior = self.pipe.vae.encode(images.to(dtype=vae_dtype)).latent_dist
        latents = posterior.mode() * self.pipe.vae.config.scaling_factor
        return latents.to(dtype=self.dtype)

    @torch.no_grad()
    def query_velocity(self, latents: torch.Tensor, clean_time: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)
        batch = latents.shape[0]
        if clean_time.ndim == 0:
            clean_time = clean_time.expand(batch)
        if clean_time.shape != (batch,):
            raise ValueError("clean_time must be scalar or shape [B]")
        native_time = 1.0 - clean_time.to(device=self.device, dtype=self.dtype)
        prompt_embeds = self._prompt_embeds
        if prompt_embeds.shape[0] != batch:
            prompt_embeds = prompt_embeds.expand(batch, -1, -1)
        native_velocity = self.pipe.transformer(
            latents,
            encoder_hidden_states=prompt_embeds,
            timestep=native_time,
            return_dict=False,
        )[0]
        return -native_velocity

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "auraflow",
            "model_id": self.model_id,
            "variant": self.config.variant,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "image_size": self.config.image_size,
            "prompt": self.config.prompt,
            "native_time": "1=noise, 0=image",
            "public_time": "clean_time: 0=noise, 1=image",
            "velocity_conversion": "public_velocity=-native_transformer_output",
            "frozen": True,
        }
