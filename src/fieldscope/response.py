"""End-to-end randomized field-response extraction."""

from __future__ import annotations

from dataclasses import asdict

import torch

from fieldscope.config import ProbeConfig
from fieldscope.contracts import FieldBackend, FieldFeatures
from fieldscope.graph import (
    cosine_affinity,
    fixed_gaussian_sketch,
    patch_pool,
    pooled_attention_affinity,
    sparsify_affinity,
)
from fieldscope.path import endpoint_estimate, rectified_state, rectified_tangent
from fieldscope.probes import generate_probes


class FieldResponseExtractor:
    """Extract state, randomized response sketches, and a relational graph."""

    def __init__(self, backend: FieldBackend, config: ProbeConfig):
        config.validate()
        self.backend = backend
        self.config = config

    def _query_chunks(self, states: torch.Tensor, clean_time: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        chunk_size = self.config.probe_batch_size
        for start in range(0, states.shape[0], chunk_size):
            stop = min(states.shape[0], start + chunk_size)
            outputs.append(self.backend.query_velocity(states[start:stop], clean_time[start:stop]))
        return torch.cat(outputs, dim=0)

    def _responses(
        self,
        state: torch.Tensor,
        clean_time: torch.Tensor,
        base_velocity: torch.Tensor,
        seed: int,
    ) -> torch.Tensor:
        batch, channels, height, width = state.shape
        directions = generate_probes(
            state,
            self.config.num_directions,
            seed,
            self.config.probe_type,
        )
        latent_scale = state.float().flatten(1).std(dim=1).clamp_min(1e-3)
        scale = (self.config.eta * latent_scale).to(state.dtype).reshape(batch, 1, 1, 1, 1)
        plus = state[:, None] + scale * directions
        flat_plus = plus.reshape(-1, channels, height, width)
        flat_time = clean_time.repeat_interleave(self.config.num_directions)
        velocity_plus = self._query_chunks(flat_plus, flat_time).reshape(
            batch, self.config.num_directions, channels, height, width
        )
        if self.config.difference == "forward":
            return (velocity_plus - base_velocity[:, None]) / scale

        minus = state[:, None] - scale * directions
        velocity_minus = self._query_chunks(
            minus.reshape(-1, channels, height, width), flat_time
        ).reshape(batch, self.config.num_directions, channels, height, width)
        return (velocity_plus - velocity_minus) / (2.0 * scale)

    @torch.no_grad()
    def extract(
        self,
        images: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
    ) -> FieldFeatures:
        z0 = self.backend.encode_images(images)
        generator = torch.Generator(device=z0.device)
        generator.manual_seed(self.config.seed)
        if noise is None:
            noise = torch.randn(z0.shape, generator=generator, device=z0.device, dtype=z0.dtype)
        else:
            noise = noise.to(device=z0.device, dtype=z0.dtype)
        if noise.shape != z0.shape:
            raise ValueError("noise shape must match encoded image latents")

        noise_views = [noise, -noise] if self.config.antithetic_noise else [noise]
        view_states: list[torch.Tensor] = []
        view_responses: list[torch.Tensor] = []
        view_affinities: list[torch.Tensor] = []
        view_graphs: dict[str, list[torch.Tensor]] = {}
        view_baselines: dict[str, list[torch.Tensor]] = {
            "z0": [],
            "zt": [],
            "velocity": [],
            "mismatch": [],
            "endpoint": [],
        }

        for path_noise in noise_views:
            tangent = rectified_tangent(z0, path_noise)
            time_states: list[torch.Tensor] = []
            time_responses: list[torch.Tensor] = []
            time_affinities: list[torch.Tensor] = []
            time_graphs: dict[str, list[torch.Tensor]] = {
                name: [] for name in view_graphs
            }
            time_baselines: dict[str, list[torch.Tensor]] = {
                name: [] for name in view_baselines
            }

            for time_index, clean_time_value in enumerate(self.config.times):
                clean_time = torch.full(
                    (z0.shape[0],),
                    clean_time_value,
                    device=z0.device,
                    dtype=z0.dtype,
                )
                state = rectified_state(z0, path_noise, clean_time)
                feature_query = getattr(
                    self.backend,
                    "query_velocity_features",
                    None,
                )
                if callable(feature_query):
                    velocity, auxiliary = feature_query(state, clean_time)
                else:
                    velocity = self.backend.query_velocity(state, clean_time)
                    auxiliary = {}
                mismatch = velocity - tangent
                endpoint = endpoint_estimate(state, velocity, clean_time)
                endpoint_residual = endpoint - z0

                velocity_norm = velocity.square().sum(dim=1, keepdim=True).sqrt()
                mismatch_norm = mismatch.square().sum(dim=1, keepdim=True).sqrt()
                state_channels = torch.cat(
                    [velocity, mismatch, endpoint_residual, velocity_norm, mismatch_norm],
                    dim=1,
                )
                pooled_state = patch_pool(state_channels, self.config.graph_grid)
                responses = self._responses(
                    state,
                    clean_time,
                    velocity,
                    seed=self.config.seed + time_index * 1009,
                )
                batch, directions, channels, height, width = responses.shape
                pooled_response = patch_pool(
                    responses.reshape(batch, directions * channels, height, width),
                    self.config.graph_grid,
                )

                time_states.append(pooled_state)
                time_responses.append(pooled_response)
                time_affinities.append(cosine_affinity(pooled_response))
                if "dit_hidden" in auxiliary:
                    time_baselines.setdefault("dit_hidden", []).append(
                        fixed_gaussian_sketch(
                            patch_pool(
                                auxiliary["dit_hidden"],
                                self.config.graph_grid,
                            ),
                            self.config.hidden_baseline_dim,
                        )
                    )
                if {
                    "dit_attention_q",
                    "dit_attention_k",
                }.issubset(auxiliary):
                    time_graphs.setdefault("dit_attention", []).append(
                        pooled_attention_affinity(
                            auxiliary["dit_attention_q"],
                            auxiliary["dit_attention_k"],
                            self.config.graph_grid,
                        )
                    )
                for name, tensor in {
                    "z0": z0,
                    "zt": state,
                    "velocity": velocity,
                    "mismatch": mismatch,
                    "endpoint": endpoint,
                }.items():
                    time_baselines[name].append(patch_pool(tensor, self.config.graph_grid))

            view_states.append(torch.stack(time_states, dim=0).mean(dim=0))
            view_responses.append(torch.cat(time_responses, dim=-1))
            view_affinities.append(torch.stack(time_affinities, dim=0).mean(dim=0))
            for name, tensors in time_baselines.items():
                if not tensors:
                    continue
                view_baselines.setdefault(name, []).append(
                    torch.stack(tensors, dim=0).mean(dim=0)
                )
            view_baselines.setdefault("trajectory", []).append(
                torch.cat(time_baselines["zt"], dim=-1)
            )
            for name, tensors in time_graphs.items():
                view_graphs.setdefault(name, []).append(
                    torch.stack(tensors, dim=0).mean(dim=0)
                )

        state_features = torch.stack(view_states, dim=0).mean(dim=0)
        response_features = torch.cat(view_responses, dim=-1)
        affinity = torch.stack(view_affinities, dim=0).mean(dim=0)
        adjacency = sparsify_affinity(
            affinity,
            self.config.graph_grid,
            self.config.topk,
            self.config.local_radius,
        )
        baselines = {
            name: torch.stack(tensors, dim=0).mean(dim=0)
            for name, tensors in view_baselines.items()
        }
        graphs = {
            name: torch.stack(tensors, dim=0).mean(dim=0)
            for name, tensors in view_graphs.items()
        }
        if "dit_attention" in graphs:
            graphs["dit_attention_adjacency"] = sparsify_affinity(
                graphs["dit_attention"],
                self.config.graph_grid,
                self.config.topk,
                self.config.local_radius,
            )
        features = FieldFeatures(
            state=state_features,
            response=response_features,
            affinity=affinity,
            adjacency=adjacency,
            grid_size=self.config.graph_grid,
            baselines=baselines,
            graphs=graphs,
            metadata={
                "backend": self.backend.describe(),
                "probe": asdict(self.config),
                "time_convention": "clean_time: 0=noise, 1=image",
                "noise_views": len(noise_views),
            },
        )
        features.validate()
        return features
