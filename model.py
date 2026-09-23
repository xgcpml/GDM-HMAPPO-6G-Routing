from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Categorical


class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)
        )
        self.register_buffer("_identity_cache", torch.empty(0), persistent=False)

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        x = F.relu(self.input_proj(node_features))
        if (
            self._identity_cache.numel() == 0
            or self._identity_cache.size(0) != adjacency.size(-1)
            or self._identity_cache.device != adjacency.device
            or self._identity_cache.dtype != adjacency.dtype
        ):
            self._identity_cache = torch.eye(
                adjacency.size(-1),
                device=adjacency.device,
                dtype=adjacency.dtype,
            )
        adj = adjacency + self._identity_cache.unsqueeze(0)
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        for layer in self.layers:
            aggregated = torch.bmm(adj, x) / degree
            x = F.relu(layer(torch.cat([x, aggregated], dim=-1)))
        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int):
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        self.attention_heads = attention_heads
        self.head_dim = hidden_dim // attention_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        batch_size, num_nodes, hidden_dim = x.shape
        shape = (batch_size, num_nodes, self.attention_heads, self.head_dim)
        query = self.query(x).view(shape).transpose(1, 2)
        key = self.key(x).view(shape).transpose(1, 2)
        value = self.value(x).view(shape).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        identity = torch.eye(num_nodes, device=x.device, dtype=torch.bool).unsqueeze(0)
        edge_mask = (adjacency > 0) | identity
        edge_bias = torch.log1p(adjacency.clamp_min(0.0)).unsqueeze(1)
        scores = (scores + edge_bias).masked_fill(
            ~edge_mask.unsqueeze(1), torch.finfo(scores.dtype).min
        )
        weights = F.softmax(scores, dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(
            batch_size, num_nodes, hidden_dim
        )
        x = self.attention_norm(x + self.output(attended))
        return self.feed_forward_norm(x + self.feed_forward(x))


class GraphAttentionEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        attention_heads: int,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GraphAttentionLayer(hidden_dim, attention_heads)
            for _ in range(num_layers)
        )

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        x = F.relu(self.input_proj(node_features))
        for layer in self.layers:
            x = layer(x, adjacency)
        return x


class NodeFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_features: Tensor, adjacency: Tensor) -> Tensor:
        del adjacency
        return F.relu(self.input_proj(node_features))


class TransformerContextBlock(nn.Module):
    def __init__(self, hidden_dim: int, attention_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.feed_forward_norm = nn.LayerNorm(hidden_dim)
        self.capture_attention = False
        self.last_attention_weights: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        attended, weights = self.attention(
            x,
            x,
            x,
            need_weights=self.capture_attention,
            average_attn_weights=False,
        )
        if self.capture_attention and weights is not None:
            self.last_attention_weights = weights.detach().cpu()
        x = self.attention_norm(x + attended)
        feed_forward = self.feed_forward(x)
        return self.feed_forward_norm(x + feed_forward)


class ConditionalDiffusionAllocator(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int,
        alloc_dim: int,
        diffusion_steps: int,
        beta_start: float,
        beta_end: float,
        noise_scale: float,
        latent_clip: float,
    ):
        super().__init__()
        self.alloc_dim = alloc_dim
        self.diffusion_steps = diffusion_steps
        self.noise_scale = noise_scale
        self.sample_noise_scale = noise_scale
        self.latent_clip = latent_clip

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prior_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim),
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(alloc_dim, hidden_dim),
            nn.ReLU(),
        )
        self.time_embedding = nn.Embedding(diffusion_steps, hidden_dim)
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim),
        )

    def sample(self, condition: Tensor, deterministic: bool) -> tuple[Tensor, Dict[str, Tensor]]:
        batch_size = condition.size(0)
        prior_latent = self.prior_head(condition).clamp(-self.latent_clip, self.latent_clip)
        latent = prior_latent.clone()
        if not deterministic:
            latent = latent + torch.randn_like(latent) * self.sample_noise_scale
        latent = latent.clamp(-self.latent_clip, self.latent_clip)
        latent, alloc_refinement_norm, pred_noise_norm = self._reverse_diffusion(
            latent=latent,
            condition=condition,
            stochastic=not deterministic,
        )

        allocation = self.latent_to_allocation(latent)
        aux = {
            "final_latent": latent,
            "prior_allocation": self.latent_to_allocation(prior_latent),
            "alloc_refinement_norm": alloc_refinement_norm,
            "pred_noise_norm": pred_noise_norm,
        }
        return allocation, aux

    def sample_candidates(self, condition: Tensor, num_candidates: int) -> tuple[Tensor, Dict[str, Tensor]]:
        batch_size = condition.size(0)
        num_candidates = max(int(num_candidates), 1)
        prior_latent = self.prior_head(condition).clamp(-self.latent_clip, self.latent_clip)
        prior_allocation = self.latent_to_allocation(prior_latent)

        if num_candidates == 1:
            return prior_allocation.unsqueeze(1), {
                "alloc_refinement_norm": torch.zeros(batch_size, 1, device=condition.device, dtype=condition.dtype),
                "pred_noise_norm": torch.zeros(batch_size, 1, device=condition.device, dtype=condition.dtype),
            }

        denoised_candidates = num_candidates - 1
        offsets = self._build_candidate_offsets(
            count=denoised_candidates,
            device=condition.device,
            dtype=condition.dtype,
        )
        initial_latents = prior_latent.unsqueeze(1) + offsets.unsqueeze(0)
        initial_latents = initial_latents.clamp(-self.latent_clip, self.latent_clip)

        flat_latents = initial_latents.reshape(batch_size * denoised_candidates, self.alloc_dim)
        flat_condition = (
            condition.unsqueeze(1)
            .expand(-1, denoised_candidates, -1)
            .reshape(batch_size * denoised_candidates, condition.size(-1))
        )
        final_latents, alloc_refinement_norm, pred_noise_norm = self._reverse_diffusion(
            latent=flat_latents,
            condition=flat_condition,
            stochastic=False,
        )
        denoised_allocations = self.latent_to_allocation(final_latents).reshape(
            batch_size, denoised_candidates, self.alloc_dim
        )

        allocations = torch.cat([prior_allocation.unsqueeze(1), denoised_allocations], dim=1)
        return allocations, {
            "alloc_refinement_norm": torch.cat(
                [
                    torch.zeros(batch_size, 1, device=condition.device, dtype=condition.dtype),
                    alloc_refinement_norm.reshape(batch_size, denoised_candidates),
                ],
                dim=1,
            ),
            "pred_noise_norm": torch.cat(
                [
                    torch.zeros(batch_size, 1, device=condition.device, dtype=condition.dtype),
                    pred_noise_norm.reshape(batch_size, denoised_candidates),
                ],
                dim=1,
            ),
        }

    def training_loss(
        self,
        condition: Tensor,
        target_allocation: Tensor,
        advantage: Tensor,
        eta: float,
        wmin: float,
        wmax: float,
        recon_coef: float,
        prior_coef: float,
        sample_weight: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        x0 = self.allocation_to_latent(target_allocation)
        batch_size = x0.size(0)
        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=condition.device)
        noise = torch.randn_like(x0)

        sqrt_alpha_bar = self.sqrt_alpha_bars[timestep].unsqueeze(-1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[timestep].unsqueeze(-1)
        noisy_latent = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        pred_noise = self.predict_noise(noisy_latent, timestep, condition)
        per_sample_noise_loss = ((pred_noise - noise) ** 2).mean(dim=-1)

        pred_x0 = (noisy_latent - sqrt_one_minus_alpha_bar * pred_noise) / sqrt_alpha_bar.clamp_min(1e-6)
        pred_x0 = pred_x0.clamp(-self.latent_clip, self.latent_clip)
        recon_allocation = self.latent_to_allocation(pred_x0)
        per_sample_recon_loss = ((recon_allocation - target_allocation) ** 2).mean(dim=-1)
        prior_allocation = self.latent_to_allocation(self.prior_head(condition).clamp(-self.latent_clip, self.latent_clip))
        per_sample_prior_loss = ((prior_allocation - target_allocation) ** 2).mean(dim=-1)

        weights = torch.clamp(torch.exp(eta * advantage.detach()), min=wmin, max=wmax)
        if sample_weight is not None:
            weights = weights * sample_weight.detach().clamp_min(0.0)

        combined_loss = per_sample_noise_loss + recon_coef * per_sample_recon_loss + prior_coef * per_sample_prior_loss
        normalizer = weights.sum().clamp_min(1e-6)
        diffusion_loss = (weights * combined_loss).sum() / normalizer

        return {
            "diffusion_loss": diffusion_loss,
            "noise_loss": per_sample_noise_loss.mean(),
            "recon_loss": per_sample_recon_loss.mean(),
            "prior_loss": per_sample_prior_loss.mean(),
            "adv_weight": weights.mean(),
            "pred_noise_norm": pred_noise.norm(dim=-1).mean(),
            "denoised_allocation_shift": (recon_allocation - target_allocation).norm(dim=-1).mean(),
        }

    def predict_noise(self, latent: Tensor, timestep: Tensor, condition: Tensor) -> Tensor:
        condition_hidden = self.condition_proj(condition)
        latent_hidden = self.noise_proj(latent)
        time_hidden = self.time_embedding(timestep)
        raw_noise = self.noise_predictor(torch.cat([condition_hidden, latent_hidden, time_hidden], dim=-1))
        return 2.0 * torch.tanh(raw_noise / 2.0)

    def _reverse_diffusion(
        self,
        latent: Tensor,
        condition: Tensor,
        stochastic: bool,
    ) -> tuple[Tensor, Tensor, Tensor]:
        denoised_norms = []
        pred_noise_norms = []

        for step in reversed(range(self.diffusion_steps)):
            step_ids = torch.full((latent.size(0),), step, dtype=torch.long, device=condition.device)
            pred_noise = self.predict_noise(latent, step_ids, condition)
            pred_noise_norms.append(pred_noise.norm(dim=-1))

            alpha_t = self.alphas[step]
            alpha_bar_t = self.alpha_bars[step]
            beta_t = self.betas[step]
            latent = (
                latent - (beta_t / torch.sqrt(1.0 - alpha_bar_t).clamp_min(1e-6)) * pred_noise
            ) / torch.sqrt(alpha_t).clamp_min(1e-6)

            if step > 0 and stochastic and self.sample_noise_scale > 0.0:
                anneal_ratio = min(
                    self.sample_noise_scale / max(self.noise_scale, 1e-6),
                    1.0,
                )
                latent = (
                    latent
                    + torch.randn_like(latent)
                    * torch.sqrt(beta_t)
                    * 0.35
                    * anneal_ratio
                )
            latent = latent.clamp(-self.latent_clip, self.latent_clip)
            denoised_norms.append(latent.norm(dim=-1))

        alloc_refinement_norm = torch.stack(denoised_norms, dim=1).mean(dim=1)
        pred_noise_norm = torch.stack(pred_noise_norms, dim=1).mean(dim=1)
        return latent, alloc_refinement_norm, pred_noise_norm

    def _build_candidate_offsets(self, count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if count <= 0:
            return torch.zeros(0, self.alloc_dim, device=device, dtype=dtype)

        zero_offset = torch.zeros(1, self.alloc_dim, device=device, dtype=dtype)
        if count == 1:
            return zero_offset

        direction_bank = self._direction_bank(device=device, dtype=dtype)
        required = count - 1
        repeats = (required + direction_bank.size(0) - 1) // direction_bank.size(0)
        tiled = direction_bank.repeat(repeats, 1)[:required]
        scale = torch.linspace(1.0, 0.55, steps=required, device=device, dtype=dtype).unsqueeze(-1)
        offsets = tiled * scale * self.noise_scale
        return torch.cat([zero_offset, offsets], dim=0)

    def _direction_bank(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        directions = []
        for src in range(self.alloc_dim):
            for dst in range(self.alloc_dim):
                if src == dst:
                    continue
                direction = torch.zeros(self.alloc_dim, device=device, dtype=dtype)
                direction[src] = 1.0
                direction[dst] = -1.0
                directions.append(direction)

        if not directions:
            return torch.zeros(1, self.alloc_dim, device=device, dtype=dtype)

        bank = torch.stack(directions, dim=0)
        return F.normalize(bank, dim=-1)

    def set_sample_noise_scale(self, noise_scale: float) -> None:
        self.sample_noise_scale = max(float(noise_scale), 0.0)

    def allocation_to_latent(self, allocation: Tensor) -> Tensor:
        logits = torch.log(allocation.clamp_min(1e-6))
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return logits.clamp(-self.latent_clip, self.latent_clip)

    def latent_to_allocation(self, latent: Tensor) -> Tensor:
        return F.softmax(latent, dim=-1)


class DirectAllocationHead(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int, alloc_dim: int, noise_scale: float):
        super().__init__()
        self.alloc_dim = alloc_dim
        self.noise_scale = noise_scale
        self.sample_noise_scale = noise_scale
        self.logit_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim),
        )

    def sample(self, condition: Tensor, deterministic: bool) -> tuple[Tensor, Dict[str, Tensor]]:
        logits = self.logit_head(condition)
        if not deterministic and self.sample_noise_scale > 0.0:
            logits = logits + torch.randn_like(logits) * self.sample_noise_scale
        allocation = F.softmax(logits, dim=-1)
        zeros = torch.zeros(condition.size(0), device=condition.device, dtype=condition.dtype)
        return allocation, {
            "alloc_refinement_norm": zeros,
            "pred_noise_norm": zeros,
        }

    def sample_candidates(self, condition: Tensor, num_candidates: int) -> tuple[Tensor, Dict[str, Tensor]]:
        num_candidates = max(int(num_candidates), 1)
        logits = self.logit_head(condition)
        base = F.softmax(logits, dim=-1)
        if num_candidates == 1:
            candidates = base.unsqueeze(1)
        else:
            offsets = self._build_candidate_offsets(
                count=num_candidates - 1,
                device=condition.device,
                dtype=condition.dtype,
            )
            perturbed = F.softmax(logits.unsqueeze(1) + offsets.unsqueeze(0), dim=-1)
            candidates = torch.cat([base.unsqueeze(1), perturbed], dim=1)
        zeros = torch.zeros(condition.size(0), candidates.size(1), device=condition.device, dtype=condition.dtype)
        return candidates, {
            "alloc_refinement_norm": zeros,
            "pred_noise_norm": zeros,
        }

    def training_loss(
        self,
        condition: Tensor,
        target_allocation: Tensor,
        advantage: Tensor,
        eta: float,
        wmin: float,
        wmax: float,
        recon_coef: float,
        prior_coef: float,
        sample_weight: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        del recon_coef, prior_coef
        allocation = F.softmax(self.logit_head(condition), dim=-1)
        per_sample_loss = ((allocation - target_allocation) ** 2).mean(dim=-1)
        weights = torch.clamp(torch.exp(eta * advantage.detach()), min=wmin, max=wmax)
        if sample_weight is not None:
            weights = weights * sample_weight.detach().clamp_min(0.0)
        normalizer = weights.sum().clamp_min(1e-6)
        allocation_loss = (weights * per_sample_loss).sum() / normalizer
        zero = allocation_loss.new_zeros(())
        return {
            "diffusion_loss": allocation_loss,
            "noise_loss": zero,
            "recon_loss": per_sample_loss.mean(),
            "prior_loss": zero,
            "adv_weight": weights.mean(),
            "pred_noise_norm": zero,
            "denoised_allocation_shift": (allocation - target_allocation).norm(dim=-1).mean(),
        }

    def _build_candidate_offsets(self, count: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if count <= 0:
            return torch.zeros(0, self.alloc_dim, device=device, dtype=dtype)
        direction_bank = self._direction_bank(device=device, dtype=dtype)
        repeats = (count + direction_bank.size(0) - 1) // direction_bank.size(0)
        tiled = direction_bank.repeat(repeats, 1)[:count]
        scale = torch.linspace(0.65, 0.25, steps=count, device=device, dtype=dtype).unsqueeze(-1)
        return tiled * scale * max(float(self.sample_noise_scale), 0.05)

    def _direction_bank(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        directions = []
        for src in range(self.alloc_dim):
            for dst in range(self.alloc_dim):
                if src == dst:
                    continue
                direction = torch.zeros(self.alloc_dim, device=device, dtype=dtype)
                direction[src] = 1.0
                direction[dst] = -1.0
                directions.append(direction)
        if not directions:
            return torch.zeros(1, self.alloc_dim, device=device, dtype=dtype)
        return F.normalize(torch.stack(directions, dim=0), dim=-1)

    def set_sample_noise_scale(self, noise_scale: float) -> None:
        self.sample_noise_scale = max(float(noise_scale), 0.0)


class GaussianAllocationHead(nn.Module):
    """Diagonal Gaussian latent policy followed by a logistic-normal map."""

    def __init__(self, condition_dim: int, hidden_dim: int, alloc_dim: int, noise_scale: float):
        super().__init__()
        self.alloc_dim = alloc_dim
        self.sample_noise_scale = noise_scale
        self.stats_head = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, alloc_dim * 2),
        )

    def _stats(self, condition: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = torch.chunk(self.stats_head(condition), 2, dim=-1)
        return mean, log_std.clamp(-4.5, 1.0)

    def sample(self, condition: Tensor, deterministic: bool) -> tuple[Tensor, Dict[str, Tensor]]:
        mean, log_std = self._stats(condition)
        latent = mean
        if not deterministic and self.sample_noise_scale > 0.0:
            latent = mean + torch.randn_like(mean) * log_std.exp() * self.sample_noise_scale
        allocation = F.softmax(latent, dim=-1)
        zeros = torch.zeros(condition.size(0), device=condition.device, dtype=condition.dtype)
        return allocation, {
            "alloc_refinement_norm": zeros,
            "pred_noise_norm": log_std.exp().norm(dim=-1),
        }

    def sample_candidates(self, condition: Tensor, num_candidates: int) -> tuple[Tensor, Dict[str, Tensor]]:
        mean, log_std = self._stats(condition)
        count = max(int(num_candidates), 1)
        if count == 1:
            latent = mean.unsqueeze(1)
        else:
            offsets = torch.linspace(-1.0, 1.0, count, device=mean.device, dtype=mean.dtype)
            latent = mean.unsqueeze(1) + offsets.view(1, count, 1) * log_std.exp().unsqueeze(1)
        allocations = F.softmax(latent, dim=-1)
        zeros = torch.zeros(condition.size(0), count, device=condition.device, dtype=condition.dtype)
        return allocations, {
            "alloc_refinement_norm": zeros,
            "pred_noise_norm": log_std.exp().norm(dim=-1, keepdim=True).expand(-1, count),
        }

    def training_loss(
        self,
        condition: Tensor,
        target_allocation: Tensor,
        advantage: Tensor,
        eta: float,
        wmin: float,
        wmax: float,
        recon_coef: float,
        prior_coef: float,
        sample_weight: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        del prior_coef
        mean, log_std = self._stats(condition)
        target_latent = torch.log(target_allocation.clamp_min(1e-6))
        target_latent = target_latent - target_latent.mean(dim=-1, keepdim=True)
        variance = torch.exp(2.0 * log_std).clamp_min(1e-6)
        per_sample_nll = 0.5 * (
            (target_latent - mean).square() / variance
            + 2.0 * log_std
            + torch.log(mean.new_tensor(2.0 * torch.pi))
        ).mean(dim=-1)
        reconstruction = F.softmax(mean, dim=-1)
        per_sample_recon = (reconstruction - target_allocation).square().mean(dim=-1)
        weights = torch.clamp(torch.exp(eta * advantage.detach()), min=wmin, max=wmax)
        if sample_weight is not None:
            weights = weights * sample_weight.detach().clamp_min(0.0)
        combined = per_sample_nll + recon_coef * per_sample_recon
        loss = (weights * combined).sum() / weights.sum().clamp_min(1e-6)
        zero = loss.new_zeros(())
        return {
            "diffusion_loss": loss,
            "noise_loss": per_sample_nll.mean(),
            "recon_loss": per_sample_recon.mean(),
            "prior_loss": zero,
            "adv_weight": weights.mean(),
            "pred_noise_norm": log_std.exp().norm(dim=-1).mean(),
            "denoised_allocation_shift": (
                reconstruction - target_allocation
            ).norm(dim=-1).mean(),
        }

    def set_sample_noise_scale(self, noise_scale: float) -> None:
        self.sample_noise_scale = max(float(noise_scale), 0.0)


class HybridRoutingPolicy(nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        task_feature_dim: int,
        candidate_feature_dim: int,
        hidden_dim: int,
        num_graph_layers: int,
        num_transformer_layers: int,
        attention_heads: int,
        alloc_dim: int,
        alloc_refinement_steps: int,
        alloc_noise_scale: float,
        diffusion_beta_start: float,
        diffusion_beta_end: float,
        diffusion_latent_clip: float,
        allocation_search_candidates: int,
        route_search_candidates: int,
        route_search_prior_coef: float,
        surrogate_penalty_coef: float,
        use_topology_route_prior: bool,
        use_search_guided_inference: bool,
        use_heuristic_search_candidates: bool = True,
        use_gnn_encoder: bool = True,
        use_graph_attention_encoder: bool = False,
        use_transformer_context: bool = True,
        use_global_graph_context: bool = True,
        use_candidate_resource_summaries: bool = True,
        use_gdm_allocator: bool = True,
        allocation_policy_family: str = "gdm",
    ):
        super().__init__()
        self.alloc_dim = alloc_dim
        self.allocation_search_candidates = max(int(allocation_search_candidates), 1)
        self.route_search_candidates = max(int(route_search_candidates), 1)
        self.route_search_prior_coef = route_search_prior_coef
        self.surrogate_penalty_coef = surrogate_penalty_coef
        self.use_topology_route_prior = bool(use_topology_route_prior)
        self.use_search_guided_inference = bool(use_search_guided_inference)
        self.use_heuristic_search_candidates = bool(use_heuristic_search_candidates)
        self.use_gnn_encoder = bool(use_gnn_encoder)
        self.use_graph_attention_encoder = bool(use_graph_attention_encoder)
        self.use_transformer_context = bool(use_transformer_context)
        self.use_global_graph_context = bool(use_global_graph_context)
        self.use_candidate_resource_summaries = bool(use_candidate_resource_summaries)
        family = allocation_policy_family.strip().lower()
        if not use_gdm_allocator and family == "gdm":
            family = "direct"
        if family not in {"gdm", "direct", "gaussian"}:
            raise ValueError(f"Unsupported allocation policy family: {family}")
        self.allocation_policy_family = family
        self.use_gdm_allocator = family == "gdm"
        self.current_topology_prior_scale = 1.0
        if not self.use_gnn_encoder:
            self.graph_encoder = NodeFeatureEncoder(node_feature_dim, hidden_dim)
        elif self.use_graph_attention_encoder:
            self.graph_encoder = GraphAttentionEncoder(
                node_feature_dim,
                hidden_dim,
                num_graph_layers,
                attention_heads,
            )
        else:
            self.graph_encoder = GraphEncoder(
                node_feature_dim, hidden_dim, num_graph_layers
            )
        self.transformer_blocks = nn.ModuleList(
            TransformerContextBlock(hidden_dim, attention_heads)
            for _ in range(max(int(num_transformer_layers), 0) if self.use_transformer_context else 0)
        )
        self.register_buffer(
            "topology_prior_weights",
            torch.tensor(
                [-1.40, 0.55, 0.30, 0.18, -0.52, -0.24, 0.22, 0.10, -0.18, -0.24],
                dtype=torch.float32,
            ),
        )
        self.register_buffer("topology_confidence_weight", torch.tensor(0.16, dtype=torch.float32))

        candidate_input_dim = hidden_dim * 3 + candidate_feature_dim + task_feature_dim
        self.route_head = nn.Sequential(
            nn.Linear(candidate_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        value_input_dim = hidden_dim * 2 + task_feature_dim
        self.value_head = nn.Sequential(
            nn.Linear(value_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        if family == "gdm":
            self.allocation_head = ConditionalDiffusionAllocator(
                condition_dim=candidate_input_dim,
                hidden_dim=hidden_dim,
                alloc_dim=alloc_dim,
                diffusion_steps=alloc_refinement_steps,
                beta_start=diffusion_beta_start,
                beta_end=diffusion_beta_end,
                noise_scale=alloc_noise_scale,
                latent_clip=diffusion_latent_clip,
            )
        elif family == "gaussian":
            self.allocation_head = GaussianAllocationHead(
                condition_dim=candidate_input_dim,
                hidden_dim=hidden_dim,
                alloc_dim=alloc_dim,
                noise_scale=alloc_noise_scale,
            )
        else:
            self.allocation_head = DirectAllocationHead(
                condition_dim=candidate_input_dim,
                hidden_dim=hidden_dim,
                alloc_dim=alloc_dim,
                noise_scale=alloc_noise_scale,
            )
        self.current_allocation_noise_scale = alloc_noise_scale
        self.current_route_sampling_temperature = 1.0

    def act(
        self,
        obs: Dict[str, Tensor],
        deterministic_allocation: bool = False,
        deterministic_route: bool = False,
    ) -> Dict[str, Tensor]:
        encoded = self._encode(obs)
        route_dist = self._route_distribution(encoded["route_logits"])
        if deterministic_route:
            route = encoded["route_logits"].argmax(dim=-1)
        else:
            route = route_dist.sample()
        condition = self._select_route_condition(encoded, route)
        allocation, alloc_aux = self.allocation_head.sample(condition, deterministic=deterministic_allocation)
        allocation = self._mask_allocation(obs, route, allocation)

        return {
            "route": route,
            "allocation": allocation,
            "route_log_prob": route_dist.log_prob(route),
            "route_entropy": route_dist.entropy(),
            "alloc_refinement_norm": alloc_aux["alloc_refinement_norm"],
            "pred_noise_norm": alloc_aux["pred_noise_norm"],
            "value": encoded["value"],
        }

    def set_allocation_noise_scale(self, noise_scale: float) -> None:
        self.current_allocation_noise_scale = max(float(noise_scale), 0.0)
        self.allocation_head.set_sample_noise_scale(self.current_allocation_noise_scale)

    def set_route_sampling_temperature(self, temperature: float) -> None:
        self.current_route_sampling_temperature = max(float(temperature), 0.05)

    def set_topology_prior_scale(self, scale: float) -> None:
        if not self.use_topology_route_prior:
            self.current_topology_prior_scale = 0.0
            return
        self.current_topology_prior_scale = max(float(scale), 0.0)

    def capture_attention_matrices(self, obs: Dict[str, Tensor]) -> list[Tensor]:
        if not self.transformer_blocks:
            raise ValueError("The policy has no Transformer attention blocks.")
        for block in self.transformer_blocks:
            block.capture_attention = True
            block.last_attention_weights = None
        try:
            self._encode(obs)
        finally:
            for block in self.transformer_blocks:
                block.capture_attention = False
        matrices = [block.last_attention_weights for block in self.transformer_blocks]
        if any(matrix is None for matrix in matrices):
            raise RuntimeError("Attention capture did not produce every layer matrix.")
        return [matrix for matrix in matrices if matrix is not None]

    def act_deterministic(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.use_search_guided_inference:
            return self.act(obs, deterministic_allocation=True, deterministic_route=True)
        return self.act_search_guided(obs, exhaustive_routes=False, use_policy_prior=True)

    def act_search_guided(
        self,
        obs: Dict[str, Tensor],
        exhaustive_routes: bool,
        use_policy_prior: bool,
    ) -> Dict[str, Tensor]:
        encoded = self._encode(obs)
        route_dist = self._route_distribution(encoded["route_logits"])
        route_log_probs = F.log_softmax(self._scaled_route_logits(encoded["route_logits"]), dim=-1)
        candidate_routes = None
        if exhaustive_routes:
            batch_size, num_routes = encoded["route_logits"].shape
            candidate_routes = torch.arange(num_routes, device=route_log_probs.device).unsqueeze(0).expand(batch_size, -1)
        route, allocation, search_aux = self._joint_route_allocation_search(
            obs=obs,
            encoded=encoded,
            route_log_probs=route_log_probs,
            candidate_routes=candidate_routes,
            use_route_prior=use_policy_prior,
        )

        return {
            "route": route,
            "allocation": allocation,
            "route_log_prob": route_dist.log_prob(route),
            "route_entropy": route_dist.entropy(),
            "alloc_refinement_norm": search_aux["alloc_refinement_norm"],
            "pred_noise_norm": search_aux["pred_noise_norm"],
            "allocation_search_score": search_aux["allocation_score"],
            "route_search_score": search_aux["joint_score"],
            "route_prior_cost": search_aux["route_prior_cost"],
            "per_route_joint_score": search_aux["per_route_joint_score"],
            "candidate_routes": search_aux["candidate_routes"],
            "value": encoded["value"],
        }

    def heuristic_guidance(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        encoded = self._encode(obs)
        route = encoded["route_prior_logits"].argmax(dim=-1)
        allocation = self._heuristic_allocation_candidates(obs, route)[:, 0]
        return {
            "route": route,
            "allocation": allocation,
            "route_prior_logits": encoded["route_prior_logits"],
        }

    def evaluate_route(self, obs: Dict[str, Tensor], route: Tensor) -> Dict[str, Tensor]:
        encoded = self._encode(obs)
        route_dist = self._route_distribution(encoded["route_logits"])
        return {
            "route_log_prob": route_dist.log_prob(route),
            "route_entropy": route_dist.entropy(),
            "policy_route_logits": encoded["policy_route_logits"],
            "route_prior_logits": encoded["route_prior_logits"],
            "route_logits": encoded["route_logits"],
            "value": encoded["value"],
            "route_condition": self._select_route_condition(encoded, route),
        }

    def value(self, obs: Dict[str, Tensor]) -> Tensor:
        return self._encode(obs)["value"]

    def diffusion_training_loss(
        self,
        obs: Dict[str, Tensor],
        route: Tensor,
        target_allocation: Tensor,
        advantage: Tensor,
        eta: float,
        wmin: float,
        wmax: float,
        recon_coef: float,
        prior_coef: float,
        sample_weight: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        encoded = self._encode(obs)
        condition = self._select_route_condition(encoded, route)
        return self.allocation_head.training_loss(
            condition=condition,
            target_allocation=target_allocation,
            advantage=advantage,
            eta=eta,
            wmin=wmin,
            wmax=wmax,
            recon_coef=recon_coef,
            prior_coef=prior_coef,
            sample_weight=sample_weight,
        )

    def evaluate_joint_action(
        self,
        obs: Dict[str, Tensor],
        route: Tensor,
        allocation: Tensor,
    ) -> Dict[str, Tensor]:
        encoded = self._encode(obs)
        route_log_probs = F.log_softmax(self._scaled_route_logits(encoded["route_logits"]), dim=-1)
        allocation_score = self._estimate_allocation_scores(obs, route, allocation.unsqueeze(1)).squeeze(1)
        route_prior_cost = -route_log_probs.gather(1, route.unsqueeze(-1)).squeeze(-1) * self.route_search_prior_coef
        return {
            "allocation_score": allocation_score,
            "route_prior_cost": route_prior_cost,
            "joint_score": allocation_score + route_prior_cost,
            "route_logits": encoded["route_logits"],
            "value": encoded["value"],
        }

    def topology_route_prior(self, obs: Dict[str, Tensor]) -> Tensor:
        prior_features = obs["candidate_features"][..., : self.topology_prior_weights.numel()]
        route_bias = (prior_features * self.topology_prior_weights.view(1, 1, -1)).sum(dim=-1)
        route_bias = route_bias + self.topology_confidence_weight * obs["candidate_mask"]
        route_bias = route_bias - route_bias.mean(dim=-1, keepdim=True)
        route_scale = route_bias.std(dim=-1, keepdim=True).clamp_min(1e-4)
        return torch.clamp(route_bias / route_scale, -3.0, 3.0)

    def _scaled_route_logits(self, route_logits: Tensor) -> Tensor:
        temperature = max(float(self.current_route_sampling_temperature), 0.05)
        return route_logits / temperature

    def _route_distribution(self, route_logits: Tensor) -> Categorical:
        return Categorical(logits=self._scaled_route_logits(route_logits))

    def _encode(self, obs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        node_embeddings = self.graph_encoder(obs["node_features"], obs["adjacency"])
        for block in self.transformer_blocks:
            node_embeddings = block(node_embeddings)

        global_context = node_embeddings.mean(dim=1)
        if not self.use_global_graph_context:
            global_context = torch.zeros_like(global_context)
        batch_indices = torch.arange(node_embeddings.size(0), device=node_embeddings.device)
        source_embeddings = node_embeddings[batch_indices, obs["source_node"]]
        path_embeddings = self._gather_path_embeddings(
            node_embeddings,
            obs["candidate_nodes"],
            obs["candidate_node_mask"],
        )

        task_context = obs["task_features"].unsqueeze(1).expand(-1, path_embeddings.size(1), -1)
        source_context = source_embeddings.unsqueeze(1).expand(
            -1, path_embeddings.size(1), -1
        )
        global_context_expanded = global_context.unsqueeze(1).expand(-1, path_embeddings.size(1), -1)
        candidate_features = obs["candidate_features"]
        if not self.use_candidate_resource_summaries:
            candidate_features = candidate_features.clone()
            candidate_features[..., [4, 5, 6, 7, 9, 12, 13]] = 0.0
        candidate_context = torch.cat(
            [
                path_embeddings,
                global_context_expanded,
                source_context,
                candidate_features,
                task_context,
            ],
            dim=-1,
        )
        route_head_logits = self.route_head(candidate_context).squeeze(-1)
        policy_route_logits = self._apply_route_mask(route_head_logits, obs["candidate_mask"])
        if self.use_topology_route_prior and self.current_topology_prior_scale > 0.0:
            route_prior_logits = self.topology_route_prior(obs)
            route_logits = policy_route_logits + self.current_topology_prior_scale * route_prior_logits
        else:
            route_prior_logits = torch.zeros_like(policy_route_logits)
            route_logits = policy_route_logits

        value = self.value_head(
            torch.cat([global_context, source_embeddings, obs["task_features"]], dim=-1)
        ).squeeze(-1)
        return {
            "candidate_context": candidate_context,
            "policy_route_logits": policy_route_logits,
            "route_prior_logits": route_prior_logits,
            "route_logits": route_logits,
            "value": value,
        }

    def _gather_path_embeddings(
        self,
        node_embeddings: Tensor,
        candidate_nodes: Tensor,
        candidate_node_mask: Tensor,
    ) -> Tensor:
        num_candidates = candidate_nodes.size(1)
        expanded_nodes = candidate_nodes.unsqueeze(-1).expand(-1, -1, -1, node_embeddings.size(-1))
        expanded_embeddings = node_embeddings.unsqueeze(1).expand(-1, num_candidates, -1, -1)
        path_embeddings = torch.gather(expanded_embeddings, dim=2, index=expanded_nodes)
        node_mask = candidate_node_mask.unsqueeze(-1)
        return (path_embeddings * node_mask).sum(dim=2) / node_mask.sum(dim=2).clamp_min(1.0)

    def _mask_allocation(
        self,
        obs: Dict[str, Tensor],
        route: Tensor,
        allocation: Tensor,
    ) -> Tensor:
        batch_indices = torch.arange(route.size(0), device=route.device)
        mask = obs["candidate_alloc_mask"][batch_indices, route]
        if allocation.ndim == 3:
            mask = mask.unsqueeze(1)
        masked = allocation * mask
        normalizer = masked.sum(dim=-1, keepdim=True)
        fallback = mask / mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.where(
            normalizer > 1e-8,
            masked / normalizer.clamp_min(1e-8),
            fallback,
        )

    def _select_route_condition(self, encoded: Dict[str, Tensor], route: Tensor) -> Tensor:
        batch_indices = torch.arange(route.size(0), device=route.device)
        return encoded["candidate_context"][batch_indices, route]

    def _apply_route_mask(self, route_logits: Tensor, candidate_mask: Tensor) -> Tensor:
        valid_routes = candidate_mask > 0.5
        if not bool(torch.all(valid_routes.any(dim=-1))):
            raise ValueError("Every task must expose at least one available candidate route.")
        return route_logits.masked_fill(~valid_routes, torch.finfo(route_logits.dtype).min)

    def _estimate_allocation_scores(
        self,
        obs: Dict[str, Tensor],
        route: Tensor,
        candidate_allocations: Tensor,
    ) -> Tensor:
        batch_indices = torch.arange(route.size(0), device=route.device)
        selected_features = obs["candidate_features"][batch_indices, route]
        selected_path_nodes = obs["candidate_nodes"][batch_indices, route]
        selected_nodes = obs["candidate_compute_nodes"][batch_indices, route]
        candidate_allocations = self._mask_allocation(
            obs,
            route,
            candidate_allocations,
        )

        node_feature_dim = obs["node_features"].size(-1)
        gather_index = selected_nodes.unsqueeze(-1).expand(-1, -1, node_feature_dim)
        selected_node_features = torch.gather(obs["node_features"], 1, gather_index)

        capacity = selected_node_features[:, :, 4].clamp_min(1e-3)
        queue_pressure = selected_node_features[:, :, 5].clamp_min(0.0)
        node_pressure = selected_node_features[:, :, 11].clamp_min(0.0)
        queue_delay = 0.55 * queue_pressure + 0.20 * node_pressure

        src_nodes = selected_path_nodes[:, :-1]
        dst_nodes = selected_path_nodes[:, 1:]
        selected_link_mask = obs["candidate_link_mask"][batch_indices, route]
        edge_batch_indices = batch_indices.unsqueeze(-1).expand_as(src_nodes)
        observed_link_rates = obs["adjacency"][edge_batch_indices, src_nodes, dst_nodes].clamp_min(1e-3)
        data_load = obs["task_features"][:, 1].unsqueeze(-1)
        path_confidence = obs["candidate_mask"][batch_indices, route].unsqueeze(-1).clamp_min(1e-3)
        propagation_delay = selected_features[:, 8].unsqueeze(-1)
        path_pressure = selected_features[:, 9].unsqueeze(-1)
        wireless_risk = (1.0 - selected_features[:, 1]).unsqueeze(-1)
        comm_latency = 0.18 * (
            (data_load / observed_link_rates) * selected_link_mask
        ).sum(dim=-1, keepdim=True)
        comm_latency = comm_latency + 0.55 * propagation_delay + 0.22 * wireless_risk
        comm_latency = comm_latency + 0.18 * (1.0 - path_confidence)
        deadline = obs["task_features"][:, 3].unsqueeze(-1).clamp_min(1e-4)
        compute_load = obs["task_features"][:, 2].view(-1, 1, 1)

        service_delay = 0.72 * candidate_allocations * compute_load / capacity.unsqueeze(1)
        stability_penalty = 0.10 * path_pressure + 0.06 * selected_features[:, 4].unsqueeze(-1)
        total_latency = comm_latency + (queue_delay.unsqueeze(1) + service_delay).sum(dim=-1) + stability_penalty
        violation = torch.relu(total_latency - deadline)
        return total_latency + self.surrogate_penalty_coef * violation

    def _heuristic_allocation_candidates(
        self,
        obs: Dict[str, Tensor],
        route: Tensor,
    ) -> Tensor:
        batch_indices = torch.arange(route.size(0), device=route.device)
        selected_nodes = obs["candidate_compute_nodes"][batch_indices, route]
        allocation_mask = obs["candidate_alloc_mask"][batch_indices, route]

        node_feature_dim = obs["node_features"].size(-1)
        gather_index = selected_nodes.unsqueeze(-1).expand(-1, -1, node_feature_dim)
        selected_node_features = torch.gather(obs["node_features"], 1, gather_index)

        capacity = selected_node_features[:, :, 4].clamp_min(1e-3)
        queue_pressure = selected_node_features[:, :, 5].clamp_min(0.0)
        node_pressure = selected_node_features[:, :, 11].clamp_min(0.0)

        path_bias = capacity.new_tensor([0.95, 0.95, 1.10]).unsqueeze(0)
        service_weights = (
            path_bias
            * capacity
            / (1.0 + 0.90 * queue_pressure + 0.55 * node_pressure)
            * allocation_mask
        )

        queue_averse_bias = capacity.new_tensor([1.00, 1.00, 0.92]).unsqueeze(0)
        queue_averse_weights = (
            queue_averse_bias
            * capacity
            / (1.0 + 1.20 * queue_pressure + 0.70 * node_pressure)
            * allocation_mask
        )

        service_weights = service_weights / service_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        queue_averse_weights = queue_averse_weights / queue_averse_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.stack([service_weights, queue_averse_weights], dim=1)

    def _joint_route_allocation_search(
        self,
        obs: Dict[str, Tensor],
        encoded: Dict[str, Tensor],
        route_log_probs: Tensor,
        candidate_routes: Tensor | None = None,
        use_route_prior: bool = True,
    ) -> tuple[Tensor, Tensor, Dict[str, Tensor]]:
        batch_size, num_routes = encoded["route_logits"].shape
        if candidate_routes is None:
            route_candidates = min(self.route_search_candidates, num_routes)
            top_routes = torch.topk(encoded["route_logits"], k=route_candidates, dim=-1).indices
            candidate_routes = top_routes
            if self.use_heuristic_search_candidates:
                masked_prior_logits = self._apply_route_mask(
                    encoded["route_prior_logits"],
                    obs["candidate_mask"],
                )
                heuristic_routes = masked_prior_logits.argmax(dim=-1, keepdim=True)
                candidate_routes = torch.cat([candidate_routes, heuristic_routes], dim=-1)
        candidate_route_valid = (
            obs["candidate_mask"].gather(1, candidate_routes) > 0.5
        )
        candidate_route_logits = encoded["route_logits"].gather(1, candidate_routes)
        route_candidates = candidate_routes.size(1)

        flat_top_routes = candidate_routes.reshape(-1)
        expanded_obs = self._repeat_observation_batch(obs, route_candidates)
        expanded_context = encoded["candidate_context"].unsqueeze(1).expand(-1, route_candidates, -1, -1)
        flat_context = expanded_context.reshape(batch_size * route_candidates, num_routes, -1)
        flat_encoded = {"candidate_context": flat_context}
        condition = self._select_route_condition(flat_encoded, flat_top_routes)

        candidate_allocations, candidate_aux = self.allocation_head.sample_candidates(
            condition,
            num_candidates=self.allocation_search_candidates,
        )
        if self.use_heuristic_search_candidates:
            heuristic_allocations = self._heuristic_allocation_candidates(
                expanded_obs, flat_top_routes
            )
            zero_search_stat = torch.zeros(
                heuristic_allocations.size(0),
                heuristic_allocations.size(1),
                device=heuristic_allocations.device,
                dtype=heuristic_allocations.dtype,
            )
            candidate_allocations = torch.cat(
                [heuristic_allocations, candidate_allocations], dim=1
            )
            candidate_aux["alloc_refinement_norm"] = torch.cat(
                [zero_search_stat, candidate_aux["alloc_refinement_norm"]],
                dim=1,
            )
            candidate_aux["pred_noise_norm"] = torch.cat(
                [zero_search_stat, candidate_aux["pred_noise_norm"]],
                dim=1,
            )
        candidate_allocations = self._mask_allocation(
            expanded_obs,
            flat_top_routes,
            candidate_allocations,
        )
        alloc_candidates = candidate_allocations.size(1)
        allocation_scores = self._estimate_allocation_scores(expanded_obs, flat_top_routes, candidate_allocations)

        if use_route_prior:
            route_prior_cost = -route_log_probs.gather(1, candidate_routes) * self.route_search_prior_coef
        else:
            route_prior_cost = torch.zeros(
                batch_size,
                route_candidates,
                device=allocation_scores.device,
                dtype=allocation_scores.dtype,
            )
        joint_scores = allocation_scores + route_prior_cost.reshape(-1, 1)

        joint_scores = joint_scores.reshape(batch_size, route_candidates, alloc_candidates)
        joint_scores = joint_scores.masked_fill(
            ~candidate_route_valid.unsqueeze(-1),
            torch.inf,
        )
        allocation_scores = allocation_scores.reshape(batch_size, route_candidates, alloc_candidates)
        candidate_allocations = candidate_allocations.reshape(
            batch_size, route_candidates, alloc_candidates, self.alloc_dim
        )
        alloc_refinement_norm = candidate_aux["alloc_refinement_norm"].reshape(
            batch_size, route_candidates, alloc_candidates
        )
        pred_noise_norm = candidate_aux["pred_noise_norm"].reshape(
            batch_size, route_candidates, alloc_candidates
        )
        per_route_joint_score, best_alloc_per_route = joint_scores.min(dim=-1)
        batch_indices = torch.arange(batch_size, device=candidate_routes.device)
        best_route_pos = per_route_joint_score.argmin(dim=-1)
        best_alloc_pos = best_alloc_per_route[batch_indices, best_route_pos]

        selected_routes = candidate_routes[batch_indices, best_route_pos]
        selected_allocations = candidate_allocations[batch_indices, best_route_pos, best_alloc_pos]

        return selected_routes, selected_allocations, {
            "joint_score": joint_scores[batch_indices, best_route_pos, best_alloc_pos],
            "allocation_score": allocation_scores[batch_indices, best_route_pos, best_alloc_pos],
            "route_prior_cost": route_prior_cost[batch_indices, best_route_pos],
            "per_route_joint_score": per_route_joint_score,
            "candidate_routes": candidate_routes,
            "alloc_refinement_norm": alloc_refinement_norm[batch_indices, best_route_pos, best_alloc_pos],
            "pred_noise_norm": pred_noise_norm[batch_indices, best_route_pos, best_alloc_pos],
            "searched_route_logit": candidate_route_logits[batch_indices, best_route_pos],
        }

    def _repeat_observation_batch(self, obs: Dict[str, Tensor], repeats: int) -> Dict[str, Tensor]:
        expanded: Dict[str, Tensor] = {}
        for key, value in obs.items():
            expanded[key] = value.repeat_interleave(repeats, dim=0)
        return expanded
