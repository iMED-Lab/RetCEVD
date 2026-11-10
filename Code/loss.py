from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEGroup(nn.Module):
    """Learned group-wise lower-bound critic used by RetCEVD."""

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int = 16):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(x_dim + y_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

    def forward(self, anchors: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size, memory_size = anchors.shape[0], memory.shape[0]
        anchor_grid = anchors.unsqueeze(1).expand(batch_size, memory_size, -1)
        memory_grid = memory.unsqueeze(0).expand(batch_size, memory_size, -1)
        scores = self.critic(torch.cat([anchor_grid, memory_grid], dim=-1))
        return scores.mean() - (scores.logsumexp(dim=1).mean() - np.log(memory_size))

    def learning_loss(self, anchors: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return -self.forward(anchors, memory)


def fit_info_nce(
    estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    anchors: torch.Tensor,
    memory: torch.Tensor,
) -> torch.Tensor:
    module = estimator.module if hasattr(estimator, "module") else estimator
    optimizer.zero_grad(set_to_none=True)
    loss = module.learning_loss(anchors.detach(), memory.detach())
    loss.backward()
    optimizer.step()
    return loss.detach()


@dataclass(frozen=True)
class PairingStats:
    unique_ratio: float
    mean_similarity: float
    max_reuse: int


class PairwiseCLUB(nn.Module):
    """Gaussian CLUB estimator for explicit cross-class pairs."""

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int = 16):
        super().__init__()
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh(),
        )

    def _distribution_parameters(self, anchors: torch.Tensor):
        return self.p_mu(anchors), self.p_logvar(anchors)

    @staticmethod
    def _log_prob(mu, logvar, targets, include_logvar: bool):
        value = (targets - mu).pow(2) * torch.exp(-logvar)
        if include_logvar:
            value = value + logvar
        return -0.5 * value.sum(dim=-1)

    def learning_loss(self, anchors: torch.Tensor, paired_targets: torch.Tensor):
        mu, logvar = self._distribution_parameters(anchors)
        return -self._log_prob(mu, logvar, paired_targets, True).mean()

    def forward(self, anchors: torch.Tensor, paired_targets: torch.Tensor):
        if anchors.shape[0] < 2:
            return anchors.sum() * 0.0
        mu, logvar = self._distribution_parameters(anchors)
        paired = self._log_prob(mu, logvar, paired_targets, False)
        shift = int(
            torch.randint(1, anchors.shape[0], (1,), device=anchors.device).item()
        )
        shuffled = self._log_prob(mu, logvar, paired_targets.roll(shift, 0), False)
        return (paired - shuffled).mean()


@torch.no_grad()
def build_cross_class_pairs(
    anchors: torch.Tensor,
    other_class_memory: torch.Tensor,
    topk: int = 16,
    temperature: float = 0.1,
    max_reuse: int = 1,
):
    normalized_anchors = F.normalize(anchors.detach(), dim=-1)
    memory = other_class_memory.detach()
    similarities = normalized_anchors @ F.normalize(memory, dim=-1).T
    candidate_count = min(topk, memory.shape[0])
    candidate_indices = similarities.topk(candidate_count, dim=1).indices
    use_counts = torch.zeros(memory.shape[0], dtype=torch.long, device=anchors.device)
    chosen = torch.empty(anchors.shape[0], dtype=torch.long, device=anchors.device)

    for anchor_index_tensor in similarities.max(dim=1).values.argsort(descending=True):
        anchor_index = int(anchor_index_tensor.item())
        candidates = candidate_indices[anchor_index]
        eligible = candidates[use_counts[candidates] < max_reuse]
        if eligible.numel() == 0:
            eligible = torch.where(use_counts < max_reuse)[0]
        if eligible.numel() == 0:
            eligible = torch.where(use_counts == use_counts.min())[0]
        probabilities = torch.softmax(similarities[anchor_index, eligible] / temperature, 0)
        selected = eligible[torch.multinomial(probabilities, 1)].squeeze(0)
        chosen[anchor_index] = selected
        use_counts[selected] += 1

    selected_similarity = similarities.gather(1, chosen[:, None]).squeeze(1)
    return memory.index_select(0, chosen), PairingStats(
        unique_ratio=float(chosen.unique().numel() / max(chosen.numel(), 1)),
        mean_similarity=float(selected_similarity.mean().item()),
        max_reuse=int(use_counts.max().item()),
    )


def fit_pairwise_club(
    estimator: nn.Module,
    optimizer: torch.optim.Optimizer,
    positive_anchors: torch.Tensor,
    negative_anchors: torch.Tensor,
    targets_for_positive: torch.Tensor,
    targets_for_negative: torch.Tensor,
    estimator_steps: int = 1,
):
    module = estimator.module if hasattr(estimator, "module") else estimator
    loss = positive_anchors.new_zeros(())
    for _ in range(estimator_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = module.learning_loss(
            positive_anchors.detach(), targets_for_positive.detach()
        ) + module.learning_loss(
            negative_anchors.detach(), targets_for_negative.detach()
        )
        loss.backward()
        optimizer.step()
    return loss.detach()
