"""Bilevel semantic reweighting for curvature-aware part tokens.

The weighting policy is optimized only by a post-update query loss.  During the
real model update its weights are detached, so the task/alignment losses cannot
train the policy through the direct weighted-error shortcut.

The lower-level variable is a small shared semantic adapter rather than the
individual token features.  Consequently, a support-view update can be judged
on a separately augmented query view.
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenBridge(nn.Module):
    """Produce P distinct, view-consistent semantic tokens.

    The visual branch is used by both training and inference.  The text branch
    is a training-only semantic teacher.  A shared low-rank part basis keeps the
    parameter count modest and gives every semantic part a stable identity.
    """

    def __init__(self, dim: int, text_dim: int, num_parts: int, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.num_parts = num_parts
        self.rank = rank

        self.anchors = nn.Parameter(torch.empty(1, num_parts, dim))
        self.part_basis = nn.Parameter(torch.empty(num_parts, rank, dim))
        self.visual_context = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, rank),
            nn.Tanh(),
        )
        self.text_context = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, rank),
            nn.Tanh(),
        )
        self.visual_scale = nn.Parameter(torch.tensor(0.1))
        self.text_scale = nn.Parameter(torch.tensor(0.1))

        nn.init.trunc_normal_(self.anchors, std=0.02)
        nn.init.trunc_normal_(self.part_basis, std=0.02)

    def _tokens(self, context: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        delta = torch.einsum("br,prc->bpc", context, self.part_basis)
        return self.anchors + torch.tanh(scale) * delta

    def from_visual(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self._tokens(self.visual_context(cls_feature), self.visual_scale)

    def from_text(self, text_feature: torch.Tensor) -> torch.Tensor:
        return self._tokens(self.text_context(text_feature.float()), self.text_scale)

    @staticmethod
    def _diversity_loss(tokens: torch.Tensor) -> torch.Tensor:
        # Penalize off-diagonal cosine similarity without forcing a particular
        # semantic basis.  This prevents all P semantic tokens from collapsing.
        proto = F.normalize(tokens.mean(dim=0), dim=-1)
        gram = proto @ proto.transpose(0, 1)
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return ((gram - eye) ** 2).sum() / max(gram.numel() - gram.size(0), 1)

    def consistency_loss(
        self,
        visual_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        diversity_weight: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        diversity = self._diversity_loss(visual_tokens)
        if text_tokens is None:
            distill = visual_tokens.new_zeros(())
        else:
            # Symmetric optimization learns a common visual/text semantic space.
            distill = (
                1.0
                - F.cosine_similarity(visual_tokens, text_tokens, dim=-1)
            ).mean()
        total = distill + float(diversity_weight) * diversity
        return total, {
            "semantic_distill_loss": distill.detach(),
            "semantic_diversity_loss": diversity.detach(),
        }


class LowRankSemanticAdapter(nn.Module):
    """Small shared lower-level variable used by the virtual update."""

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.trunc_normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.trunc_normal_(self.up.weight, std=0.02)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.gelu(self.down(self.norm(x))))

    def functional_forward(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        h = F.layer_norm(
            x,
            (self.dim,),
            params["norm.weight"],
            params["norm.bias"],
            self.norm.eps,
        )
        h = F.linear(h, params["down.weight"], params["down.bias"])
        h = F.gelu(h)
        h = F.linear(h, params["up.weight"], params["up.bias"])
        return x + h


class CurvatureSemanticWeightPolicy(nn.Module):
    """Predict an update policy from visual, semantic and HVP signals."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        tau: float = 1.0,
        reference_mix: float = 0.5,
    ):
        super().__init__()
        if tau <= 0:
            raise ValueError("meta policy temperature must be positive")
        if not 0.0 <= reference_mix <= 1.0:
            raise ValueError("reference_mix must be in [0, 1]")
        self.tau = float(tau)
        self.reference_mix = float(reference_mix)
        self.net = nn.Sequential(
            nn.LayerNorm(4 * dim + 2),
            nn.Linear(4 * dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        visual = F.normalize(visual_tokens, dim=-1)
        semantic = F.normalize(semantic_tokens, dim=-1)
        curvature = curvature.squeeze(-1) if curvature.dim() == 3 else curvature
        curvature = curvature / curvature.mean(dim=1, keepdim=True).clamp_min(1e-6)
        cosine = (visual * semantic).sum(dim=-1, keepdim=True)
        features = torch.cat(
            (
                visual,
                semantic,
                torch.abs(visual - semantic),
                visual * semantic,
                torch.log1p(curvature).unsqueeze(-1),
                cosine,
            ),
            dim=-1,
        )
        logits = self.net(features).squeeze(-1)
        learned = torch.softmax(logits / self.tau, dim=1)
        reference = torch.full_like(learned, 1.0 / learned.size(1))
        # p_i >= (1-rho)/P: no semantic part can be completely excluded.
        p = (1.0 - self.reference_mix) * reference + self.reference_mix * learned
        entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1)
        return p, {
            "policy_entropy": entropy.mean().detach(),
            "policy_effective_parts": entropy.exp().mean().detach(),
            "policy_max": p.max(dim=1).values.mean().detach(),
        }


class BilevelSemanticController(nn.Module):
    """One-step differentiable bilevel semantic reweighting controller."""

    def __init__(
        self,
        dim: int,
        text_dim: int,
        num_parts: int,
        semantic_rank: int = 64,
        adapter_rank: int = 64,
        policy_hidden_dim: int = 128,
        policy_tau: float = 1.0,
        reference_mix: float = 0.5,
        diversity_weight: float = 0.01,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.diversity_weight = float(diversity_weight)
        self.bridge = SemanticTokenBridge(dim, text_dim, num_parts, semantic_rank)
        self.adapter = LowRankSemanticAdapter(dim, adapter_rank)
        self.policy = CurvatureSemanticWeightPolicy(
            dim, policy_hidden_dim, policy_tau, reference_mix
        )

    def visual_semantics(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self.bridge.from_visual(cls_feature)

    def text_semantics(self, text_feature: torch.Tensor) -> torch.Tensor:
        return self.bridge.from_text(text_feature)

    def bridge_loss(
        self,
        visual_semantics: torch.Tensor,
        text_semantics: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.bridge.consistency_loss(
            visual_semantics, text_semantics, self.diversity_weight
        )

    @staticmethod
    def make_state(
        part_tokens: torch.Tensor,
        policy_semantics: torch.Tensor,
        target_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "part_tokens": part_tokens,
            "policy_semantics": policy_semantics,
            "target_semantics": target_semantics,
            "curvature": curvature,
        }

    def _adapter_parameters(self) -> OrderedDict:
        return OrderedDict(self.adapter.named_parameters())

    def alignment_error(
        self,
        visual_tokens: torch.Tensor,
        target_semantics: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        adapted = self.adapt_parts(visual_tokens, params)
        target = target_semantics.to(dtype=adapted.dtype)
        return 1.0 - F.cosine_similarity(adapted, target, dim=-1)

    def adapt_parts(
        self,
        part_tokens: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Apply the real lower-level variable used by training and inference."""
        adapter_dtype = next(self.adapter.parameters()).dtype
        tokens = part_tokens.to(dtype=adapter_dtype)
        if params is None:
            return self.adapter(tokens)
        return self.adapter.functional_forward(tokens, params)

    @staticmethod
    def reference_distribution(
        error: torch.Tensor,
        curvature: torch.Tensor,
        q_mode: str,
    ) -> torch.Tensor:
        """Build a stopped evaluator distribution independent of the policy."""
        if q_mode == "uniform":
            q = torch.full_like(error, 1.0 / error.size(1))
        elif q_mode == "hvp":
            q = curvature.squeeze(-1) if curvature.dim() == 3 else curvature
            q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            raise ValueError("meta q_mode must be 'uniform' or 'hvp'")
        return q.detach()

    def policy_distribution(
        self,
        part_tokens: torch.Tensor,
        policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Inputs are stopped while phi remains differentiable.  This makes p a
        # gradient selector instead of adding e_i * d p_i / d x to the inner step.
        policy_dtype = next(self.policy.parameters()).dtype
        return self.policy(
            part_tokens.detach().to(dtype=policy_dtype),
            policy_semantics.detach().to(dtype=policy_dtype),
            curvature.detach().to(dtype=policy_dtype),
        )

    def pool_parts(
        self,
        part_tokens: torch.Tensor,
        policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, stats = self.policy_distribution(
            part_tokens, policy_semantics, curvature
        )
        # The same adapter is used by the virtual inner update, the real model
        # update and inference.  Without this link, a better virtual alignment
        # would not imply a better downstream representation.
        adapted = self.adapt_parts(part_tokens)
        pooled = (p.detach().unsqueeze(-1) * adapted).sum(dim=1)
        return pooled, stats

    def meta_objective(
        self,
        support: Dict[str, torch.Tensor],
        query: Dict[str, torch.Tensor],
        inner_lr: float,
        q_mode: str = "uniform",
        kl_weight: float = 0.01,
        outer_task_fn: Optional[
            Callable[[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor], torch.Tensor]
        ] = None,
        task_weight: float = 1.0,
        semantic_weight: float = 0.1,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return a hypergradient-bearing task-feedback outer objective.

        ``p_phi`` appears only in the support-view inner update.  The query
        evaluator uses a fixed ``q`` and an optional downstream task callback,
        so the policy cannot minimize the original error by direct weighting.
        """
        support_tokens = support["part_tokens"].detach().float()
        support_policy_sem = support["policy_semantics"].detach().float()
        support_target_sem = support["target_semantics"].detach().float()
        support_curvature = support["curvature"].detach().float()

        query_tokens = query["part_tokens"].detach().float()
        query_target_sem = query["target_semantics"].detach().float()
        query_curvature = query["curvature"].detach().float()

        p, policy_stats = self.policy(
            support_tokens, support_policy_sem, support_curvature
        )
        base_params = self._adapter_parameters()
        support_error = self.alignment_error(
            support_tokens, support_target_sem, base_params
        )
        # Multiplication by P keeps the inner gradient scale comparable to a mean.
        inner_loss = (self.num_parts * p * support_error).mean()
        inner_grads = torch.autograd.grad(
            inner_loss,
            tuple(base_params.values()),
            create_graph=True,
            allow_unused=False,
        )
        fast_params = OrderedDict(
            (name, param - float(inner_lr) * grad)
            for (name, param), grad in zip(base_params.items(), inner_grads)
        )

        query_error_before = self.alignment_error(
            query_tokens, query_target_sem, base_params
        )
        query_error_after = self.alignment_error(
            query_tokens, query_target_sem, fast_params
        )

        q = self.reference_distribution(
            query_error_after, query_curvature, q_mode
        )
        outer_align = (q * query_error_after).sum(dim=1).mean()

        outer_task = outer_align.new_zeros(())
        task_before = outer_align.new_zeros(())
        if outer_task_fn is not None:
            outer_task = outer_task_fn(query, fast_params, q)
            # Only a diagnostic baseline; it must not create another path into
            # the policy or retain a needless graph.
            with torch.no_grad():
                task_before = outer_task_fn(query, base_params, q).detach()

        uniform = torch.full_like(p, 1.0 / self.num_parts)
        policy_kl = (
            p * (p.clamp_min(1e-8).log() - uniform.log())
        ).sum(dim=1).mean()
        meta_loss = (
            float(task_weight) * outer_task
            + float(semantic_weight) * outer_align
            + float(kl_weight) * policy_kl
        )

        before = (q * query_error_before.detach()).sum(dim=1).mean()
        improvement = before - outer_align.detach()
        stats = {
            "meta_loss": meta_loss.detach(),
            "meta_outer_task": outer_task.detach(),
            "meta_task_improvement": task_before - outer_task.detach(),
            "meta_outer_align": outer_align.detach(),
            "meta_inner_align": inner_loss.detach(),
            "meta_improvement": improvement,
            "meta_policy_kl": policy_kl.detach(),
            **policy_stats,
        }
        return meta_loss, stats

    def real_weighted_alignment(
        self, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Model loss using the learned policy without a direct phi gradient."""
        p, stats = self.policy_distribution(
            state["part_tokens"],
            state["policy_semantics"],
            state["curvature"],
        )
        error = self.alignment_error(
            state["part_tokens"], state["target_semantics"]
        )
        loss = (self.num_parts * p.detach() * error).mean()
        return loss, {"meta_real_align": loss.detach(), **stats}
