"""Curvature-aware semantic weighting with a differentiable bilevel loop.

V5 learns a policy over local semantic part tokens. V6 additionally builds
pairwise part relations (appearance + spatial layout), measures semantic
curvature on those relation tokens, and learns which relations should guide a
one-step support update by evaluating its effect on a separate query view.

Policy parameters ``phi`` are isolated from ordinary task losses: policy inputs
are stopped, and weights used by real losses and classification are detached.
Only the post-update outer objective updates ``phi``.
"""

import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticTokenBridge(nn.Module):
    """Produce P distinct, view-consistent semantic tokens."""

    def __init__(self, dim: int, text_dim: int, num_parts: int, rank: int = 64):
        super().__init__()
        self.dim = dim
        self.num_parts = num_parts
        self.anchors = nn.Parameter(torch.empty(1, num_parts, dim))
        self.part_basis = nn.Parameter(torch.empty(num_parts, rank, dim))
        self.visual_context = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, rank), nn.Tanh()
        )
        self.text_context = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, rank), nn.Tanh()
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
            distill = (
                1.0 - F.cosine_similarity(visual_tokens, text_tokens, dim=-1)
            ).mean()
        total = distill + float(diversity_weight) * diversity
        return total, {
            "semantic_distill_loss": distill.detach(),
            "semantic_diversity_loss": diversity.detach(),
        }


class LowRankSemanticAdapter(nn.Module):
    """Small shared lower-level variable used by a virtual update."""

    def __init__(self, dim: int, rank: int = 64):
        super().__init__()
        self.dim = dim
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
            x, (self.dim,), params["norm.weight"], params["norm.bias"],
            self.norm.eps,
        )
        h = F.gelu(F.linear(h, params["down.weight"], params["down.bias"]))
        h = F.linear(h, params["up.weight"], params["up.bias"])
        return x + h


class CurvatureSemanticWeightPolicy(nn.Module):
    """Predict an update policy from tokens, semantics and stopped HVP values."""

    def __init__(
        self, dim: int, hidden_dim: int = 128, tau: float = 1.0,
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
        # Uniform at initialization; no random part/edge preference.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, visual_tokens: torch.Tensor, semantic_tokens: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        visual = F.normalize(visual_tokens, dim=-1)
        semantic = F.normalize(semantic_tokens, dim=-1)
        curvature = curvature.squeeze(-1) if curvature.dim() == 3 else curvature
        curvature = curvature / curvature.mean(dim=1, keepdim=True).clamp_min(1e-6)
        cosine = (visual * semantic).sum(dim=-1, keepdim=True)
        features = torch.cat(
            (visual, semantic, torch.abs(visual - semantic), visual * semantic,
             torch.log1p(curvature.clamp_min(0)).unsqueeze(-1), cosine),
            dim=-1,
        )
        logits = self.net(features).squeeze(-1)
        learned = torch.softmax(logits / self.tau, dim=1)
        reference = torch.full_like(learned, 1.0 / learned.size(1))
        p = (1.0 - self.reference_mix) * reference + self.reference_mix * learned
        entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=1)
        return p, {
            "policy_entropy": entropy.mean().detach(),
            "policy_effective_items": entropy.exp().mean().detach(),
            "policy_max": p.max(dim=1).values.mean().detach(),
        }


class PairwiseRelationEncoder(nn.Module):
    """Encode all undirected part pairs as appearance-layout relations.

    With P=8 this produces 28 edges. Spatial descriptors are translation and
    horizontal-flip robust: absolute centroid displacement, distance, absolute
    spread difference, and soft-attention overlap.
    """

    def __init__(self, dim: int, num_parts: int):
        super().__init__()
        if num_parts < 2:
            raise ValueError("relation modeling needs at least two parts")
        self.dim = dim
        self.num_parts = num_parts
        self.register_buffer(
            "pair_index", torch.triu_indices(num_parts, num_parts, offset=1),
            persistent=False,
        )
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(2 * dim + 6), nn.Linear(2 * dim + 6, dim),
            nn.GELU(), nn.Linear(dim, dim),
        )
        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim),
            nn.GELU(), nn.Linear(dim, dim),
        )

    @property
    def num_relations(self) -> int:
        return int(self.pair_index.size(1))

    def _split_pairs(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            tokens.index_select(1, self.pair_index[0]),
            tokens.index_select(1, self.pair_index[1]),
        )

    @staticmethod
    def _patch_coordinates(
        num_patches: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        height = int(math.sqrt(num_patches))
        if height * height != num_patches:
            raise ValueError(
                "relation geometry expects a square patch grid; "
                f"received {num_patches} patches"
            )
        axis = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        return torch.stack((xx, yy), dim=-1).reshape(num_patches, 2)

    def spatial_geometry(self, part_attn: torch.Tensor) -> torch.Tensor:
        if part_attn.dim() != 3 or part_attn.size(1) != self.num_parts:
            raise ValueError("part_attn must have shape (B, P, N)")
        prob = part_attn.clamp_min(0)
        prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        coords = self._patch_coordinates(prob.size(-1), prob.device, prob.dtype)
        centers = torch.einsum("bpn,nc->bpc", prob, coords)
        centered = coords.view(1, 1, -1, 2) - centers.unsqueeze(2)
        spreads = (prob.unsqueeze(-1) * centered.square()).sum(dim=2)

        center_l, center_r = self._split_pairs(centers)
        spread_l, spread_r = self._split_pairs(spreads)
        prob_l, prob_r = self._split_pairs(prob)
        center_delta = torch.abs(center_l - center_r)
        center_distance = torch.linalg.vector_norm(center_delta, dim=-1, keepdim=True)
        spread_delta = torch.abs(spread_l - spread_r)
        overlap = torch.sqrt((prob_l * prob_r).clamp_min(1e-8)).sum(
            dim=-1, keepdim=True
        )
        return torch.cat(
            (center_delta, center_distance, spread_delta, overlap), dim=-1
        )

    def visual_relations(
        self, part_tokens: torch.Tensor, part_attn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right = self._split_pairs(part_tokens)
        appearance = torch.cat((left * right, torch.abs(left - right)), dim=-1)
        geometry = self.spatial_geometry(part_attn)
        return self.visual_encoder(torch.cat((appearance, geometry), dim=-1)), geometry

    def semantic_relations(self, semantics: torch.Tensor) -> torch.Tensor:
        left, right = self._split_pairs(semantics)
        relation = torch.cat((left * right, torch.abs(left - right)), dim=-1)
        return self.semantic_encoder(relation)


class BilevelSemanticController(nn.Module):
    """Part- and relation-level one-step differentiable bilevel controller."""

    VALID_SCOPES = ("part", "relation", "hybrid")

    def __init__(
        self, dim: int, text_dim: int, num_parts: int, semantic_rank: int = 64,
        adapter_rank: int = 64, policy_hidden_dim: int = 128,
        policy_tau: float = 1.0, reference_mix: float = 0.5,
        diversity_weight: float = 0.01, enable_relations: bool = True,
        relation_hvp_samples: int = 1,
    ):
        super().__init__()
        if relation_hvp_samples < 1:
            raise ValueError("relation_hvp_samples must be >= 1")
        self.num_parts = num_parts
        self.diversity_weight = float(diversity_weight)
        self.enable_relations = bool(enable_relations)
        self.relation_hvp_samples = int(relation_hvp_samples)
        self.bridge = SemanticTokenBridge(dim, text_dim, num_parts, semantic_rank)
        self.adapter = LowRankSemanticAdapter(dim, adapter_rank)
        self.policy = CurvatureSemanticWeightPolicy(
            dim, policy_hidden_dim, policy_tau, reference_mix
        )
        if self.enable_relations:
            self.relation_encoder = PairwiseRelationEncoder(dim, num_parts)
            self.relation_adapter = LowRankSemanticAdapter(dim, adapter_rank)
            self.relation_policy = CurvatureSemanticWeightPolicy(
                dim, policy_hidden_dim, policy_tau, reference_mix
            )

    @property
    def num_relations(self) -> int:
        return self.relation_encoder.num_relations if self.enable_relations else 0

    def visual_semantics(self, cls_feature: torch.Tensor) -> torch.Tensor:
        return self.bridge.from_visual(cls_feature)

    def text_semantics(self, text_feature: torch.Tensor) -> torch.Tensor:
        return self.bridge.from_text(text_feature)

    def bridge_loss(
        self, visual_semantics: torch.Tensor,
        text_semantics: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.bridge.consistency_loss(
            visual_semantics, text_semantics, self.diversity_weight
        )

    @staticmethod
    def make_state(
        part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        target_semantics: torch.Tensor, curvature: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "part_tokens": part_tokens,
            "policy_semantics": policy_semantics,
            "target_semantics": target_semantics,
            "curvature": curvature,
        }

    @staticmethod
    def _alignment_error(
        adapter: LowRankSemanticAdapter, visual_tokens: torch.Tensor,
        target_semantics: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        adapted = (
            adapter(visual_tokens) if params is None
            else adapter.functional_forward(visual_tokens, params)
        )
        return 1.0 - F.cosine_similarity(adapted, target_semantics, dim=-1)

    def alignment_error(
        self, visual_tokens: torch.Tensor, target_semantics: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self._alignment_error(self.adapter, visual_tokens, target_semantics, params)

    def relation_alignment_error(
        self, visual_relations: torch.Tensor, target_relations: torch.Tensor,
        params: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self._alignment_error(
            self.relation_adapter, visual_relations, target_relations, params
        )

    @staticmethod
    def _stopped_policy_distribution(
        policy: CurvatureSemanticWeightPolicy, visual_tokens: torch.Tensor,
        policy_semantics: torch.Tensor, curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return policy(
            visual_tokens.detach(), policy_semantics.detach(), curvature.detach()
        )

    def policy_distribution(
        self, part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self._stopped_policy_distribution(
            self.policy, part_tokens, policy_semantics, curvature
        )

    def relation_policy_distribution(
        self, relation_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self.enable_relations:
            raise RuntimeError("relation controller is disabled")
        return self._stopped_policy_distribution(
            self.relation_policy, relation_state["tokens"],
            relation_state["policy_semantics"], relation_state["curvature"],
        )

    def pool_parts(
        self, part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        curvature: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, stats = self.policy_distribution(part_tokens, policy_semantics, curvature)
        return (p.detach().unsqueeze(-1) * part_tokens).sum(dim=1), stats

    def pool_relations(
        self, relation_state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, stats = self.relation_policy_distribution(relation_state)
        pooled = (p.detach().unsqueeze(-1) * relation_state["tokens"]).sum(dim=1)
        return pooled, {"rel_" + key: value for key, value in stats.items()}

    def _endpoint_prior(self, part_curvature: torch.Tensor) -> torch.Tensor:
        left, right = self.relation_encoder._split_pairs(part_curvature.unsqueeze(-1))
        prior = torch.sqrt((left.squeeze(-1) * right.squeeze(-1)).clamp_min(1e-8))
        return prior / prior.mean(dim=1, keepdim=True).clamp_min(1e-6)

    def relation_hvp_curvature(
        self, relation_tokens: torch.Tensor, target_relations: torch.Tensor,
    ) -> torch.Tensor:
        """Hutchinson HVP norm of stopped semantic relation alignment."""
        if not self.enable_relations:
            raise RuntimeError("relation controller is disabled")
        fp32_context = (
            torch.cuda.amp.autocast(enabled=False)
            if relation_tokens.is_cuda else nullcontext()
        )
        with torch.enable_grad():
            with fp32_context:
                z = relation_tokens.detach().float().requires_grad_(True)
                target = target_relations.detach().float()
                frozen_params = OrderedDict(
                    (name, param.detach().float())
                    for name, param in self.relation_adapter.named_parameters()
                )
                error = self.relation_alignment_error(z, target, frozen_params)
                gradient = torch.autograd.grad(
                    error.mean(), z, create_graph=True, retain_graph=True
                )[0]
                estimate = torch.zeros_like(error)
                for sample_idx in range(self.relation_hvp_samples):
                    vector = torch.empty_like(z).bernoulli_(0.5).mul_(2).sub_(1)
                    vector = vector / math.sqrt(z.size(-1))
                    hvp = torch.autograd.grad(
                        (gradient * vector).sum(), z,
                        retain_graph=(sample_idx + 1 < self.relation_hvp_samples),
                        create_graph=False,
                    )[0]
                    estimate = estimate + torch.linalg.vector_norm(hvp, dim=-1)
                estimate = estimate / float(self.relation_hvp_samples)
        estimate = estimate / estimate.mean(dim=1, keepdim=True).clamp_min(1e-6)
        return estimate.detach().to(relation_tokens.dtype)

    def make_relation_state(
        self, part_tokens: torch.Tensor, policy_semantics: torch.Tensor,
        target_semantics: torch.Tensor, part_attn: torch.Tensor,
        part_curvature: torch.Tensor, compute_hvp: bool,
    ) -> Dict[str, torch.Tensor]:
        if not self.enable_relations:
            raise RuntimeError("relation controller is disabled")
        visual_relations, geometry = self.relation_encoder.visual_relations(
            part_tokens, part_attn
        )
        policy_relations = self.relation_encoder.semantic_relations(policy_semantics)
        target_relations = self.relation_encoder.semantic_relations(target_semantics)
        curvature = (
            self.relation_hvp_curvature(visual_relations, target_relations)
            if compute_hvp else self._endpoint_prior(part_curvature.detach())
        )
        return {
            "tokens": visual_relations,
            "policy_semantics": policy_relations,
            "target_semantics": target_relations,
            "curvature": curvature,
            "geometry": geometry,
        }

    @staticmethod
    def _single_meta_objective(
        support: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor],
        policy: CurvatureSemanticWeightPolicy, adapter: LowRankSemanticAdapter,
        num_items: int, inner_lr: float, q_mode: str, kl_weight: float,
        prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        support_tokens = support["tokens"].detach().float()
        support_policy_sem = support["policy_semantics"].detach().float()
        support_target_sem = support["target_semantics"].detach().float()
        support_curvature = support["curvature"].detach().float()
        query_tokens = query["tokens"].detach().float()
        query_target_sem = query["target_semantics"].detach().float()
        query_curvature = query["curvature"].detach().float()

        p, policy_stats = policy(support_tokens, support_policy_sem, support_curvature)
        base_params = OrderedDict(adapter.named_parameters())
        support_error = BilevelSemanticController._alignment_error(
            adapter, support_tokens, support_target_sem, base_params
        )
        inner_loss = (num_items * p * support_error).mean()
        inner_grads = torch.autograd.grad(
            inner_loss, tuple(base_params.values()), create_graph=True,
            allow_unused=False,
        )
        fast_params = OrderedDict(
            (name, param - float(inner_lr) * grad)
            for (name, param), grad in zip(base_params.items(), inner_grads)
        )
        query_error_before = BilevelSemanticController._alignment_error(
            adapter, query_tokens, query_target_sem, base_params
        )
        query_error_after = BilevelSemanticController._alignment_error(
            adapter, query_tokens, query_target_sem, fast_params
        )
        if q_mode == "uniform":
            q = torch.full_like(query_error_after, 1.0 / num_items)
        elif q_mode == "hvp":
            q = query_curvature / query_curvature.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-6)
        else:
            raise ValueError("meta q_mode must be 'uniform' or 'hvp'")
        q = q.detach()
        outer_align = (q * query_error_after).sum(dim=1).mean()
        uniform = torch.full_like(p, 1.0 / num_items)
        policy_kl = (
            p * (p.clamp_min(1e-8).log() - uniform.log())
        ).sum(dim=1).mean()
        meta_loss = outer_align + float(kl_weight) * policy_kl
        before = (q * query_error_before.detach()).sum(dim=1).mean()
        stats = {
            prefix + "meta_outer_align": outer_align.detach(),
            prefix + "meta_inner_align": inner_loss.detach(),
            prefix + "meta_improvement": before - outer_align.detach(),
            prefix + "meta_policy_kl": policy_kl.detach(),
        }
        stats.update({prefix + key: value for key, value in policy_stats.items()})
        return meta_loss, stats

    @staticmethod
    def _part_view(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "tokens": state["part_tokens"],
            "policy_semantics": state["policy_semantics"],
            "target_semantics": state["target_semantics"],
            "curvature": state["curvature"],
        }

    @classmethod
    def _check_scope(cls, scope: str) -> None:
        if scope not in cls.VALID_SCOPES:
            raise ValueError(f"meta scope must be one of {cls.VALID_SCOPES}")

    def meta_objective(
        self, support: Dict[str, torch.Tensor], query: Dict[str, torch.Tensor],
        inner_lr: float, q_mode: str = "uniform", kl_weight: float = 0.01,
        scope: str = "relation", relation_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return an exact hypergradient for a one-step unrolled objective."""
        self._check_scope(scope)
        objectives = []
        stats: Dict[str, torch.Tensor] = {}
        if scope in ("part", "hybrid"):
            part_loss, part_stats = self._single_meta_objective(
                self._part_view(support), self._part_view(query), self.policy,
                self.adapter, self.num_parts, inner_lr, q_mode, kl_weight,
                "part_",
            )
            objectives.append(part_loss)
            stats.update(part_stats)
        if scope in ("relation", "hybrid"):
            if not self.enable_relations:
                raise RuntimeError("relation meta scope requested but disabled")
            rel_loss, rel_stats = self._single_meta_objective(
                support["relation_state"], query["relation_state"],
                self.relation_policy, self.relation_adapter, self.num_relations,
                inner_lr, q_mode, kl_weight, "rel_",
            )
            objectives.append(float(relation_weight) * rel_loss)
            stats.update(rel_stats)
        meta_loss = torch.stack(objectives).sum()
        stats["meta_loss"] = meta_loss.detach()
        return meta_loss, stats

    @staticmethod
    def _single_real_alignment(
        state: Dict[str, torch.Tensor], policy: CurvatureSemanticWeightPolicy,
        adapter: LowRankSemanticAdapter, num_items: int, prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        p, policy_stats = BilevelSemanticController._stopped_policy_distribution(
            policy, state["tokens"], state["policy_semantics"], state["curvature"]
        )
        error = BilevelSemanticController._alignment_error(
            adapter, state["tokens"], state["target_semantics"]
        )
        loss = (num_items * p.detach() * error).mean()
        stats = {prefix + "real_align": loss.detach()}
        stats.update({prefix + key: value for key, value in policy_stats.items()})
        return loss, stats

    def real_weighted_alignment(
        self, state: Dict[str, torch.Tensor], scope: str = "relation",
        relation_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Real model loss; detached policies prevent a direct phi gradient."""
        self._check_scope(scope)
        losses = []
        stats: Dict[str, torch.Tensor] = {}
        if scope in ("part", "hybrid"):
            loss, item_stats = self._single_real_alignment(
                self._part_view(state), self.policy, self.adapter,
                self.num_parts, "part_meta_",
            )
            losses.append(loss)
            stats.update(item_stats)
        if scope in ("relation", "hybrid"):
            loss, item_stats = self._single_real_alignment(
                state["relation_state"], self.relation_policy,
                self.relation_adapter, self.num_relations, "rel_meta_",
            )
            losses.append(float(relation_weight) * loss)
            stats.update(item_stats)
        return torch.stack(losses).sum(), stats

    def policy_parameters(self, scope: str) -> Iterable[nn.Parameter]:
        """Return exactly the parameters owned by the requested meta policy."""
        self._check_scope(scope)
        if scope == "part":
            return self.policy.parameters()
        if scope == "relation":
            if not self.enable_relations:
                raise RuntimeError("relation meta scope requested but disabled")
            return self.relation_policy.parameters()
        return tuple(self.policy.parameters()) + tuple(self.relation_policy.parameters())

    def all_policy_parameters(self) -> Iterable[nn.Parameter]:
        """All phi parameters, used to exclude policies from the main optimizer."""
        parameters = tuple(self.policy.parameters())
        if self.enable_relations:
            parameters = parameters + tuple(self.relation_policy.parameters())
        return parameters
