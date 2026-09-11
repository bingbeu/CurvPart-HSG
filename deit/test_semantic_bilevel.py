"""CPU tests for V6 relation curvature and bilevel gradient invariants."""

import unittest

import torch

from semantic_bilevel import BilevelSemanticController


class BilevelSemanticControllerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.batch = 4
        self.parts = 6
        self.dim = 32
        self.controller = BilevelSemanticController(
            dim=self.dim,
            text_dim=24,
            num_parts=self.parts,
            semantic_rank=8,
            adapter_rank=8,
            policy_hidden_dim=16,
            reference_mix=0.5,
            relation_hvp_samples=1,
        )

    def _state(self, seed, compute_hvp=True):
        generator = torch.Generator().manual_seed(seed)
        part_tokens = torch.randn(
            self.batch, self.parts, self.dim, generator=generator,
            requires_grad=True,
        )
        policy_semantics = torch.randn(
            self.batch, self.parts, self.dim, generator=generator,
            requires_grad=True,
        )
        target_semantics = torch.randn(
            self.batch, self.parts, self.dim, generator=generator,
            requires_grad=True,
        )
        part_curvature = torch.rand(
            self.batch, self.parts, generator=generator
        ) + 0.2
        # 4x4 patch grid; positive attention is normalized by the encoder.
        part_attn = torch.rand(
            self.batch, self.parts, 16, generator=generator,
            requires_grad=True,
        )
        state = self.controller.make_state(
            part_tokens, policy_semantics, target_semantics, part_curvature
        )
        state["relation_state"] = self.controller.make_relation_state(
            part_tokens,
            policy_semantics,
            target_semantics,
            part_attn,
            part_curvature,
            compute_hvp=compute_hvp,
        )
        return state

    def test_part_meta_gradient_and_direct_gradient_isolation(self):
        support = self._state(1, compute_hvp=False)
        query = self._state(2, compute_hvp=False)
        meta_loss, _ = self.controller.meta_objective(
            support, query, inner_lr=0.1, scope="part"
        )
        policy_params = tuple(self.controller.policy.parameters())
        meta_grads = torch.autograd.grad(meta_loss, policy_params)
        self.assertGreater(sum(g.abs().sum() for g in meta_grads).item(), 0.0)

        for param in self.controller.parameters():
            param.grad = None
        real_loss, _ = self.controller.real_weighted_alignment(
            support, scope="part"
        )
        real_loss.backward()
        self.assertFalse(any(param.grad is not None for param in policy_params))
        self.assertTrue(any(p.grad is not None for p in self.controller.adapter.parameters()))

    def test_relation_hvp_and_strict_bilevel_isolation(self):
        support = self._state(3, compute_hvp=True)
        query = self._state(4, compute_hvp=False)
        curvature = support["relation_state"]["curvature"]
        self.assertEqual(
            tuple(curvature.shape),
            (self.batch, self.parts * (self.parts - 1) // 2),
        )
        self.assertFalse(curvature.requires_grad)
        self.assertTrue(torch.isfinite(curvature).all())
        self.assertTrue((curvature >= 0).all())

        meta_loss, _ = self.controller.meta_objective(
            support, query, inner_lr=0.1, scope="relation"
        )
        policy_params = tuple(self.controller.relation_policy.parameters())
        meta_grads = torch.autograd.grad(meta_loss, policy_params)
        self.assertGreater(sum(g.abs().sum() for g in meta_grads).item(), 0.0)

        for param in self.controller.parameters():
            param.grad = None
        real_loss, _ = self.controller.real_weighted_alignment(
            support, scope="relation"
        )
        real_loss.backward()
        self.assertFalse(any(param.grad is not None for param in policy_params))
        self.assertTrue(
            any(p.grad is not None for p in self.controller.relation_adapter.parameters())
        )

    def test_relation_geometry_is_horizontal_flip_invariant(self):
        generator = torch.Generator().manual_seed(11)
        attention = torch.rand(
            self.batch, self.parts, 4, 4, generator=generator
        )
        original = self.controller.relation_encoder.spatial_geometry(
            attention.flatten(2)
        )
        flipped = self.controller.relation_encoder.spatial_geometry(
            attention.flip(-1).flatten(2)
        )
        self.assertTrue(torch.allclose(original, flipped, atol=1e-5, rtol=1e-5))

    def test_reference_mixture_floor_for_parts_and_relations(self):
        state = self._state(5, compute_hvp=False)
        part_p, _ = self.controller.policy_distribution(
            state["part_tokens"], state["policy_semantics"], state["curvature"]
        )
        relation_p, _ = self.controller.relation_policy_distribution(
            state["relation_state"]
        )
        self.assertTrue(torch.all(part_p >= 0.5 / self.parts - 1e-7))
        self.assertTrue(
            torch.all(relation_p >= 0.5 / self.controller.num_relations - 1e-7)
        )
        self.assertTrue(torch.allclose(part_p.sum(1), torch.ones(self.batch)))
        self.assertTrue(torch.allclose(relation_p.sum(1), torch.ones(self.batch)))


if __name__ == "__main__":
    unittest.main()
