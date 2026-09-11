"""Small CPU tests for the V5 bilevel gradient invariants."""

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
        )

    def _state(self, seed):
        generator = torch.Generator().manual_seed(seed)
        return self.controller.make_state(
            torch.randn(
                self.batch, self.parts, self.dim, generator=generator,
                requires_grad=True,
            ),
            torch.randn(
                self.batch, self.parts, self.dim, generator=generator,
                requires_grad=True,
            ),
            torch.randn(
                self.batch, self.parts, self.dim, generator=generator,
                requires_grad=True,
            ),
            torch.rand(self.batch, self.parts, generator=generator) + 0.2,
        )

    def test_meta_gradient_and_direct_gradient_isolation(self):
        support = self._state(1)
        query = self._state(2)
        meta_loss, _ = self.controller.meta_objective(
            support, query, inner_lr=0.1
        )
        policy_params = tuple(self.controller.policy.parameters())
        meta_grads = torch.autograd.grad(meta_loss, policy_params)
        meta_norm = sum(grad.abs().sum() for grad in meta_grads)
        self.assertGreater(meta_norm.item(), 0.0)

        for param in self.controller.parameters():
            param.grad = None
        real_loss, _ = self.controller.real_weighted_alignment(support)
        real_loss.backward()
        self.assertFalse(any(param.grad is not None for param in policy_params))
        self.assertTrue(
            any(param.grad is not None for param in self.controller.adapter.parameters())
        )

    def test_reference_mixture_floor(self):
        state = self._state(3)
        p, _ = self.controller.policy_distribution(
            state["part_tokens"],
            state["policy_semantics"],
            state["curvature"],
        )
        expected_floor = 0.5 / self.parts
        self.assertTrue(torch.all(p >= expected_floor - 1e-7))
        self.assertTrue(torch.allclose(p.sum(dim=1), torch.ones(self.batch)))


if __name__ == "__main__":
    unittest.main()

