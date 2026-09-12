# Task-Feedback Bilevel Semantic Loop (V7)

V7 closes the gap between semantic adaptation and hierarchical recognition.
The lower-level variable is a shared low-rank adapter `psi` that is used by the
real training path **and inference**, rather than a disposable feature tensor.

## Objective

For independent support/query augmentations of the same image:

```text
p_phi = policy(part, semantics, stopgrad(curvature))
psi+  = psi - inner_lr * grad_psi sum_i P * p_phi[i] * e_i(support; psi)

L_outer = task_weight * L_hier(query; psi+, q)
        + semantic_weight * sum_i q[i] * e_i(query; psi+)
        + kl_weight * KL(p_phi || uniform)
```

`q` is uniform or stopped HVP. Learned `p_phi` never multiplies a raw query
error. `create_graph=True` preserves the one-step hypergradient, and only the
policy optimizer updates `phi`. The real alignment loss uses `p_phi.detach()`.

The outer task is the available-label hierarchical classification objective:
fine loss is masked to fine-labelled samples, family loss is masked to samples
with at least family labels, and the basic loss uses every sample.

## Critical implementation invariants

- `bilevel.adapter` is used by support adaptation, real model training and
  inference pooling.
- The query backbone is evaluated without gradients; the virtual adapter is
  evaluated functionally in FP32.
- Policy parameters are excluded from the main optimizer.
- Outer evaluation uses fixed `q`, never learned `p_phi`.
- The three residual gates start at zero, so the classifier is recoverable to
  the E2 path at initialization.
- Bilevel training refuses silent random initialization unless
  `--allow-random-init` is explicitly passed.

## Aircraft example

```bash
python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 32 --epochs 100 --num_workers 8 \
  --data-set AIR-HIER --data-path /data \
  --output_dir ./output/air_task_bilevel_v7 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 --fm_proportion 0.6 \
  --seed 0 --random_seed 0 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel --num-parts 8 \
  --lam-cls 0.0 --lam-attr 1.0 --proto-align-weight 0.0 \
  --meta-start-epoch 5 --meta-inner-lr 0.1 --meta-lr 1e-4 \
  --meta-real-weight 0.1 --meta-reference-mix 0.5 \
  --meta-task-weight 1.0 --meta-semantic-weight 0.1 \
  --meta-fine-weight 1.0 --meta-family-weight 0.5 \
  --meta-basic-weight 0.5 --meta-q uniform
```

Start with batch size 32 because two views and the unrolled graph increase
memory. Use a new output directory. A V5 optimizer state is not a resumable V7
optimizer state; load an E2/DeiT checkpoint with `--finetune` instead.

## Verification

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

The tests cover task-loss hypergradients, direct-gradient isolation, reference
probability bounds and the shared adapter's presence in inference pooling.

This is a strict separation of lower/upper variables optimized by a
differentiable one-step truncated bilevel solver. It does not claim that the
inner argmin is solved exactly.
