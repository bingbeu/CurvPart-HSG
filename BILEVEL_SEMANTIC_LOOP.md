# Curvature-Guided Bilevel Semantic Feedback Loop (V5)

## Optimization

For support/query augmentations of the same image, the semantic policy predicts
part weights `p_phi` from visual part tokens, visual semantic tokens, and a
detached part-level HVP curvature prior.

The lower-level variable is the shared low-rank semantic adapter `psi`:

```text
psi+ = psi - inner_lr * grad_psi L_inner(support, p_phi)
```

The outer loss evaluates the virtually updated adapter on the query view:

```text
L_meta(phi) = L_align(query; psi+)
```

The implementation preserves the hypergradient with `create_graph=True`.
`phi` is updated only by `L_meta`. During the real model update, `p_phi` is
detached, so task/alignment losses cannot directly optimize the weighting
policy.

The policy distribution is bounded by

```text
p = (1-rho) * uniform + rho * softmax(logits / tau)
```

so every part has probability at least `(1-rho)/P`.

## Main files

- `deit/semantic_bilevel.py`: semantic bridge, fast adapter, policy and meta objective.
- `deit/semantic_part_v4.py`: exports detached part-level HVP curvature.
- `deit/models_hier.py`: uses P visual semantic tokens in both train and inference.
- `deit/engine_vit_hier_partial.py`: alternating meta/model updates.
- `deit/dataset/datasets_partial.py`: independent support/query augmentations.

## Aircraft example

```bash
python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 32 \
  --epochs 100 \
  --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /data \
  --output_dir ./output/air_bilevel_v5 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 \
  --fm_proportion 0.6 \
  --seed 0 \
  --random_seed 0 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel \
  --num-parts 8 \
  --meta-start-epoch 5 \
  --meta-inner-lr 0.1 \
  --meta-lr 1e-4 \
  --meta-real-weight 0.1 \
  --meta-reference-mix 0.5 \
  --meta-q uniform
```

Two augmented views and the HVP path increase memory use. Start with batch size
32. If memory is still insufficient, reduce `--meta-adapter-rank` and
`--semantic-rank` from 64 to 32. Do not disable the higher-order graph in the
meta objective.

## Gradient checks

After installing project dependencies, run:

```bash
PYTHONPATH=deit python -m unittest deit/test_semantic_bilevel.py
```

The tests verify that:

1. `L_meta` gives a nonzero gradient to the policy;
2. the real weighted alignment gives no direct gradient to the policy;
3. the real loss still updates the shared semantic adapter;
4. policy probabilities satisfy the reference-mixture floor and sum to one.

## Recommended ablation

1. Existing E1+E2 without `--enable-bilevel`.
2. V5 with `--meta-reference-mix 0` (uniform policy).
3. V5 with learned policy and `--meta-q uniform`.
4. V5 with `--meta-q hvp`.
5. V5 with HVP disabled (`--no-hvp`) to isolate curvature contribution.

In the paper, describe this as an explicit bilevel objective optimized with a
one-step differentiable unrolling approximation. Do not claim that the inner
argmin is solved exactly.

