# RelCurv-HSG V6: relation-level curvature and a bilevel semantic loop

V6 keeps the local semantic-part path that is useful on CUB and adds a separate,
zero-gated relation path for configuration-dominated datasets such as
FGVC-Aircraft. The method is both a semantic feedback loop and an explicit
one-step differentiable bilevel optimization.

## Method

For `P` semantic parts, every undirected pair `(a,b)` forms a relation. With the
default `P=8`, the model handles only `P(P-1)/2=28` relations.

The visual relation contains symmetric appearance and layout terms:

```text
r_ab = Enc_v[t_a * t_b, |t_a-t_b|,
             |mu_a-mu_b|, ||mu_a-mu_b||,
             |var_a-var_b|, overlap(A_a,A_b)]
```

`mu`, `var`, and `overlap` are computed from part-to-patch attention. Absolute
geometry makes this descriptor robust to horizontal flip augmentation. The
semantic target relation is

```text
u_ab = Enc_s[s_a * s_b, |s_a-s_b|].
```

Relation curvature is a Hutchinson HVP norm of the semantic relation objective:

```text
kappa_ab = || H_r L_rel(r,u) v ||_2.
```

The HVP is computed in FP32 on stopped copies of relation features and adapter
parameters. `kappa` is detached before entering the policy, so V6 does not
create a third-order training path.

For support/query augmentations of the same image, the relation policy predicts
`p_phi` from `(r, u_visual, stopgrad(kappa))`. The lower-level variable is a
shared low-rank relation adapter `psi`:

```text
psi+ = psi - inner_lr * grad_psi sum_ab p_phi(ab) e_ab(support; psi)
L_meta(phi) = sum_ab q_ab e_ab(query; psi+)
```

The recommended reference distribution is `q=uniform`. It prevents the same
curvature signal from both choosing and judging relations. The implementation
uses `create_graph=True`, so the policy receives the exact hypergradient of this
one-step unrolled objective.

During the real model update, `p_phi` is detached. All policy parameters are
excluded from the main optimizer and placed in a separate AdamW meta optimizer.
Thus the direct weighted-error shortcut cannot train the policy.

To mitigate concentration rather than claim to eliminate it mathematically,
V6 uses

```text
p = (1-rho) * uniform + rho * softmax(logits / tau),
```

which guarantees `p_i >= (1-rho)/M` for `M` parts or relations.

## Why CUB remains protected

The local part residual and relation residual have independent zero-initialized
gates. Enabling V6 therefore starts from the original CLS baseline. CUB may
activate the local path while leaving the configuration path near zero;
Aircraft may learn the reverse. This is a safe initialization, not a guarantee
that every trained seed will improve, so both gates and validation metrics must
be reported.

## Main files

- `deit/semantic_bilevel.py`: semantic bridge, relation encoder, relation HVP,
  part/relation policies, fast adapters, and one-step bilevel objectives.
- `deit/semantic_part_v4.py`: exports live attention for geometry and stopped
  part curvature for the policy prior.
- `deit/models_hier.py`: same visual semantic path at train/test and independent
  zero-gated local/relation classifier residuals.
- `deit/engine_vit_hier_partial.py`: support/query unrolling and strictly
  separated meta/model optimizer steps.
- `deit/dataset/datasets_partial.py`: independent support/query augmentations.
- `deit/test_semantic_bilevel.py`: relation geometry, HVP, hypergradient, and
  gradient-isolation tests.

## Aircraft command

```bash
python deit/main_hier_partial.py \
  --model deit_small_patch16_224 \
  --batch-size 32 \
  --epochs 100 \
  --num_workers 8 \
  --data-set AIR-HIER \
  --data-path /data \
  --output_dir ./output/air_relcurv_v6 \
  --texts captions/air_caps.txt \
  --sp_proportion 0.3 \
  --fm_proportion 0.6 \
  --seed 0 \
  --random_seed 0 \
  --finetune deit_small_patch16_224-cd65a155.pth \
  --enable-bilevel \
  --meta-scope relation \
  --num-parts 8 \
  --lam-cls 0.0 \
  --lam-attr 1.0 \
  --proto-align-weight 0.0 \
  --meta-start-epoch 5 \
  --meta-inner-lr 0.1 \
  --meta-lr 1e-4 \
  --meta-real-weight 0.1 \
  --meta-reference-mix 0.5 \
  --relation-hvp-samples 1 \
  --meta-q uniform
```

Two views plus relation HVP increase memory use. Start with batch size 32. If
needed, reduce `--meta-adapter-rank` and `--semantic-rank` from 64 to 32.

## Verification

```bash
python -m compileall -q deit
PYTHONPATH=deit python -m unittest -v deit/test_semantic_bilevel.py
```

The tensor tests require PyTorch. They verify:

1. relation HVP is finite, nonnegative, correctly shaped, and detached;
2. `L_meta` gives nonzero gradients to the selected policy;
3. real weighted alignment gives no direct gradient to either policy;
4. the shared lower-level adapter still receives real model gradients;
5. relation geometry is invariant to horizontal image flips;
6. both policy distributions obey the probability floor and sum to one.

## Required ablation matrix

| ID | Scope | Relation HVP | Policy | Purpose |
|---|---|---:|---:|---|
| A0 | none | no | none | E1+E2 baseline |
| A1 | part | part | learned | V5 local-token loop |
| A2 | relation | no | learned | relation loop without relation HVP |
| A3 | relation | yes | uniform (`rho=0`) | relation curvature without learned selection |
| A4 | relation | yes | learned | full V6 |
| A5 | hybrid | yes | learned | local + relation joint loop |

Run Aircraft and CUB with at least three seeds. Choose checkpoints on a held-out
validation split using the declared primary metric; do not tune on the test set.

## Paper claim boundary

Use: “an explicit bilevel objective optimized through one-step differentiable
unrolling” and “mitigates degenerate relation concentration by decoupling
policy generation from direct weighted-error minimization.” Do not claim an
exactly solved inner argmin or mathematically guaranteed elimination of collapse.
