"""
SemanticPartTokenGeneratorV4 — 类别–属性层次语义接地 + 曲率感知部件对齐
======================================================================
在 V3（已实测 best 92.98，超过 V0/V1/V2）基础上，按论文"改后叙事"补齐的版本：
把 V3 的"纯属性接地 + 属性相似度 HVP"升级为
"类别(全局) + 属性(局部) 层次接地 + 联合对齐目标 HVP 曲率"。

相对 V3 的增量（全部标注 [V4]）：
  1. [V4-1] 类别语义进入 part query：
        q = part_q + softplus(gamma_cls)·c + softplus(gamma_attr)·(Σ_k β_jk a_k)
     （V3 只有属性项；这里 tokens[0]=类别语义，tokens[1:]=属性语义）
  2. [V4-2] 新增逐 token 类别相似度  s_cls = cos(x_i, c)。
  3. [V4-3] HVP 目标从"属性相似度"换成"联合类别–属性对齐目标"：
        ℓ_i = lam_cls·(1 - s_i^cls) + lam_attr·(1 - s_i^attr)
        κ_i = || HVP_{x_i}( mean_i ℓ_i ) ||_2
  4. [V4-4] 对齐损失同步改成联合目标（曲率概率加权，detach 防反馈回路）。
  5. [V4-5] curv_head 学生输入同时注入类别上下文 + 属性上下文，
        让学生能拟合"联合曲率"老师（回归分支仍 detach，不污染 backbone）。
  6. [V4-6] 可选全局类别对齐项（V0 式，默认关闭 global_align_weight=0.0）。

保留 V3 全部功能（零改动）：mean 曲率归一化、全强度曲率 attention 偏置、
中心化有界特征增益、概率化对齐、attr 语义接地、HVP 蒸馏、AMP fp32 回归分支、
gate_status 监控、return_aux 全部字段。

tokens 约定（务必与训练器一致）：
    extra_tokens = [类别token, 属性token_1, 属性token_2, ...]
  - len==1：仅类别语义，无属性分支（lam_attr 自动归零，联合目标退化为纯类别对齐）。
  - len>1 ：tokens[0]=类别，tokens[1:]=属性。

若训练脚本按 SemanticPartTokenGenerator 导入，可在末尾加别名：
    SemanticPartTokenGenerator = SemanticPartTokenGeneratorV4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    from torch.nn.init import trunc_normal_

try:
    from apex import amp
except ImportError:
    amp = None


class SemanticPartTokenGeneratorV4(nn.Module):
    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        num_parts: int,
        attn_drop: float = 0.0,
        enable_hvp: bool = True,
        assign_scale: float = 5.0,
        curv_tau: float = 1.0,
        hvp_probe: str = "rademacher",
        hvp_samples: int = 4,
        curv_norm_eps: float = 1e-4,
        curv_reg_weight: float = 0.05,          # [V3-3] 0.1 -> 0.05
        curv_weight_max: float = 8.0,
        feat_gain_max: float = 0.25,
        lam_cls: float = 0.5,                   # [V4] 联合目标中类别对齐权重
        lam_attr: float = 0.5,                  # [V4] 联合目标中属性对齐权重
        global_align_weight: float = 0.0,       # [V4-6] 可选全局类别对齐项，默认关
    ):
        super().__init__()
        if curv_tau <= 0:
            raise ValueError("curv_tau must be positive")
        if hvp_samples < 1:
            raise ValueError("hvp_samples must be at least 1")

        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_parts = num_parts
        self.enable_hvp = enable_hvp
        self.assign_scale = assign_scale
        self.curv_tau = max(curv_tau, 0.1)          # 温度下限，防 softmax 坍缩
        self.hvp_probe = hvp_probe
        self.hvp_samples = hvp_samples
        self.curv_norm_eps = curv_norm_eps
        self.curv_reg_weight = curv_reg_weight
        self.curv_weight_max = curv_weight_max
        self.feat_gain_max = feat_gain_max
        self.lam_cls = float(lam_cls)
        self.lam_attr = float(lam_attr)
        self.global_align_weight = float(global_align_weight)
        self.scale = embed_dim ** -0.5
        self.eps = 1e-6

        self.input_proj = nn.Linear(in_dim, embed_dim)
        self.class_proj = nn.Linear(embed_dim, embed_dim)   # [V4] 类别语义投影
        self.semantic_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_norm = nn.LayerNorm(embed_dim)
        hidden_dim = max(embed_dim // 4, 16)
        self.curv_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.part_queries = nn.Parameter(torch.zeros(1, num_parts, embed_dim))
        self.attn_drop = nn.Dropout(attn_drop)

        # [V3-2] 关键门控"近全强度"起步（实测该管线门控学不动，必须初始值就位）
        self.curv_feat_alpha = nn.Parameter(torch.tensor(0.1))   # 特征增益 sigmoid(0.1)≈0.525
        self.curv_sem_alpha = nn.Parameter(torch.tensor(0.5))   # 属性语义注入 tanh(0.5)≈0.46
        self.curv_logit_alpha = nn.Parameter(torch.tensor(1.0)) # 曲率偏置 tanh(1.0)≈0.76
        self.sim_logit_alpha = nn.Parameter(torch.tensor(0.5))  # 相似度偏置 tanh(0.5)≈0.46
        self.attr_logit_alpha = nn.Parameter(torch.tensor(1.0))

        # [V4-1] 类别/属性语义注入强度（softplus 保证恒正、处处有梯度；softplus(0)≈0.693 开放起步）
        self.gamma_cls = nn.Parameter(torch.tensor(0.0))
        self.gamma_attr = nn.Parameter(torch.tensor(0.0))
        # [V4-5] 类别上下文注入学生曲率头（tanh(0.5)≈0.46，与 curv_sem_alpha 对称）
        self.cls_sem_alpha = nn.Parameter(torch.tensor(0.5))

        trunc_normal_(self.part_queries, std=0.02)

    # ------------------------------------------------------------------ utils
    def gate_status(self):
        """返回门控实际生效值，供训练循环打点监控。"""
        with torch.no_grad():
            return {
                "curv_feat_gain": (self.feat_gain_max * self.curv_feat_alpha.sigmoid()).item(),
                "curv_sem_gate": self.curv_sem_alpha.tanh().item(),
                "curv_logit_gate": self.curv_logit_alpha.tanh().item(),
                "sim_logit_gate": self.sim_logit_alpha.tanh().item(),
                "attr_scale": (0.1 + F.softplus(self.attr_logit_alpha)).item(),
                "gamma_cls": F.softplus(self.gamma_cls).item(),     # [V4]
                "gamma_attr": F.softplus(self.gamma_attr).item(),   # [V4]
                "cls_sem_gate": self.cls_sem_alpha.tanh().item(),   # [V4]
            }

    def _expand_token(self, token, batch_size, device, dtype):
        if token.dim() == 2:
            token = token.unsqueeze(1)
        token = token.to(device=device, dtype=dtype)
        if token.shape[0] == 1 and batch_size > 1:
            token = token.expand(batch_size, -1, -1)
        return token

    def _flatten_input(self, x):
        if x.dim() == 4:
            x = x.flatten(2).transpose(1, 2)
        elif x.dim() != 3:
            raise ValueError(f"Expected x to be 3D or 4D, got shape {tuple(x.shape)}")
        return x

    def _part_semantics(self, cls_tokens, attr_tokens, batch_size):
        """[V4] 层次语义接地：类别(全局) + 属性(局部) 同时注入 part query。"""
        part_q = self.part_queries.expand(batch_size, -1, -1)   # (B,P,C)
        cls_sem = self.class_proj(cls_tokens)                    # (B,1,C)
        sem_k = self.semantic_proj(attr_tokens)                  # (B,K,C)

        part_q_norm = F.normalize(part_q, dim=-1)
        sem_k_norm = F.normalize(sem_k, dim=-1)
        attr_logits = part_q_norm @ sem_k_norm.transpose(-2, -1)
        attr_scale = 0.1 + F.softplus(self.attr_logit_alpha)    # 处处有梯度
        attr_attn = torch.softmax(attr_logits * attr_scale, dim=-1)
        sem_per_part = attr_attn @ sem_k                          # (B,P,C)

        # [V4-1] 层次接地：q = part_q + γ_cls·c + γ_attr·Σβ a
        cls_gate = F.softplus(self.gamma_cls)
        attr_gate = F.softplus(self.gamma_attr)
        q = part_q + cls_gate * cls_sem + attr_gate * sem_per_part
        return q, cls_sem, sem_per_part, attr_attn

    def _token_part_similarity(self, x, sem_per_part):
        x_norm = F.normalize(x, dim=-1)
        sem_norm = F.normalize(sem_per_part, dim=-1)
        token_part_sim = torch.einsum("bnc,bpc->bnp", x_norm, sem_norm)
        part_assign = torch.softmax(token_part_sim * self.assign_scale, dim=-1)
        per_token_sim = (part_assign * token_part_sim).sum(dim=-1)
        return token_part_sim, per_token_sim, part_assign

    def _make_probe(self, hvp_x):
        if self.hvp_probe == "rademacher":
            return torch.empty_like(hvp_x).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        if self.hvp_probe == "normal":
            return torch.randn_like(hvp_x)
        raise ValueError("hvp_probe must be 'rademacher' or 'normal'")

    def _compute_hvp_curvature(self, x, cls_sem, sem_per_part, lam_cls, lam_attr):
        """[V4-3] 对联合类别–属性对齐目标求逐 token 对角 Hessian（Hutchinson 估计）。

        ℓ_i = lam_cls·(1 - s_i^cls) + lam_attr·(1 - s_i^attr)
        曲率 = 该联合目标对 x_i 的二阶敏感度。
        float32、仅训练的教师；推理用 curv_head 学生，保证前后一致。
        """
        with torch.enable_grad():
            hvp_x = x.detach().float().requires_grad_(True)          # (B,N,C)
            hvp_cls = cls_sem.detach().float()                        # (B,1,C)
            hvp_sem = sem_per_part.detach().float()                   # (B,P,C)

            _, per_token_sim, _ = self._token_part_similarity(hvp_x, hvp_sem)  # s_attr (B,N)
            s_cls = F.cosine_similarity(hvp_x, hvp_cls.expand_as(hvp_x), dim=-1)  # s_cls (B,N)

            # 联合对齐目标
            ell = lam_cls * (1.0 - s_cls) + lam_attr * (1.0 - per_token_sim)
            objective = ell.mean()
            grad = torch.autograd.grad(objective, hvp_x, create_graph=True)[0]

            diag_acc = torch.zeros_like(hvp_x)
            for sample_idx in range(self.hvp_samples):
                probe = self._make_probe(hvp_x)
                retain_graph = sample_idx < self.hvp_samples - 1
                hvp = torch.autograd.grad(
                    (grad * probe).sum(), hvp_x, retain_graph=retain_graph
                )[0]
                diag_acc = diag_acc + probe * hvp

            diag_estimate = diag_acc / float(self.hvp_samples)
            curvature = diag_estimate.norm(p=2, dim=-1, keepdim=True).detach()
        return curvature.to(dtype=x.dtype, device=x.device)

    def _normalize_curvature(self, curvature):
        # [V3-1] 除以 per-sample mean：mean=1、典型范围 [0.2,5]，给 log1p/softmax 真实对比度
        denom = curvature.mean(dim=1, keepdim=True).clamp_min(self.curv_norm_eps).detach()
        return curvature / denom

    # ---------------------------------------------------------------- forward
    def forward(self, x, extra_tokens, return_aux: bool = False, compute_hvp: bool = None):
        x = self._flatten_input(x)
        x = self.input_proj(x)
        B, N, C = x.shape

        if not isinstance(extra_tokens, (list, tuple)) or len(extra_tokens) == 0:
            raise ValueError("extra_tokens must be a non-empty list/tuple")

        tokens = [self._expand_token(t, B, x.device, x.dtype) for t in extra_tokens]

        # [V4] tokens[0]=类别语义；tokens[1:]=属性语义
        cls_tokens = tokens[0]
        if len(tokens) > 1:
            attr_tokens = torch.cat(tokens[1:], dim=1)
            has_attr = True
        else:
            # 无属性 token：属性分支退化为类别语义，lam_attr 归零 -> 联合目标退化为纯类别对齐
            attr_tokens = cls_tokens
            has_attr = False

        lam_cls = self.lam_cls
        lam_attr = self.lam_attr if has_attr else 0.0

        q, cls_sem, sem_per_part, attr_attn = self._part_semantics(cls_tokens, attr_tokens, B)

        q = q.to(dtype=x.dtype)
        cls_sem = cls_sem.to(dtype=x.dtype)
        sem_per_part = sem_per_part.to(dtype=x.dtype)

        token_part_sim, per_token_sim, part_assign = \
            self._token_part_similarity(x, sem_per_part)

        # [V4-2] 逐 token 类别相似度
        cls_sem_exp = cls_sem.expand_as(x)                      # (B,N,C)
        s_cls = F.cosine_similarity(x, cls_sem_exp, dim=-1)    # (B,N)

        # 属性上下文（逐 token 软指派聚合）
        token_sem = torch.einsum(
            "bnp,bpc->bnc", part_assign.float(), sem_per_part.float()
        ).to(dtype=x.dtype)

        sem_gate = self.curv_sem_alpha.tanh().to(dtype=x.dtype)       # 属性上下文门控
        cls_gate_h = self.cls_sem_alpha.tanh().to(dtype=x.dtype)      # [V4] 类别上下文门控

        # [V4-5] 分类/特征路径：保持端到端梯度；学生输入 = x + 类别上下文 + 属性上下文
        forward_input = x + sem_gate * token_sem + cls_gate_h * cls_sem_exp
        pred_curvature = self.curv_head(forward_input).clamp_min(self.eps)

        curv_reg_loss = pred_curvature.new_zeros(())
        hvp_curvature = None

        do_hvp = self.enable_hvp if compute_hvp is None else (self.enable_hvp and compute_hvp)
        if (
            do_hvp
            and self.training
            and torch.is_grad_enabled()
        ):
            # 回归分支不能把梯度传回 backbone 或语义分支
            reg_input = (
                x.detach()
                + sem_gate * token_sem.detach()
                + cls_gate_h * cls_sem_exp.detach()
            )
            head_dtype = next(self.curv_head.parameters()).dtype
            reg_input = reg_input.to(dtype=head_dtype)

            # 回归分支强制 fp32：只禁用原生 autocast。
            # 注意：不用 apex.amp.disable_casts()——某些 apex 版本该接口有 bug
            # ('AmpState' object has no attribute 'handle')，会直接崩。
            def _run_reg_head():
                return self.curv_head(reg_input)

            try:
                with torch.cuda.amp.autocast(enabled=False):
                    pred_curvature_reg = _run_reg_head()
            except (TypeError, AttributeError, RuntimeError):
                pred_curvature_reg = _run_reg_head()

            pred_curvature_reg = pred_curvature_reg.clamp_min(self.eps)

            # [V4-3] 联合类别–属性对齐目标的 HVP 曲率（老师）
            hvp_curvature = self._compute_hvp_curvature(
                x, cls_sem, sem_per_part, lam_cls, lam_attr
            )

            student_curvature = self._normalize_curvature(pred_curvature_reg)
            teacher_curvature = self._normalize_curvature(
                hvp_curvature
            ).detach().to(dtype=student_curvature.dtype)

            curv_reg_loss = F.smooth_l1_loss(student_curvature, teacher_curvature)

        # 后续使用预测曲率
        curvature = self._normalize_curvature(pred_curvature)
        curv_logits = curvature.squeeze(-1) / self.curv_tau
        curv_prob = torch.softmax(curv_logits, dim=1)                          # (B,N) Σ=1
        curv_weight = (N * curv_prob).clamp(max=self.curv_weight_max)          # (B,N) 有界

        entropy_denom = torch.log(curv_prob.new_tensor(float(max(N, 2))))
        curv_entropy = -(
            curv_prob * curv_prob.clamp_min(self.eps).log()
        ).sum(dim=1).mean() / entropy_denom

        # [V4-4] 联合对齐损失：类别 + 属性，曲率概率加权（detach 防反馈回路）
        joint_align = lam_cls * (1.0 - s_cls) + lam_attr * (1.0 - per_token_sim)
        align_loss = (joint_align * curv_prob.detach()).sum(dim=1).mean()

        # [V4-6] 可选全局类别对齐项（V0 式图像摘要-类别语义对齐，默认关）
        cls_align_loss = pred_curvature.new_zeros(())
        if self.global_align_weight > 0:
            cls_align_loss = (
                1.0 - F.cosine_similarity(x.mean(dim=1), cls_sem.squeeze(1), dim=-1)
            ).mean()

        part_aux_loss = (
            align_loss
            + self.curv_reg_weight * curv_reg_loss
            + self.global_align_weight * cls_align_loss
        )

        # 特征增强：中心化 + 有界增益，恒正、有界、无全局放大
        feat_gain = self.feat_gain_max * self.curv_feat_alpha.sigmoid()        # (0, 0.25)
        centered_weight = curv_weight - curv_weight.mean(dim=1, keepdim=True)  # 零均值
        weighted_x = x * (1.0 + feat_gain * centered_weight.unsqueeze(-1))
        k = self.key_proj(weighted_x)
        v = self.value_proj(weighted_x)

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.sim_logit_alpha.tanh() * token_part_sim.transpose(1, 2)
        attn_logits = attn_logits + self.curv_logit_alpha.tanh() * torch.log1p(curvature).transpose(1, 2)
        attn = self.attn_drop(torch.softmax(attn_logits, dim=-1))
        part_tokens = attn @ v
        part_tokens = self.out_norm(
            self.out_proj(part_tokens)
            + self.part_queries.expand(B, -1, -1).to(part_tokens.dtype)
        )

        # Aggregate patch-level HVP/student curvature to the semantic-part level.
        # The detached value is used as an input prior by the bilevel policy; no
        # third-order gradient is allowed through the HVP computation.
        part_curvature = (
            attn.detach() * curvature.detach().transpose(1, 2)
        ).sum(dim=-1)
        part_curvature = part_curvature / part_curvature.mean(
            dim=1, keepdim=True
        ).clamp_min(self.eps)

        if return_aux:
            return part_tokens, {
                "align_loss": align_loss,
                "curv_reg_loss": curv_reg_loss,
                "cls_align_loss": cls_align_loss,       # [V4]
                "part_aux_loss": part_aux_loss,
                "curvature": curvature.detach(),
                "hvp_curvature": (
                    None if hvp_curvature is None
                    else self._normalize_curvature(hvp_curvature).detach()
                ),
                "curv_weight": curv_weight.detach(),
                "curv_weight_max": curv_weight.max().detach(),
                "curv_weight_mean": curv_weight.mean().detach(),
                "curv_entropy": curv_entropy.detach(),
                "part_curvature": part_curvature.detach(),
                "part_attn": attn.detach(),
                # V6 relation geometry needs a live attention tensor so the
                # ordinary relation loss can refine spatial part assignment.
                # The HVP teacher detaches it again before second derivatives.
                "part_attn_live": attn,
                "attr_attn": attr_attn.detach(),
                "part_assign": part_assign.detach(),
                "per_token_sim": per_token_sim.detach(),
                "s_cls": s_cls.detach(),                # [V4]
                "has_attr": has_attr,                    # [V4]
                "gate_status": self.gate_status(),
            }
        return part_tokens


# 若训练脚本按旧名导入，取消下一行注释即可无缝替换：
# SemanticPartTokenGenerator = SemanticPartTokenGeneratorV4
