# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.layers.helpers import to_2tuple

from semantic_part_v4 import SemanticPartTokenGeneratorV4


__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
    'deit_conv_small_patch16_224', 'deit_conv_base_patch16_224',
]


class HierVisionTransformer(VisionTransformer):
    def __init__(self, nb_classes, texts, *args, **kwargs):
        # ---- pop V4 (semantic part token) args before passing to VisionTransformer ----
        num_parts = kwargs.pop('num_parts', 8)
        enable_hvp = kwargs.pop('enable_hvp', True)
        lam_cls = kwargs.pop('lam_cls', 0.5)
        lam_attr = kwargs.pop('lam_attr', 0.5)
        global_align_weight = kwargs.pop('global_align_weight', 0.0)
        curv_reg_weight = kwargs.pop('curv_reg_weight', 0.05)
        text_dim = kwargs.pop('text_dim', 512)
        proto_align_weight = kwargs.pop('proto_align_weight', 0.1)

        super().__init__(*args, **kwargs)
        #self.pos_embed = nn.Parameter(torch.randn(1, self.embed_len, self.embed_dim) * .02)
        
        self.len_classes = len(nb_classes)
        print("nb_classes", nb_classes)
        self.num_classes = nb_classes[0]
        self.num_family = nb_classes[1]
        if self.len_classes == 3:
            self.num_manufacturer = nb_classes[2]
        self.texts = texts
        #####################
        self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.family_head = nn.Linear(self.embed_dim, self.num_family) if self.num_family > 0 else nn.Identity()
        self.family_head.apply(self._init_weights)
        if self.len_classes == 3:
            self.manufacturer_head = nn.Linear(self.embed_dim, self.num_manufacturer) if self.num_manufacturer > 0 else nn.Identity() 
            self.manufacturer_head.apply(self._init_weights)
            # [E2] part 作为零初始化残差：初始严格等价 baseline（tanh(0)=0，part 贡献为 0）
            self.part_adapter = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.embed_dim),
            )
            self.part_adapter.apply(self._init_weights)
            self.part_gate = nn.Parameter(torch.zeros(3))   # 第一轮只用 gate[0]（fine）

        #if self.texts is not None:
        self.feats_layer = nn.Linear(self.embed_dim*196, 512) 
        self.feats_layer.apply(self._init_weights)
        # HSG：科级/目级特征投影到 512 维文本空间（用于层级语义接地）
        self.family_proj = nn.Linear(self.embed_dim, 512)
        self.order_proj = nn.Linear(self.embed_dim, 512)
        self.family_proj.apply(self._init_weights)
        self.order_proj.apply(self._init_weights)

        # ---- V4: curvature-aware semantic part token generator ----
        self.num_parts = num_parts
        self.part_gen = SemanticPartTokenGeneratorV4(
            in_dim=self.embed_dim, embed_dim=self.embed_dim, num_parts=num_parts,
            enable_hvp=enable_hvp, lam_cls=lam_cls, lam_attr=lam_attr,
            global_align_weight=global_align_weight, curv_reg_weight=curv_reg_weight)
        # 类别语义 = 可学习全局身份 token；属性语义 = caption 文本(训练期) / 学习原型(推理期)
        self.category_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.attr_proto = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.caption_proj = nn.Linear(text_dim, self.embed_dim)
        trunc_normal_(self.category_token, std=.02)
        trunc_normal_(self.attr_proto, std=.02)
        self.caption_proj.apply(self._init_weights)
        self.proto_align_weight = proto_align_weight

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        B = x.shape[0]
 
        x = self.patch_embed(x)
  
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks

        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        # intermediate results
        intermediates = {}
  
        k = 0
        blks = [7, 9, 11]
        blk_id = {7:2, 9:3, 11:4}
        # #print("blks", blks)
        for blk in self.blocks:
            x = blk(x)
            if k in blks:
                intermediates[blk_id[k]] = x
            k += 1
       
        return intermediates

    def forward(self, x, caps_embed=None):
        intermediates = self.forward_features(x)
        if self.len_classes == 3:
            B = x.shape[0]
            # 用最后一层 patch tokens 作为 V4 的视觉输入
            x_tokens = intermediates[4][:, 1:]                     # (B, 196, embed_dim)
            category = self.category_token.expand(B, -1, -1)      # (B, 1, embed_dim)
            if caps_embed is not None:
                # 训练期：属性语义 = caption 文本投影
                attr_sem = self.caption_proj(caps_embed).unsqueeze(1)
            else:
                # 推理期：属性语义 = 学习到的文本原型（无逐样本文本，避免泄露类别）
                attr_sem = self.attr_proto.expand(B, -1, -1)

            part_tokens, aux = self.part_gen(x_tokens, [category, attr_sem], return_aux=True)
            part_feat = part_tokens.mean(dim=1)                    # (B, embed_dim)
            # [E2] part 作为残差 delta，零门控初始等价 baseline；第一轮只让 fine 层用 part
            delta = self.part_adapter(part_feat)                  # (B, embed_dim)
            gate = torch.tanh(self.part_gate)                     # (3,)，初始 0
            cls_s = self.norm(intermediates[4][:, 0])             # 物种级 CLS
            cls_f = self.norm(intermediates[3][:, 0])             # 科级 CLS
            cls_o = self.norm(intermediates[2][:, 0])             # 目级 CLS
            out = self.head(cls_s + gate[0] * delta)
            family_out = self.family_head(cls_f)
            manu_out = self.manufacturer_head(cls_o)

            feats = intermediates[4][:, 1:]
            feats = self.feats_layer(feats.view(feats.size(0), -1))

            # HSG：科级/目级特征（各自 CLS → 512 文本空间）
            family_feat = self.family_proj(cls_f)   # (B, 512)
            order_feat = self.order_proj(cls_o)     # (B, 512)

            part_aux_loss = aux['part_aux_loss']
            # 训练期把属性原型拉向 caption 均值，使推理期的原型有意义
            if caps_embed is not None and self.proto_align_weight > 0:
                proto = self.caption_proj(caps_embed).mean(dim=0, keepdim=True)   # (1, embed_dim)
                proto_align = (1.0 - F.cosine_similarity(self.attr_proto, proto, dim=-1)).mean()
                part_aux_loss = part_aux_loss + self.proto_align_weight * proto_align

            return out, family_out, manu_out, feats, family_feat, order_feat, part_aux_loss

        else:
            out = self.norm(intermediates[3][:, 0])
            out = self.head(out)
            family_out = self.family_head(intermediates[4][:, 0])
            return out, family_out

    @torch.jit.ignore
    def no_weight_decay(self):
        # 门控(0 维标量)/可学习 token 不落入 timm 的 decay 组（timm 只用 len(shape)==1 判定）
        nd = set(super().no_weight_decay())
        nd.update({'category_token', 'attr_proto', 'part_gate'})
        for name, _ in self.part_gen.named_parameters():
            if any(k in name for k in ('alpha', 'gamma', 'part_queries')):
                nd.add('part_gen.' + name)
        return nd


class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, nb_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

        self.num_classes = nb_classes[0]
        self.num_family = nb_classes[1]
        self.num_manufacturer = nb_classes[2]

        #####################
        #self.head = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.family_head = nn.Linear(self.embed_dim, self.num_family) if self.num_family > 0 else nn.Identity()
        self.manufacturer_head = nn.Linear(self.embed_dim, self.num_manufacturer) if self.num_manufacturer > 0 else nn.Identity()
        self.family_head.apply(self._init_weights)
        self.manufacturer_head.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        k = 0
        for blk in self.blocks:
            x = blk(x)
            k += 1

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


class ConvStem(nn.Module):
    """
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 16, 'ConvStem only supports patch size of 16'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 8 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = HierVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(nb_classes, texts=None, pretrained=False, **kwargs):
    model = HierVisionTransformer(
        nb_classes=nb_classes,
        texts=texts,
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = HierVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model_dict = model.state_dict() 
        pretrained_dict = {k: v for k, v in checkpoint["model"].items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)  
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_conv_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_conv_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model

