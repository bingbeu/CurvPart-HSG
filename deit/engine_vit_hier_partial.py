# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch
import torch.distributed as dist

from mixup_hier import Mixup # we do not use mixup here. 
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

import torch.nn.functional as F
import time
from collections import Counter


def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, meta_optimizer=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()

    # HSG：从数据集取科级/目级文本语义（冻结，训练期层级接地用）
    z_family = getattr(data_loader.dataset, 'z_family', None)
    z_order = getattr(data_loader.dataset, 'z_order', None)
    if z_family is not None:
        z_family = z_family.to(device)
        z_order = z_order.to(device)


    for data in metric_logger.log_every(data_loader, print_freq, header):
        #if args.texts is not None:
        samples, targets, fine_targets, sub_targets, basic_targets, caps_embed = data
        caps_embed = caps_embed.to(device, non_blocking=True)
        caps_model_embed = caps_embed
        # else:
        #     samples, targets, fine_targets, sub_targets, basic_targets = data
        if isinstance(samples, (tuple, list)):
            if len(samples) != 2:
                raise ValueError("bilevel training expects exactly two augmented views")
            support_samples = samples[0].to(device, non_blocking=True)
            query_samples = samples[1].to(device, non_blocking=True)
        else:
            support_samples = samples.to(device, non_blocking=True)
            query_samples = None
        targets = targets.to(device, non_blocking=True)
        fine_targets = fine_targets.to(device, non_blocking=True)
        sub_targets = sub_targets.to(device, non_blocking=True)
        basic_targets = basic_targets.to(device, non_blocking=True)
        

        if 'BIRD' in args.data_set:
            leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
            sub_labels = torch.nonzero(targets > 12, as_tuple=False)

        elif 'IMNET-F' in args.data_set:
            leaf_labels = torch.nonzero(targets > 146, as_tuple=False)
            sub_labels = torch.nonzero(targets > 19, as_tuple=False)    

        elif 'AIR' in args.data_set:
            leaf_labels = torch.nonzero(targets > 99, as_tuple=False)
            sub_labels = torch.nonzero(targets > 29, as_tuple=False)    
        elif 'INAT21' in args.data_set:
                leaf_labels = torch.nonzero(targets > 1375, as_tuple=False)
                sub_labels = torch.nonzero(targets > 272, as_tuple=False)     
        
        else:
            raise ValueError('Unknown dataset')

        with torch.cuda.amp.autocast():
            out = model(support_samples, caps_embed)
            sim_loss = torch.tensor(0.0)  
            outputs, sub_out, basic_out, feats, family_feat, order_feat, part_aux_loss, *extra = out
            support_meta_state = extra[0] if extra else None

            feats = feats / feats.norm(dim=-1, keepdim=True)
            caps_embed = caps_embed / caps_embed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            labels = torch.arange(len(targets)).to(device)
            logits = torch.matmul(feats, caps_embed.t()) 
            loss_i = F.cross_entropy(logits, labels)
            loss_t = F.cross_entropy(logits.t(), labels)

            # Text-attr loss (CLIP style)
            sim_loss = (loss_i + loss_t) / 2

            # HSG：科级/目级语义接地 —— 类原型分类（导师 E1，消除 batch 内 false negative）
            if z_family is not None:
                ff = F.normalize(family_feat, dim=-1)
                zf = F.normalize(z_family, dim=-1)          # (num_family, D)
                family_logits = ff @ zf.t() / 0.07          # (B, num_family)
                fam_sem_loss = F.cross_entropy(family_logits, sub_targets)

                of = F.normalize(order_feat, dim=-1)
                zo = F.normalize(z_order, dim=-1)           # (num_order, D)
                order_logits = of @ zo.t() / 0.07           # (B, num_order)
                ord_sem_loss = F.cross_entropy(order_logits, basic_targets)
            else:
                fam_sem_loss = torch.tensor(0.0, device=device)
                ord_sem_loss = torch.tensor(0.0, device=device)
   
            
            loss_fine = torch.zeros((), device=device)
            loss_sub = torch.zeros((), device=device)
            loss_basic = torch.zeros((), device=device)

            if leaf_labels.shape[0] > 0:
                # supervision for samples who have fine-grained labels. 
                leaf_index = leaf_labels.flatten()
                select_leaf_output = torch.index_select(outputs, 0, leaf_index)
                select_leaf_labels = torch.index_select(fine_targets, 0, leaf_index)
                loss_fine += (F.cross_entropy(select_leaf_output, select_leaf_labels))
    
            if sub_labels.shape[0] > 0:
                # supervision for samples who have subordinate labels. 
                sub_index = sub_labels.flatten()
                select_sub_labels = torch.index_select(sub_targets, 0, sub_index)
                select_sub_output = torch.index_select(sub_out, 0, sub_index)
                loss_sub += (F.cross_entropy(select_sub_output, select_sub_labels))

            loss_basic = (F.cross_entropy(basic_out, basic_targets))

        meta_stats = {}
        meta_real_loss = torch.zeros((), device=device)
        meta_active = (
            getattr(args, 'enable_bilevel', False)
            and epoch >= getattr(args, 'meta_start_epoch', 0)
        )
        if meta_active:
            if query_samples is None:
                raise RuntimeError(
                    "--enable-bilevel requires TwoViewTransform for the training dataset"
                )
            if meta_optimizer is None:
                raise RuntimeError("meta optimizer was not constructed")
            core_model = model.module if hasattr(model, 'module') else model
            if support_meta_state is None:
                raise RuntimeError("model did not return a bilevel semantic state")

            # Query features are evaluation evidence for the virtual support
            # update.  Backbone gradients are intentionally disabled here.
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    query_out = model(query_samples, caps_model_embed, compute_hvp=False)
                query_meta_state = query_out[-1]

            # Meta step: exact hypergradient of the one-step unrolled objective.
            # Only phi (the policy) is updated; theta/psi are untouched here.
            with torch.cuda.amp.autocast(enabled=False):
                meta_loss, meta_stats = core_model.bilevel.meta_objective(
                    support_meta_state,
                    query_meta_state,
                    inner_lr=args.meta_inner_lr,
                    q_mode=args.meta_q,
                    kl_weight=args.meta_kl_weight,
                )
            policy_params = tuple(core_model.bilevel.policy.parameters())
            policy_grads = torch.autograd.grad(
                meta_loss, policy_params, allow_unused=False
            )
            meta_optimizer.zero_grad(set_to_none=True)
            for param, grad in zip(policy_params, policy_grads):
                param.grad = grad.detach()
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                    param.grad.div_(dist.get_world_size())
            if args.meta_clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(policy_params, args.meta_clip_grad)
            meta_optimizer.step()
            meta_optimizer.zero_grad(set_to_none=True)

            # Real model step: p is recomputed after the meta update and detached
            # inside this loss, so no direct weighted-error gradient reaches phi.
            meta_real_loss, real_stats = core_model.bilevel.real_weighted_alignment(
                support_meta_state
            )
            meta_stats.update(real_stats)

        loss = (
            loss_fine
            + loss_sub
            + loss_basic
            + sim_loss * args.sim_loss_weight
            + part_aux_loss * args.part_aux_weight
            + fam_sem_loss * args.family_sem_weight
            + ord_sem_loss * args.order_sem_weight
            + meta_real_loss * getattr(args, 'meta_real_weight', 0.0)
        )
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(sp_loss=loss_fine.item())
        metric_logger.update(subord_loss=loss_sub.item())
        metric_logger.update(basic_loss=loss_basic.item())
        metric_logger.update(sim_loss=sim_loss.item())
        metric_logger.update(part_aux_loss=part_aux_loss.item())
        if meta_stats:
            for key, value in meta_stats.items():
                metric_logger.update(**{key: value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        del feats, caps_embed, caps_model_embed, loss_i, loss_t, sim_loss, logits, support_samples
        if query_samples is not None:
            del query_samples
        del samples, targets, outputs, loss
        torch.cuda.empty_cache()

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
   

@torch.no_grad()
def evaluate(data_loader, model, device, n_classes=3, texts=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target, sub_targets, basic_targets in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sub_targets = sub_targets.to(device, non_blocking=True)
        basic_targets = basic_targets.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output, sub_out, basic_out, *rest  = model(images)

            loss_fine = criterion(output, target)
            loss_sub = criterion(sub_out, sub_targets)
            loss_basic = criterion(basic_out, basic_targets)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        sub_acc1, sub_acc5 = accuracy(sub_out, sub_targets, topk=(1, 5))
        basic_acc1, basic_acc5 = accuracy(basic_out, basic_targets, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(sploss=loss_fine.item())
        metric_logger.update(subordloss=loss_sub.item())
        metric_logger.update(manuloss=loss_basic.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['sub_acc1'].update(sub_acc1.item(), n=batch_size)
        metric_logger.meters['basic_acc1'].update(basic_acc1.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} sub@1 {subtop1.global_avg:.3f}' 
        ' manu@1 {manutop1.global_avg:.3f} sploss {losses.global_avg:.3f} fmloss {fmlosses.global_avg:.3f} basicloss {basiclosses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.sploss, fmlosses=metric_logger.subordloss, basiclosses=metric_logger.manuloss,
                subtop1=metric_logger.sub_acc1, manutop1=metric_logger.basic_acc1))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
