from __future__ import annotations

import argparse
import json

import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataloader import *
from utils.metrics_soft import compute_all_metrics
from modules.PAM_MIL import PAM_MIL

CLASS_NAMES = ["Basal", "LumA", "LumB"]
EPS = 1e-8


def set_global_reproducibility(seed: int, deterministic: bool = True, warn_only: bool = True) -> None:
    """Set all RNGs and deterministic backend flags used by this script."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    if deterministic and hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=warn_only)


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker deterministically."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int, fold: int, offset: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(int(seed) + 10007 * int(fold) + int(offset))
    return g


def set_model_view(model: nn.Module, view_index: int, num_views: int) -> None:
    module = model.module if hasattr(model, "module") else model
    preproc = getattr(module, "preproc", None)
    if preproc is not None and hasattr(preproc, "set_eval_view"):
        preproc.set_eval_view(view_index=view_index, num_views=num_views)



class PAM50RegressionLoss(nn.Module):
    def __init__(
            self,
            alpha_Basal: float = 3.0,
            alpha_LumA: float = 1.0,
            alpha_LumB: float = 1.8,
            loss_type: str = "mse",
            huber_beta: float = 0.10,
            use_conf_weight: bool = False,
            conf_weight_min: float = 0.35,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "class_weights",
            torch.tensor([alpha_Basal, alpha_LumA, alpha_LumB], dtype=torch.float32),
        )
        self.loss_type = str(loss_type).lower()
        self.huber_beta = float(huber_beta)
        self.use_conf_weight = bool(use_conf_weight)
        self.conf_weight_min = float(conf_weight_min)
        if self.loss_type not in {"mse", "huber"}:
            raise ValueError("--reg_loss must be 'mse' or 'huber'")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor,
                diff12: Optional[torch.Tensor] = None) -> torch.Tensor:
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        preds = preds.float()
        targets = targets.float()

        if self.loss_type == "huber":
            per_class = F.smooth_l1_loss(preds, targets, reduction="none", beta=self.huber_beta)
        else:
            per_class = (preds - targets).pow(2)

        weights = self.class_weights.to(device=preds.device, dtype=preds.dtype).view(1, 3)
        per_sample = (per_class * weights).mean(dim=1)

        if self.use_conf_weight and diff12 is not None:
            diff12 = diff12.to(device=preds.device, dtype=preds.dtype).view(-1)
            sw = self.conf_weight_min + (1.0 - self.conf_weight_min) * diff12.clamp(0.0, 1.0)
            per_sample = per_sample * sw

        return per_sample.mean()


class OnlineSoftLabelRankLoss(nn.Module):

    def __init__(
            self,
            n_classes: int = 3,
            memory_size: int = 512,
            target_margin: float = 0.05,
            score_margin: float = 0.02,
            temperature: float = 10.0,
            class_weights: Tuple[float, float, float] = (2.0, 1.0, 1.5),
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.memory_size = int(memory_size)
        self.target_margin = float(target_margin)
        self.score_margin = float(score_margin)
        self.temperature = float(temperature)
        self.register_buffer("pred_bank", torch.empty(0, self.n_classes), persistent=False)
        self.register_buffer("target_bank", torch.empty(0, self.n_classes), persistent=False)
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def reset(self) -> None:
        device = self.pred_bank.device
        self.pred_bank = torch.empty(0, self.n_classes, device=device)
        self.target_bank = torch.empty(0, self.n_classes, device=device)

    @torch.no_grad()
    def _update_memory(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        if self.memory_size <= 0:
            return
        p = preds.detach()
        y = targets.detach()
        if self.pred_bank.numel() == 0:
            new_p, new_y = p, y
        else:
            new_p = torch.cat([self.pred_bank.to(preds.device), p], dim=0)
            new_y = torch.cat([self.target_bank.to(targets.device), y], dim=0)
        if new_p.shape[0] > self.memory_size:
            new_p = new_p[-self.memory_size:]
            new_y = new_y[-self.memory_size:]
        self.pred_bank = new_p
        self.target_bank = new_y

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        preds = preds.float()
        targets = targets.float()

        if self.pred_bank.numel() == 0:
            self._update_memory(preds, targets)
            return preds.new_tensor(0.0), {"pairs": 0, "memory": int(self.pred_bank.shape[0])}

        old_p = self.pred_bank.to(preds.device)
        old_y = self.target_bank.to(targets.device)
        weights = self.class_weights.to(device=preds.device, dtype=preds.dtype)
        temp = max(self.temperature, EPS)
        losses = []
        active_w = []
        total_pairs = 0

        for c in range(self.n_classes):
            cur_p = preds[:, c].view(-1, 1)
            cur_y = targets[:, c].view(-1, 1)
            mem_p = old_p[:, c].view(1, -1)
            mem_y = old_y[:, c].view(1, -1)

            cur_higher = cur_y > (mem_y + self.target_margin)
            mem_higher = mem_y > (cur_y + self.target_margin)
            cls_terms = []
            if torch.any(cur_higher):
                violation = mem_p + self.score_margin - cur_p
                cls_terms.append(F.softplus(violation[cur_higher] * temp).mean() / temp)
                total_pairs += int(cur_higher.sum().item())
            if torch.any(mem_higher):
                violation = cur_p + self.score_margin - mem_p
                cls_terms.append(F.softplus(violation[mem_higher] * temp).mean() / temp)
                total_pairs += int(mem_higher.sum().item())
            if cls_terms:
                losses.append(torch.stack(cls_terms).mean() * weights[c])
                active_w.append(weights[c])

        if losses:
            loss = torch.stack(losses).sum() / torch.stack(active_w).sum().clamp_min(EPS)
        else:
            loss = preds.new_tensor(0.0)

        self._update_memory(preds, targets)
        return loss, {"pairs": int(total_pairs), "memory": int(self.pred_bank.shape[0])}


class OnlineCorrelationLoss(nn.Module):
    def __init__(
            self,
            n_classes: int = 3,
            memory_size: int = 512,
            min_samples: int = 32,
            class_weights: Tuple[float, float, float] = (1.2, 1.0, 1.1),
    ) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.memory_size = int(memory_size)
        self.min_samples = int(min_samples)
        self.register_buffer("pred_bank", torch.empty(0, self.n_classes), persistent=False)
        self.register_buffer("target_bank", torch.empty(0, self.n_classes), persistent=False)
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32), persistent=False)

    @torch.no_grad()
    def reset(self) -> None:
        device = self.pred_bank.device
        self.pred_bank = torch.empty(0, self.n_classes, device=device)
        self.target_bank = torch.empty(0, self.n_classes, device=device)

    @torch.no_grad()
    def _update_memory(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        p = preds.detach()
        y = targets.detach()
        if self.pred_bank.numel() == 0:
            new_p, new_y = p, y
        else:
            new_p = torch.cat([self.pred_bank.to(preds.device), p], dim=0)
            new_y = torch.cat([self.target_bank.to(targets.device), y], dim=0)
        if new_p.shape[0] > self.memory_size:
            new_p = new_p[-self.memory_size:]
            new_y = new_y[-self.memory_size:]
        self.pred_bank = new_p
        self.target_bank = new_y

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, int]]:
        if preds.ndim == 1:
            preds = preds.unsqueeze(0)
        if targets.ndim == 1:
            targets = targets.unsqueeze(0)
        preds = preds.float()
        targets = targets.float()
        if self.pred_bank.numel() == 0:
            all_p, all_y = preds, targets
        else:
            all_p = torch.cat([self.pred_bank.to(preds.device), preds], dim=0)
            all_y = torch.cat([self.target_bank.to(targets.device), targets], dim=0)

        n = int(all_p.shape[0])
        if n < self.min_samples:
            loss = preds.new_tensor(0.0)
        else:
            xm = all_p - all_p.mean(dim=0, keepdim=True)
            ym = all_y - all_y.mean(dim=0, keepdim=True)
            corr = (xm * ym).mean(dim=0) / torch.sqrt(xm.pow(2).mean(dim=0) * ym.pow(2).mean(dim=0) + EPS)
            weights = self.class_weights.to(device=preds.device, dtype=preds.dtype)
            loss = ((1.0 - corr) * weights).sum() / weights.sum().clamp_min(EPS)
        self._update_memory(preds, targets)
        return loss, {"n": n, "memory": int(self.pred_bank.shape[0])}


def scheduled_scale(epoch: int, warmup_epochs: int, ramp_epochs: int, max_scale: float) -> float:
    warmup_epochs = max(0, int(warmup_epochs))
    ramp_epochs = max(0, int(ramp_epochs))
    max_scale = float(max_scale)
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return max_scale
    p = min(1.0, max(0.0, (epoch - warmup_epochs + 1.0) / float(ramp_epochs)))
    p = p * p * (3.0 - 2.0 * p)  # smoothstep
    return float(max_scale * p)


def unpack_batch(data: Any, device: torch.device) -> Tuple[
    torch.Tensor, torch.Tensor, List[str], Optional[torch.Tensor], int]:
    bag = data[0]
    target = data[1].to(device, dtype=torch.float32)
    names = list(data[2]) if isinstance(data[2], (list, tuple)) else [str(data[2])]
    diff12 = data[3].to(device, dtype=torch.float32) if len(data) > 3 else None

    if isinstance(bag, (list, tuple)):
        raise TypeError("This optimized main expects one tensor bag per slide, not a list/tuple of bags.")
    bag = bag.to(device)
    batch_size = int(bag.size(0))
    return bag, target, names, diff12, batch_size


def make_weighted_sampler(train_strat_labels: Sequence[Any], args: argparse.Namespace,
                          fold: int) -> WeightedRandomSampler:
    weight_map = {
        "Basal": float(args.sampler_Basal),
        "LumA": float(args.sampler_LumA),
        "LumB": float(args.sampler_LumB),
        0: float(args.sampler_Basal),
        1: float(args.sampler_LumA),
        2: float(args.sampler_LumB),
    }
    weights = torch.tensor([weight_map.get(str(x), weight_map.get(x, 1.0)) for x in train_strat_labels],
                           dtype=torch.float32)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=make_generator(args.seed, fold, offset=778441440),
    )


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model != "PAM_MIL":
        raise NotImplementedError("This optimized script intentionally supports only --model spatial_mamba.")
    model = PAM_MIL(
        in_dim=args.num_feats_ab,
        dropout=args.dropout,
        act=args.act,
        layer=args.mambamil_layer,
        rate=args.mambamil_rate,
        type=args.mambamil_type,
        train_sampling_mode=args.train_sampling_mode,
        max_patches=args.model_nmax,
        coverage_bins=args.coverage_bins,
        coverage_per_bin=args.coverage_per_bin,
        saliency_ratio=args.saliency_ratio,
    ).to(device)
    return model


def build_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.lr_sche == "const":
        return None
    if args.lr_sche == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=args.min_lr)
    if args.lr_sche == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.num_epoch // 2), gamma=0.2)
    raise ValueError(f"Unknown lr_sche: {args.lr_sche}")


class EarlyStopper:
    def __init__(self, patience: int = 20, min_epoch: int = 30) -> None:
        self.patience = int(patience)
        self.min_epoch = int(min_epoch)
        self.best = -float("inf")
        self.best_epoch = -1
        self.num_bad = 0

    def step(self, epoch: int, score: float) -> bool:
        if not np.isfinite(score):
            return False
        if score > self.best + 1e-8:
            self.best = float(score)
            self.best_epoch = int(epoch)
            self.num_bad = 0
        else:
            self.num_bad += 1
        return epoch + 1 >= self.min_epoch and self.num_bad >= self.patience


def get_monitor_score(metrics: Dict[str, float], args: argparse.Namespace) -> float:
    macro_auc = float(metrics.get("auroc_ovr_macro", np.nan))
    pearson = float(metrics.get("pearson_macro_overall", 0.0))
    basal_auc = float(metrics.get("auroc_ovr_Basal", 0.0))
    if not np.isfinite(macro_auc):
        return float("nan")
    if args.monitor_metric == "auroc_ovr_macro":
        return macro_auc
    if args.monitor_metric == "pearson_macro_overall":
        return pearson
    # Composite remains dominated by macro-AUC but avoids selecting a checkpoint
    # with very poor continuous-score fitting.
    return macro_auc + args.monitor_pearson_weight * pearson + args.monitor_basal_weight * basal_auc



def forward_regression(args: argparse.Namespace, model: nn.Module, bag: torch.Tensor) -> torch.Tensor:
    out = model(bag)
    preds = out[0] if isinstance(out, (tuple, list)) else out
    if preds.ndim == 1:
        preds = preds.unsqueeze(0)
    return preds


def forward_regression_multiview(args: argparse.Namespace, model: nn.Module, bag: torch.Tensor) -> torch.Tensor:
    eval_views = max(1, int(args.eval_views))
    preds_all = []
    for v in range(eval_views):
        set_model_view(model, v, eval_views)
        preds_all.append(forward_regression(args, model, bag))
    set_model_view(model, 0, 1)
    if len(preds_all) == 1:
        return preds_all[0]
    return torch.stack(preds_all, dim=0).mean(dim=0)


def train_one_epoch(
        args: argparse.Namespace,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: PAM50RegressionLoss,
        rank_loss: Optional[OnlineSoftLabelRankLoss],
        corr_loss: Optional[OnlineCorrelationLoss],
        device: torch.device,
        epoch: int,
        scaler: Optional[GradScaler],
) -> Dict[str, float]:
    model.train()
    if rank_loss is not None:
        rank_loss.reset()
    if corr_loss is not None:
        corr_loss.reset()

    if args.train_sampling_mode == "deterministic" and args.train_views > 1:
        set_model_view(model, epoch % int(args.train_views), int(args.train_views))
    else:
        set_model_view(model, 0, 1)

    totals = {"loss": 0.0, "reg": 0.0, "rank": 0.0, "corr": 0.0, "rank_pairs": 0.0, "n": 0.0}
    optimizer.zero_grad(set_to_none=True)
    rank_scale = scheduled_scale(epoch, args.rank_warmup_epochs, args.rank_ramp_epochs, args.lambda_soft_rank)
    corr_scale = scheduled_scale(epoch, args.corr_warmup_epochs, args.corr_ramp_epochs, args.lambda_corr)

    autocast_ctx = torch.cuda.amp.autocast if args.amp else nullcontext

    for i, data in enumerate(loader):
        bag, target, _names, diff12, batch_size = unpack_batch(data, device)
        with autocast_ctx():
            preds = forward_regression(args, model, bag)
            loss_reg = criterion(preds, target, diff12=diff12)
            total_loss = loss_reg

            if rank_loss is not None and rank_scale > 0:
                loss_rank, debug = rank_loss(preds, target)
                total_loss = total_loss + rank_scale * loss_rank
                rank_pairs = float(debug.get("pairs", 0))
            else:
                loss_rank = preds.new_tensor(0.0)
                rank_pairs = 0.0

            if corr_loss is not None and corr_scale > 0:
                loss_corr, _ = corr_loss(preds, target)
                total_loss = total_loss + corr_scale * loss_corr
            else:
                loss_corr = preds.new_tensor(0.0)

            loss_for_backward = total_loss / max(1, args.accumulation_steps)

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(loader):
                if args.clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_for_backward.backward()
            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == len(loader):
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        totals["loss"] += float(total_loss.detach().item()) * batch_size
        totals["reg"] += float(loss_reg.detach().item()) * batch_size
        totals["rank"] += float(loss_rank.detach().item()) * batch_size
        totals["corr"] += float(loss_corr.detach().item()) * batch_size
        totals["rank_pairs"] += rank_pairs
        totals["n"] += batch_size

        if (i % args.log_iter == 0 or i == len(loader) - 1) and not args.no_log:
            lr = float(np.mean([pg["lr"] for pg in optimizer.param_groups]))
            denom = max(totals["n"], 1.0)
            print(
                f"  [{i:04d}/{len(loader) - 1:04d}] "
                f"loss={totals['loss'] / denom:.5f} reg={totals['reg'] / denom:.5f} "
                f"rank={totals['rank'] / denom:.5f}*{rank_scale:.3g} "
                f"corr={totals['corr'] / denom:.5f}*{corr_scale:.3g} "
                f"pairs={int(totals['rank_pairs'])} lr={lr:.2e}"
            )

    denom = max(totals["n"], 1.0)
    return {k: (v / denom if k not in {"rank_pairs", "n"} else v) for k, v in totals.items()}


@torch.no_grad()
def inference_loop(
        args: argparse.Namespace,
        model: nn.Module,
        loader: DataLoader,
        criterion: PAM50RegressionLoss,
        device: torch.device,
) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_names: List[str] = []
    loss_sum = 0.0
    n_sum = 0

    for data in loader:
        bag, target, names, diff12, batch_size = unpack_batch(data, device)
        preds = forward_regression_multiview(args, model, bag)
        loss = criterion(preds, target, diff12=diff12)
        loss_sum += float(loss.item()) * batch_size
        n_sum += batch_size
        all_preds.extend(preds.detach().cpu().float().numpy())
        all_targets.extend(target.detach().cpu().float().numpy())
        all_names.extend(names)

    preds_np = np.asarray(all_preds, dtype=np.float32)
    targets_np = np.asarray(all_targets, dtype=np.float32)
    metrics = compute_all_metrics(preds_np, targets_np) if len(preds_np) else {}
    return metrics, loss_sum / max(1, n_sum), preds_np, targets_np, all_names


# ════════════════════════════════════════════════════════════════════════════
# Fold logic
# ════════════════════════════════════════════════════════════════════════════


def run_fold(
        args: argparse.Namespace,
        fold: int,
        train_slides: np.ndarray,
        train_labels: np.ndarray,
        train_diff12: np.ndarray,
        train_strat_labels: np.ndarray,
        val_slides: np.ndarray,
        val_labels: np.ndarray,
        val_diff12: np.ndarray,
        test_slides: np.ndarray,
        test_labels: np.ndarray,
        test_diff12: np.ndarray,
) -> Dict[str, float]:
    set_global_reproducibility(args.seed + fold, args.deterministic, args.deterministic_warn_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = TCGADataset(
        file_name=train_slides,
        file_label=train_labels,
        file_diff12=train_diff12,
        max_patch=args.tcga_max_patch,
        root=args.dataset_root,
        persistence=args.persistence,
        is_train=True,
    )
    val_set = TCGADataset(
        file_name=val_slides,
        file_label=val_labels,
        file_diff12=val_diff12,
        max_patch=args.tcga_max_patch,
        root=args.dataset_root,
        persistence=args.persistence,
    )
    test_set = TCGADataset(
        file_name=test_slides,
        file_label=test_labels,
        file_diff12=test_diff12,
        max_patch=args.tcga_max_patch,
        root=args.dataset_root,
        persistence=args.persistence,
    )

    if args.use_weighted_sampler:
        sampler = make_weighted_sampler(train_strat_labels, args, fold)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed, fold, offset=1000),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed, fold, offset=2000),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=make_generator(args.seed, fold, offset=3000),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = build_model(args, device)
    criterion = PAM50RegressionLoss(
        alpha_Basal=args.alpha_Basal,
        alpha_LumA=args.alpha_LumA,
        alpha_LumB=args.alpha_LumB,
        loss_type=args.reg_loss,
        huber_beta=args.huber_beta,
        use_conf_weight=args.use_conf_weight,
        conf_weight_min=args.conf_weight_min,
    ).to(device)

    rank_loss = OnlineSoftLabelRankLoss(
        n_classes=3,
        memory_size=args.soft_rank_memory_size,
        target_margin=args.soft_rank_target_margin,
        score_margin=args.soft_rank_score_margin,
        temperature=args.soft_rank_temperature,
        class_weights=(args.soft_rank_weight_Basal, args.soft_rank_weight_LumA, args.soft_rank_weight_LumB),
    ).to(device) if args.lambda_soft_rank > 0 else None

    corr_loss = OnlineCorrelationLoss(
        n_classes=3,
        memory_size=args.corr_memory_size,
        min_samples=args.corr_min_samples,
        class_weights=(args.corr_weight_Basal, args.corr_weight_LumA, args.corr_weight_LumB),
    ).to(device) if args.lambda_corr > 0 else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(args, optimizer)
    scaler = GradScaler() if args.amp else None
    stopper = EarlyStopper(patience=args.patience, min_epoch=args.stop_epoch) if args.early_stopping else None

    fold_dir = os.path.join(args.model_path, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    # best_model_path = os.path.join(fold_dir, "model_best.pt")
    best_model_path = os.path.join(args.model_path, f"fold_{fold}_model_best.pt")
    best_score = -float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []

    if not args.no_log:
        print(f"\n========== Fold {fold}/{args.cv_fold - 1} ==========")
        print(f"Train={len(train_set)}  Val={len(val_set)}  Test={len(test_set)}  Device={device}")

    for epoch in range(args.num_epoch):
        t0 = time.time()
        train_log = train_one_epoch(
            args=args,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            rank_loss=rank_loss,
            corr_loss=corr_loss,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )
        val_metrics, val_loss, _, _, _ = inference_loop(args, model, val_loader, criterion, device)
        monitor = get_monitor_score(val_metrics, args)

        if np.isfinite(monitor) and monitor > best_score:
            best_score = float(monitor)
            best_epoch = int(epoch)
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "monitor_score": best_score,
                    "args": vars(args),
                    "val_metrics": val_metrics,
                },
                best_model_path,
            )

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch + 1,
            "train_loss": train_log["loss"],
            "train_reg": train_log["reg"],
            "train_rank": train_log["rank"],
            "train_corr": train_log["corr"],
            "val_loss": val_loss,
            "val_monitor": monitor,
            "val_pearson_macro_overall": float(val_metrics.get("pearson_macro_overall", np.nan)),
            "val_auroc_ovr_macro": float(val_metrics.get("auroc_ovr_macro", np.nan)),
            "val_auroc_ovr_Basal": float(val_metrics.get("auroc_ovr_Basal", np.nan)),
            "lr": float(np.mean([pg["lr"] for pg in optimizer.param_groups])),
            "time_sec": time.time() - t0,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)

        if not args.no_log:
            print(
                f"Epoch [{epoch + 1:03d}/{args.num_epoch}] "
                f"train={row['train_loss']:.5f} val_loss={val_loss:.5f} "
                f"PearsonOA={row['val_pearson_macro_overall']:.4f} "
                f"OvR-MacroAUC={row['val_auroc_ovr_macro']:.4f} "
                f"BasalAUC={row['val_auroc_ovr_Basal']:.4f} "
                f"monitor={monitor:.4f} best={best_score:.4f}@{best_epoch + 1}"
            )

        if args.wandb and wandb is not None:
            wandb.log({f"fold{fold}/{k}": v for k, v in row.items()})

        if stopper is not None and stopper.step(epoch, monitor):
            if not args.no_log:
                print(f"Early stopping at epoch {epoch + 1}; best epoch = {best_epoch + 1}")
            break

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        torch.save({"model": model.state_dict(), "epoch": args.num_epoch - 1, "args": vars(args)}, best_model_path)

    test_metrics, test_loss, test_preds, test_targets, test_names = inference_loop(args, model, test_loader, criterion,
                                                                                   device)
    pred_cls = np.argmax(test_preds, axis=1).astype(int)
    true_cls = np.argmax(test_targets, axis=1).astype(int)

    df_pred = pd.DataFrame({
        "slide_id": test_names,
        "true_Basal": test_targets[:, 0],
        "true_LumA": test_targets[:, 1],
        "true_LumB": test_targets[:, 2],
        "pred_Basal": test_preds[:, 0],
        "pred_LumA": test_preds[:, 1],
        "pred_LumB": test_preds[:, 2],
        "true_decision": [CLASS_NAMES[i] for i in true_cls],
        "pred_decision": [CLASS_NAMES[i] for i in pred_cls],
    })
    df_pred.to_csv(os.path.join(args.model_path, f"TEST_RESULT_FOLD_{fold}.csv"), index=False)
    df_pred.to_csv(os.path.join(fold_dir, "test_predictions.csv"), index=False)

    fold_row = {"fold": fold, "best_epoch": best_epoch + 1, "best_val_monitor": best_score, "test_loss": test_loss}
    fold_row.update({f"test_{k}": v for k, v in test_metrics.items()})
    if not args.no_log:
        print(
            f"[Fold {fold}] Test Pearson-OA={test_metrics.get('pearson_macro_overall', np.nan):.4f}  "
            f"Test OvR-MacroAUC={test_metrics.get('auroc_ovr_macro', np.nan):.4f}"
        )
    return fold_row


# ════════════════════════════════════════════════════════════════════════════
# Main CV
# ════════════════════════════════════════════════════════════════════════════


def resolve_project_path(args: argparse.Namespace) -> None:
    args.model_path = os.path.join(args.model_path, args.project, args.title)
    os.makedirs(args.model_path, exist_ok=True)


def main(args: argparse.Namespace) -> None:
    resolve_project_path(args)
    set_global_reproducibility(args.seed, args.deterministic, args.deterministic_warn_only)

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.datasets.lower() != "tcga_pam50":
        raise NotImplementedError("This optimized script supports --datasets tcga_pam50 only.")

    label_path = args.label_csv if args.label_csv else os.path.join(args.dataset_root,
                                                                    "Matched_Result_unique_slides_final_label.csv")
    slides, soft_labels, strat_labels, groups, diff12 = get_patient_label(label_path)

    if args.cv_fold <= 1:
        raise ValueError("--cv_fold must be > 1")

    split = get_kfold_soft(
        k=args.cv_fold,
        slides_array=slides,
        soft_labels_array=soft_labels,
        strat_labels=strat_labels,
        groups=groups,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    (
        train_slides,
        train_labels,
        train_diff12,
        train_strat,
        test_slides,
        test_labels,
        test_diff12,
        val_slides,
        val_labels,
        val_diff12,
    ) = split

    for fold in range(args.cv_fold):
        if len(val_slides[fold]) == 0:
            # Kept for compatibility with your previous val_ratio=0 setup.
            # For unbiased model selection, set --val_ratio > 0.
            val_slides[fold] = test_slides[fold]
            val_labels[fold] = test_labels[fold]
            val_diff12[fold] = test_diff12[fold]

    if not args.no_log:
        print(args)
        if args.val_ratio == 0:
            print("[Warning] val_ratio=0: test fold is reused as validation for checkpoint selection.")
        print("[Main idea] Metrics are computed from regression scores only: pred_Basal/pred_LumA/pred_LumB.")

    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed, but --wandb was provided.")
        wandb.init(project=args.project, name=args.title, config=vars(args), dir=args.model_path)

    fold_rows = []
    for fold in range(args.fold_start, args.cv_fold):
        row = run_fold(
            args=args,
            fold=fold,
            train_slides=train_slides[fold],
            train_labels=train_labels[fold],
            train_diff12=train_diff12[fold],
            train_strat_labels=train_strat[fold],
            val_slides=val_slides[fold],
            val_labels=val_labels[fold],
            val_diff12=val_diff12[fold],
            test_slides=test_slides[fold],
            test_labels=test_labels[fold],
            test_diff12=test_diff12[fold],
        )
        fold_rows.append(row)
        pd.DataFrame(fold_rows).to_csv(os.path.join(args.model_path, "TEST_RESULT_PATIENT_BASED_FINAL.csv"),
                                       index=False)

    df = pd.DataFrame(fold_rows)
    metric_cols = [c for c in df.columns if c.startswith("test_") and pd.api.types.is_numeric_dtype(df[c])]
    summary_rows = []
    for c in metric_cols:
        summary_rows.append({"metric": c, "mean": float(df[c].mean()), "std": float(df[c].std(ddof=0))})
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.model_path, "TEST_RESULT_SUMMARY_MEAN_STD.csv"), index=False)

    if args.wandb and wandb is not None:
        wandb.finish()

    if not args.no_log:
        print("\n========== CV summary ==========")
        for r in summary_rows:
            if r["metric"] in {"test_pearson_macro_overall", "test_auroc_ovr_macro", "test_auroc_ovr_Basal",
                               "test_auroc_ovo_macro"}:
                print(f"{r['metric']}: {r['mean']:.4f} ± {r['std']:.4f}")
        print(f"Saved to: {args.model_path}")


# ════════════════════════════════════════════════════════════════════════════
# Arguments
# ════════════════════════════════════════════════════════════════════════════


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PAM50 MambaMIL soft-label regression, reproducible optimized version")

    # Data / CV
    parser.add_argument("--datasets", default="tcga_pam50", type=str)
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--label_csv", default="", type=str)
    parser.add_argument("--tcga_max_patch", default=-1, type=int)
    parser.add_argument("--val_ratio", default=0.0, type=float)
    parser.add_argument("--fold_start", default=0, type=int)
    parser.add_argument("--cv_fold", default=5, type=int)
    parser.add_argument("--persistence", action="store_true")

    # Loss
    parser.add_argument("--reg_loss", default="mse", choices=["mse", "huber"])
    parser.add_argument("--huber_beta", default=0.10, type=float)
    parser.add_argument("--alpha_Basal", default=3.0, type=float)
    parser.add_argument("--alpha_LumA", default=1.0, type=float)
    parser.add_argument("--alpha_LumB", default=1.8, type=float)
    parser.add_argument("--use_conf_weight", action="store_true")
    parser.add_argument("--conf_weight_min", default=0.35, type=float)

    # Direct score-ranking regularization
    parser.add_argument("--lambda_soft_rank", default=0.05, type=float)
    parser.add_argument("--rank_warmup_epochs", default=5, type=int)
    parser.add_argument("--rank_ramp_epochs", default=10, type=int)
    parser.add_argument("--soft_rank_memory_size", default=512, type=int)
    parser.add_argument("--soft_rank_target_margin", default=0.05, type=float)
    parser.add_argument("--soft_rank_score_margin", default=0.02, type=float)
    parser.add_argument("--soft_rank_temperature", default=10.0, type=float)
    parser.add_argument("--soft_rank_weight_Basal", default=2.0, type=float)
    parser.add_argument("--soft_rank_weight_LumA", default=1.0, type=float)
    parser.add_argument("--soft_rank_weight_LumB", default=1.5, type=float)

    # Optional correlation regularization; default off because MSE already optimizes calibration.
    parser.add_argument("--lambda_corr", default=0.0, type=float)
    parser.add_argument("--corr_warmup_epochs", default=10, type=int)
    parser.add_argument("--corr_ramp_epochs", default=10, type=int)
    parser.add_argument("--corr_memory_size", default=512, type=int)
    parser.add_argument("--corr_min_samples", default=32, type=int)
    parser.add_argument("--corr_weight_Basal", default=1.2, type=float)
    parser.add_argument("--corr_weight_LumA", default=1.0, type=float)
    parser.add_argument("--corr_weight_LumB", default=1.1, type=float)

    # Sampling / reproducibility
    parser.add_argument("--use_weighted_sampler", dest="use_weighted_sampler", action="store_true", default=True)
    parser.add_argument("--no_weighted_sampler", dest="use_weighted_sampler", action="store_false")
    parser.add_argument("--sampler_Basal", default=4.0, type=float)
    parser.add_argument("--sampler_LumA", default=1.0, type=float)
    parser.add_argument("--sampler_LumB", default=2.0, type=float)
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--non_deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--deterministic_warn_only", dest="deterministic_warn_only", action="store_true", default=True)
    parser.add_argument("--deterministic_strict", dest="deterministic_warn_only", action="store_false")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    # Model / token selection
    parser.add_argument("--model", default="PAM_MIL", choices=["PAM_MIL"])
    parser.add_argument("--num_feats_ab", default=1536, type=int)
    parser.add_argument("--dropout", default=0.25, type=float)
    parser.add_argument("--act", default="gelu", choices=["relu", "gelu"])
    parser.add_argument("--mambamil_rate", default=10, type=int)
    parser.add_argument("--mambamil_layer", default=2, type=int)
    parser.add_argument("--mambamil_type", default="PAM_MIL",
                        choices=["Mamba", "BiMamba", "SRMamba", "PAM_MIL"])
    parser.add_argument("--model_nmax", default=4096, type=int)
    parser.add_argument("--coverage_bins", default=32, type=int)
    parser.add_argument("--coverage_per_bin", default=8, type=int)
    parser.add_argument("--saliency_ratio", default=0.25, type=float)
    parser.add_argument("--train_sampling_mode", default="deterministic", choices=["random", "deterministic", "off"])
    parser.add_argument("--train_views", default=4, type=int)
    parser.add_argument("--eval_views", default=1, type=int)

    # Optimization
    parser.add_argument("--num_epoch", default=80, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--lr_sche", default="cosine", choices=["cosine", "step", "const"])
    parser.add_argument("--weight_decay", default=1e-5, type=float)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--clip_grad", default=1.0, type=float)
    parser.add_argument("--amp", action="store_true")

    # Checkpoint monitor / early stop
    parser.add_argument("--monitor_metric", default="auc_pearson_composite",
                        choices=["auroc_ovr_macro", "pearson_macro_overall", "auc_pearson_composite"])
    parser.add_argument("--monitor_pearson_weight", default=0.15, type=float)
    parser.add_argument("--monitor_basal_weight", default=0.05, type=float)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--stop_epoch", default=35, type=int)

    # Logging / output
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--project", default="pam50_mil", type=str)
    parser.add_argument("--title", default="PAM_MIL", type=str)
    parser.add_argument("--log_iter", default=100, type=int)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument("--wandb", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())