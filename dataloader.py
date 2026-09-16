import os
import csv
import torch
import torch.nn as nn
import random
import numpy as np
from collections import Counter
from torch.utils.data import Dataset
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
import pandas as pd
import pickle
import torch.nn.functional as F
#----------------------------------------------------------------------

"""
kfold_soft.py
=============
PAM50 软标签回归任务 — 数据读取与 K 折划分

函数列表:
    get_patient_label(csv_file)   读取标签 CSV，返回五元组（含 diff12）
    get_kfold_soft(...)           Stratified Group K-Fold + 内部 val 切割
    _stratified_val_split(...)    内部辅助：分层 val 切割
    _print_fold_stats(...)        内部辅助：折内统计打印

改动（本版本新增）:
    get_kfold_soft : 新增 train_strat_list 返回（第4项）
                     供 one_fold 构建 WeightedRandomSampler 使用
                     返回值从 9 项扩展为 10 项

返回值顺序:
    train_slides, train_labels, train_diff12, train_strat,  ← train 组（4项）
    test_slides,  test_labels,  test_diff12,               ← test  组（3项）
    val_slides,   val_labels,   val_diff12                 ← val   组（3项）

依赖: numpy, pandas, sklearn, collections
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


# ════════════════════════════════════════════════════════════════════════════
# 1. 标签读取
# ════════════════════════════════════════════════════════════════════════════

def get_patient_label(csv_file):
    """
    读取 PAM50 软标签 CSV，返回软标签回归任务所需的五个数组。

    CSV 必需列
    ----------
    slide_id      : 切片 ID，与 WSI 特征文件名（.pt）对应
    patient_id    : 患者 ID（前 12 位 TCGA barcode），用于防患者泄露
    final_subtype : 硬标签（LumA / LumB / Basal），仅用于分层，不作为训练目标
    label_Basal   : 回归目标 ∈ [0, 1]
    label_LumA    : 回归目标 ∈ [0, 1]
    label_LumB    : 回归目标 ∈ [0, 1]

    CSV 可选列
    ----------
    diff12        : float，PAM50 分类器 top1-top2 置信度分差 ∈ [0, 1]
                    缺失时自动填充 -1.0（哨兵值）
                    PAM50WeightedMSELoss 遇到 diff12 < 0 时跳过置信度降权

    Returns
    -------
    slides       : ndarray [N], dtype=object    切片 ID
    soft_labels  : ndarray [N, 3], dtype=float32  列序 = [Basal, LumA, LumB]
    strat_labels : ndarray [N], dtype=object    final_subtype，供分层 + 过采样使用
    groups       : ndarray [N], dtype=object    patient_id，供分组使用
    diff12       : ndarray [N], dtype=float32   置信度分差，供 criterion 降权
    """
    df = pd.read_csv(csv_file)

    required_cols = {'slide_id', 'patient_id', 'final_subtype',
                     'label_Basal', 'label_LumA', 'label_LumB'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}")

    slides       = df['slide_id'].to_numpy(dtype=object)
    soft_labels  = df[['label_Basal', 'label_LumA', 'label_LumB']].to_numpy(dtype=np.float32)
    strat_labels = df['final_subtype'].to_numpy(dtype=object)
    groups       = df['patient_id'].to_numpy(dtype=object)

    # diff12：可选列，缺失时填 -1.0（哨兵值，criterion 遇负值跳过降权）
    if 'diff12' in df.columns:
        diff12 = df['diff12'].to_numpy(dtype=np.float32)
        print(f"[Label] diff12 loaded  -> "
              f"mean={diff12.mean():.3f}  "
              f"min={diff12.min():.3f}  "
              f"max={diff12.max():.3f}")
    else:
        diff12 = np.full(len(slides), -1.0, dtype=np.float32)
        print("[Label] diff12 not found in CSV, filled with -1.0 "
              "(confidence weighting disabled)")

    # ── 统计打印 ─────────────────────────────────────────────────────────────
    subtype_counts = dict(Counter(strat_labels))
    print(f"[Label] slides={len(slides)} | unique_patients={len(set(groups))}")
    print(f"[Label] subtype distribution: {subtype_counts}")
    print(f"[Label] soft_label mean  -> "
          f"Basal={soft_labels[:, 0].mean():.3f}  "
          f"LumA={soft_labels[:, 1].mean():.3f}  "
          f"LumB={soft_labels[:, 2].mean():.3f}")
    print(f"[Label] soft_label range -> "
          f"[{soft_labels.min():.3f}, {soft_labels.max():.3f}]")

    return slides, soft_labels, strat_labels, groups, diff12


# ════════════════════════════════════════════════════════════════════════════
# 2. K 折划分
# ════════════════════════════════════════════════════════════════════════════

def get_kfold_soft(
    k,
    slides_array,
    soft_labels_array,
    strat_labels,
    diff12_array=None,
    groups=None,
    val_ratio=0.15,
    shuffle=True,
    seed=42,
):
    """
    面向软标签回归的 Stratified (Group) K-Fold 数据划分。

    Parameters
    ----------
    k                 : int, 折数，必须 > 1
    slides_array      : array-like [N], 切片 ID
    soft_labels_array : array-like [N, 3], float，列序 = [Basal, LumA, LumB]
    strat_labels      : array-like [N], 分层用硬标签（str 或 int 均可）
    diff12_array      : array-like [N] 或 None
                        置信度分差；为 None 时各折 diff12 填充全 -1.0 数组
    groups            : array-like [N] 或 None
                        患者 ID；为 None 时退化为无组约束的 StratifiedKFold
    val_ratio         : float ∈ [0, 1)，从每折 train 内再切出 val 的比例
    shuffle           : bool
    seed              : int

    Returns（10 项，解包示例见文件末尾）
    -------
    train 组（4项）
    ├── train_slides_list : list[k], ndarray [n_train]        切片 ID
    ├── train_labels_list : list[k], ndarray [n_train, 3]     软标签
    ├── train_diff12_list : list[k], ndarray [n_train]        置信度分差
    └── train_strat_list  : list[k], ndarray [n_train] str    ← 新增
                            final_subtype 字符串，供 WeightedRandomSampler 使用

    test 组（3项）
    ├── test_slides_list  : list[k], ndarray [n_test]
    ├── test_labels_list  : list[k], ndarray [n_test, 3]
    └── test_diff12_list  : list[k], ndarray [n_test]

    val 组（3项）
    ├── val_slides_list   : list[k], ndarray [n_val]  (val_ratio=0 时为空)
    ├── val_labels_list   : list[k], ndarray [n_val, 3]
    └── val_diff12_list   : list[k], ndarray [n_val]
    """
    # ── 参数校验 ─────────────────────────────────────────────────────────────
    if k <= 1:
        raise ValueError("k must be > 1")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")

    slides_array      = np.asarray(slides_array, dtype=object)
    soft_labels_array = np.asarray(soft_labels_array, dtype=np.float32)
    strat_labels      = np.asarray(strat_labels)
    N                 = len(slides_array)

    if soft_labels_array.ndim != 2 or soft_labels_array.shape[1] != 3:
        raise ValueError(
            f"soft_labels_array must be shape [N, 3], got {soft_labels_array.shape}"
        )
    if len(soft_labels_array) != N:
        raise ValueError(
            f"slides_array ({N}) and soft_labels_array "
            f"({len(soft_labels_array)}) length mismatch"
        )

    # diff12：缺失时填 -1.0，随折索引同步切分
    if diff12_array is not None:
        diff12_array = np.asarray(diff12_array, dtype=np.float32)
        if len(diff12_array) != N:
            raise ValueError(
                f"diff12_array ({len(diff12_array)}) length mismatch with "
                f"slides_array ({N})"
            )
    else:
        diff12_array = np.full(N, -1.0, dtype=np.float32)

    # ── 选择 KFold 策略 ──────────────────────────────────────────────────────
    if groups is not None:
        groups = np.asarray(groups, dtype=object)
        if len(groups) != N:
            raise ValueError("groups length must match slides_array length")
        splitter   = StratifiedGroupKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(slides_array, strat_labels, groups)
    else:
        splitter   = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        split_iter = splitter.split(slides_array, strat_labels)

    # train 组 4 列表
    train_slides_list = []
    train_labels_list = []
    train_diff12_list = []
    train_strat_list  = []   # ← 新增：final_subtype 字符串，供 WeightedRandomSampler

    # test 组 3 列表
    test_slides_list  = []
    test_labels_list  = []
    test_diff12_list  = []

    # val 组 3 列表
    val_slides_list   = []
    val_labels_list   = []
    val_diff12_list   = []

    for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx  = np.asarray(test_idx,  dtype=np.int64)

        x_test = slides_array[test_idx]
        y_test = soft_labels_array[test_idx]
        d_test = diff12_array[test_idx]

        # ── 可选：从 train 内分层切 val ──────────────────────────────────────
        if val_ratio > 0.0:
            val_idx, train_idx = _stratified_val_split(
                train_idx,
                strat_labels=strat_labels,
                val_ratio=val_ratio,
                seed=seed + fold_id,
            )
            x_val  = slides_array[val_idx]
            y_val  = soft_labels_array[val_idx]
            d_val  = diff12_array[val_idx]
        else:
            # x_val = np.array([], dtype=object)
            # y_val = np.empty((0, 3), dtype=np.float32)
            # d_val = np.array([], dtype=np.float32)
            x_val = x_test.copy()
            y_val = y_test.copy()
            d_val = d_test.copy()

        # ── train / test 切分 ────────────────────────────────────────────────
        x_train = slides_array[train_idx]
        y_train = soft_labels_array[train_idx]
        d_train = diff12_array[train_idx]
        s_train = strat_labels[train_idx]    # ← 新增：final_subtype 字符串 [n_train]

        # ── 追加 ─────────────────────────────────────────────────────────────
        train_slides_list.append(x_train)
        train_labels_list.append(y_train)
        train_diff12_list.append(d_train)
        train_strat_list.append(s_train)     # ← 新增

        test_slides_list.append(x_test)
        test_labels_list.append(y_test)
        test_diff12_list.append(d_test)

        val_slides_list.append(x_val)
        val_labels_list.append(y_val)
        val_diff12_list.append(d_val)

        # ── 折内统计 ─────────────────────────────────────────────────────────
        _print_fold_stats(fold_id, strat_labels, train_idx, test_idx,
                          val_idx if val_ratio > 0.0 else np.array([], dtype=np.int64))

    return (
        # train 组（4项）
        train_slides_list, train_labels_list, train_diff12_list, train_strat_list,
        # test 组（3项）
        test_slides_list,  test_labels_list,  test_diff12_list,
        # val 组（3项）
        val_slides_list,   val_labels_list,   val_diff12_list,
    )


# ════════════════════════════════════════════════════════════════════════════
# 3. 内部辅助
# ════════════════════════════════════════════════════════════════════════════

def _stratified_val_split(train_idx, strat_labels, val_ratio, seed):
    """
    从 train_idx 内做分层 val 切割，返回全局索引。
    使用 StratifiedKFold 取第一折 test 侧作为 val，对稀有类（Basal n≈26）更鲁棒。
    """
    local_strat = strat_labels[train_idx]
    n_splits    = max(2, round(1.0 / val_ratio))
    skf         = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    remaining_local, val_local = next(iter(skf.split(train_idx, local_strat)))
    return train_idx[val_local], train_idx[remaining_local]


def _print_fold_stats(fold_id, strat_labels, train_idx, test_idx, val_idx):
    """打印每折的样本量与亚型分布，便于快速 QC。"""
    def _dist(idx):
        if len(idx) == 0:
            return {}
        return dict(Counter(strat_labels[idx]))

    print(
        f"[Fold {fold_id}] "
        f"train={len(train_idx)} {_dist(train_idx)} | "
        f"val={len(val_idx)} {_dist(val_idx)} | "
        f"test={len(test_idx)} {_dist(test_idx)}"
    )


#----------------------------------------------------------------------


#
# def get_kfold_soft(
#         k,
#         slides_array,
#         soft_labels_array,
#         strat_labels,
#         diff12_array=None,
#         groups=None,
#         val_ratio=0.15,
#         shuffle=True,
#         seed=42,
# ):
#     """
#     面向软标签回归的 Stratified (Group) K-Fold 数据划分。
#
#     Parameters
#     ----------
#     k                 : int, 折数，必须 > 1
#     slides_array      : array-like [N], 切片 ID
#     soft_labels_array : array-like [N, 3], float，列序 = [Basal, LumA, LumB]
#     strat_labels      : array-like [N], 分层用硬标签（str 或 int 均可）
#     diff12_array      : array-like [N] 或 None
#                         置信度分差；为 None 时各折 diff12 列表填充全 -1.0 数组
#     groups            : array-like [N] 或 None
#                         患者 ID；为 None 时退化为无组约束的 StratifiedKFold
#     val_ratio         : float ∈ [0, 1)，从每折 train 内再切出 val 的比例
#     shuffle           : bool
#     seed              : int
#
#     Returns
#     -------
#     train_slides_list  : list[k], ndarray [n_train]
#     train_labels_list  : list[k], ndarray [n_train, 3]
#     train_diff12_list  : list[k], ndarray [n_train]
#     test_slides_list   : list[k], ndarray [n_test]
#     test_labels_list   : list[k], ndarray [n_test, 3]
#     test_diff12_list   : list[k], ndarray [n_test]
#     val_slides_list    : list[k], ndarray [n_val]  (val_ratio=0 时为空)
#     val_labels_list    : list[k], ndarray [n_val, 3]
#     val_diff12_list    : list[k], ndarray [n_val]
#     """
#     # ── 参数校验 ─────────────────────────────────────────────────────────────
#     if k <= 1:
#         raise ValueError("k must be > 1")
#     if not (0.0 <= val_ratio < 1.0):
#         raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
#
#     slides_array = np.asarray(slides_array, dtype=object)
#     soft_labels_array = np.asarray(soft_labels_array, dtype=np.float32)
#     strat_labels = np.asarray(strat_labels)
#     N = len(slides_array)
#
#     if soft_labels_array.ndim != 2 or soft_labels_array.shape[1] != 3:
#         raise ValueError(
#             f"soft_labels_array must be shape [N, 3], got {soft_labels_array.shape}"
#         )
#     if len(soft_labels_array) != N:
#         raise ValueError(
#             f"slides_array ({N}) and soft_labels_array "
#             f"({len(soft_labels_array)}) length mismatch"
#         )
#
#     # diff12：缺失时填 -1.0（哨兵值），随折索引同步切分
#     if diff12_array is not None:
#         diff12_array = np.asarray(diff12_array, dtype=np.float32)
#         if len(diff12_array) != N:
#             raise ValueError(
#                 f"diff12_array ({len(diff12_array)}) length mismatch with "
#                 f"slides_array ({N})"
#             )
#     else:
#         diff12_array = np.full(N, -1.0, dtype=np.float32)
#
#     # ── 选择 KFold 策略 ──────────────────────────────────────────────────────
#     if groups is not None:
#         groups = np.asarray(groups, dtype=object)
#         if len(groups) != N:
#             raise ValueError("groups length must match slides_array length")
#         splitter = StratifiedGroupKFold(n_splits=k, shuffle=shuffle, random_state=seed)
#         split_iter = splitter.split(slides_array, strat_labels, groups)
#     else:
#         splitter = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
#         split_iter = splitter.split(slides_array, strat_labels)
#
#     train_slides_list, train_labels_list, train_diff12_list = [], [], []
#     test_slides_list, test_labels_list, test_diff12_list = [], [], []
#     val_slides_list, val_labels_list, val_diff12_list = [], [], []
#
#     # for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
#     #     train_idx = np.asarray(train_idx, dtype=np.int64)
#     #     test_idx = np.asarray(test_idx, dtype=np.int64)
#     #
#     #     # ── 可选：从 train 内分层切 val ──────────────────────────────────────
#     #     if val_ratio > 0.0:
#     #         val_idx, train_idx = _stratified_val_split(
#     #             train_idx,
#     #             strat_labels=strat_labels,
#     #             val_ratio=val_ratio,
#     #             seed=seed + fold_id,
#     #         )
#     #         x_val = slides_array[val_idx]
#     #         y_val = soft_labels_array[val_idx]
#     #         d_val = diff12_array[val_idx]
#     #     else:
#     #         x_val = np.array([], dtype=object)
#     #         y_val = np.empty((0, 3), dtype=np.float32)
#     #         d_val = np.array([], dtype=np.float32)
#     #
#     #     x_train = slides_array[train_idx]
#     #     y_train = soft_labels_array[train_idx]
#     #     d_train = diff12_array[train_idx]
#     #
#     #     x_test = slides_array[test_idx]
#     #     y_test = soft_labels_array[test_idx]
#     #     d_test = diff12_array[test_idx]
#     #
#     #     train_slides_list.append(x_train)
#     #     train_labels_list.append(y_train)
#     #     train_diff12_list.append(d_train)
#     #
#     #     test_slides_list.append(x_test)
#     #     test_labels_list.append(y_test)
#     #     test_diff12_list.append(d_test)
#     #
#     #     val_slides_list.append(x_val)
#     #     val_labels_list.append(y_val)
#     #     val_diff12_list.append(d_val)
#
#     for fold_id, (train_idx, test_idx) in enumerate(split_iter, start=1):
#         train_idx = np.asarray(train_idx, dtype=np.int64)
#         test_idx = np.asarray(test_idx, dtype=np.int64)
#
#         # 先取 test
#         x_test = slides_array[test_idx]
#         y_test = soft_labels_array[test_idx]
#         d_test = diff12_array[test_idx]
#
#         # 再决定 train / val
#         if val_ratio > 0.0:
#             val_idx, train_idx = _stratified_val_split(
#                 train_idx,
#                 strat_labels=strat_labels,
#                 val_ratio=val_ratio,
#                 seed=seed + fold_id,
#             )
#             val_idx = np.asarray(val_idx, dtype=np.int64)
#
#             x_val = slides_array[val_idx]
#             y_val = soft_labels_array[val_idx]
#             d_val = diff12_array[val_idx]
#         else:
#             x_val = x_test.copy()
#             y_val = y_test.copy()
#             d_val = d_test.copy()
#
#         x_train = slides_array[train_idx]
#         y_train = soft_labels_array[train_idx]
#         d_train = diff12_array[train_idx]
#
#         train_slides_list.append(x_train)
#         train_labels_list.append(y_train)
#         train_diff12_list.append(d_train)
#
#         test_slides_list.append(x_test)
#         test_labels_list.append(y_test)
#         test_diff12_list.append(d_test)
#
#         val_slides_list.append(x_val)
#         val_labels_list.append(y_val)
#         val_diff12_list.append(d_val)
#
#         # ── 折内统计 ─────────────────────────────────────────────────────────
#         _print_fold_stats(fold_id, strat_labels, train_idx, test_idx,
#                           val_idx if val_ratio > 0.0 else np.array([], dtype=np.int64))
#
#     return (
#         train_slides_list, train_labels_list, train_diff12_list,
#         test_slides_list, test_labels_list, test_diff12_list,
#         val_slides_list, val_labels_list, val_diff12_list,
#     )


def _stratified_val_split(train_idx, strat_labels, val_ratio, seed):
    """
    从 train_idx 内做分层 val 切割，返回全局索引。
    使用 StratifiedKFold 取第一折 test 侧作为 val，对稀有类（Basal n≈26）更鲁棒。
    """
    local_strat = strat_labels[train_idx]
    n_splits = max(2, round(1.0 / val_ratio))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    remaining_local, val_local = next(iter(skf.split(train_idx, local_strat)))
    return train_idx[val_local], train_idx[remaining_local]


def _print_fold_stats(fold_id, strat_labels, train_idx, test_idx, val_idx):
    """打印每折的样本量与亚型分布，便于快速 QC。"""

    def _dist(idx):
        if len(idx) == 0:
            return {}
        return dict(Counter(strat_labels[idx]))

    print(
        f"[Fold {fold_id}] "
        f"train={len(train_idx)} {_dist(train_idx)} | "
        f"val={len(val_idx)} {_dist(val_idx)} | "
        f"test={len(test_idx)} {_dist(test_idx)}"
    )
#
# def _stratified_val_split(train_idx, strat_labels, val_ratio, seed):
#     """
#     从 train_idx 内做分层 val 切割，返回全局索引。
#
#     实现方式
#     --------
#     使用 StratifiedKFold(n_splits = round(1 / val_ratio)) 并取第一折的
#     test 部分作为 val。相比 train_test_split，此方式对稀有类（Basal n≈26）
#     更鲁棒——即使单折 train 内 Basal 只有约 20 例也不会报错。
#
#     Parameters
#     ----------
#     train_idx    : ndarray [n_train], 全局索引
#     strat_labels : ndarray [N], 全量分层标签（取 train_idx 子集使用）
#     val_ratio    : float, val 占 train+val 的比例
#     seed         : int
#
#     Returns
#     -------
#     val_idx           : ndarray, 全局索引
#     remaining_train_idx: ndarray, 全局索引
#     """
#     local_strat = strat_labels[train_idx]
#     n_splits = max(2, round(1.0 / val_ratio))  # val_ratio=0.15 → n_splits=7
#     skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
#
#     # 取第一折的 test 侧作为 val
#     remaining_local, val_local = next(iter(skf.split(train_idx, local_strat)))
#
#     return train_idx[val_local], train_idx[remaining_local]
#
#
# def _print_fold_stats(fold_id, strat_labels, train_idx, test_idx, val_idx):
#     """打印每折的样本量与亚型分布，便于快速 QC。"""
#
#     def _dist(idx):
#         if len(idx) == 0:
#             return {}
#         return dict(Counter(strat_labels[idx]))
#
#     print(
#         f"[Fold {fold_id}] "
#         f"train={len(train_idx)} {_dist(train_idx)} | "
#         f"val={len(val_idx)} {_dist(val_idx)} | "
#         f"test={len(test_idx)} {_dist(test_idx)}"
#     )
# def readCSV(filename):
#     lines = []
#     # with open(filename, "r") as f:
#     with open(filename, "r", encoding='utf-8-sig') as f:
#         csvreader = csv.reader(f)
#         for line in csvreader:
#             lines.append(line)
#     return lines

# def get_patient_label(csv_file):
#     patients_list=[]
#     labels_list=[]
#     label_file = readCSV(csv_file)
#     for i in range(0, len(label_file)):
#         patients_list.append(label_file[i][0])
#         labels_list.append(label_file[i][1])
#     a=Counter(labels_list)
#     print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
#     print("all_counter:{}".format(dict(a)))
#     return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

#
# def get_patient_label(csv_file):
#     """
#     读取 PAM50 软标签 CSV，返回软标签回归任务所需的四个数组。
#
#     CSV 必需列:
#         slide_id       : 切片 ID（与 WSI 特征文件名对应）
#         patient_id     : 患者 ID（用于 StratifiedGroupKFold 防泄露）
#         final_subtype  : 硬标签（LumA / LumB / Basal），仅用于分层
#         label_Basal    : 回归目标 ∈ [0, 1]
#         label_LumA     : 回归目标 ∈ [0, 1]
#         label_LumB     : 回归目标 ∈ [0, 1]
#
#     Returns:
#         slides       : ndarray [N], dtype=object,  切片 ID
#         soft_labels  : ndarray [N, 3], dtype=float32, [Basal, LumA, LumB]
#         strat_labels : ndarray [N], dtype=object,  final_subtype（分层用）
#         groups       : ndarray [N], dtype=object,  patient_id（分组用）
#     """
#     df = pd.read_csv(csv_file)
#
#     required_cols = {'slide_id', 'patient_id', 'final_subtype',
#                      'label_Basal', 'label_LumA', 'label_LumB'}
#     missing = required_cols - set(df.columns)
#     if missing:
#         raise ValueError(f"CSV 缺少必需列: {missing}")
#
#     slides       = df['slide_id'].to_numpy(dtype=object)
#     soft_labels  = df[['label_Basal', 'label_LumA', 'label_LumB']].to_numpy(dtype=np.float32)
#     strat_labels = df['final_subtype'].to_numpy(dtype=object)
#     groups       = df['patient_id'].to_numpy(dtype=object)
#
#     # ── 统计信息 ─────────────────────────────────────────────────────────────
#     subtype_counts = Counter(strat_labels)
#     print(f"slides: {len(slides)} | patients: {len(set(groups))}")
#     print(f"subtype distribution: {dict(subtype_counts)}")
#     print(f"soft_labels mean  -> Basal={soft_labels[:,0].mean():.3f}  "
#           f"LumA={soft_labels[:,1].mean():.3f}  LumB={soft_labels[:,2].mean():.3f}")
#     print(f"soft_labels range -> [{soft_labels.min():.3f}, {soft_labels.max():.3f}]")
#
#     return slides, soft_labels, strat_labels, groups

def get_patient_label(csv_file):
    """
    读取 PAM50 软标签 CSV，返回软标签回归任务所需的五个数组。

    CSV 必需列
    ----------
    slide_id      : 切片 ID，与 WSI 特征文件名（.pt）对应
    patient_id    : 患者 ID（前 12 位 TCGA barcode），用于防患者泄露
    final_subtype : 硬标签（LumA / LumB / Basal），仅用于分层，不作为训练目标
    label_Basal   : 回归目标 ∈ [0, 1]
    label_LumA    : 回归目标 ∈ [0, 1]
    label_LumB    : 回归目标 ∈ [0, 1]

    CSV 可选列
    ----------
    diff12        : float，PAM50 分类器 top1-top2 置信度分差 ∈ [0, 1]
                    缺失时自动填充 -1.0（哨兵值）
                    PAM50WeightedMSELoss 遇到 diff12 < 0 时跳过置信度降权

    Returns
    -------
    slides       : ndarray [N], dtype=object    切片 ID
    soft_labels  : ndarray [N, 3], dtype=float32  列序 = [Basal, LumA, LumB]
    strat_labels : ndarray [N], dtype=object    final_subtype，供分层使用
    groups       : ndarray [N], dtype=object    patient_id，供分组使用
    diff12       : ndarray [N], dtype=float32   置信度分差，供 criterion 降权
    """
    df = pd.read_csv(csv_file)

    required_cols = {'slide_id', 'patient_id', 'final_subtype',
                     'label_Basal', 'label_LumA', 'label_LumB'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必需列: {missing}")

    slides = df['slide_id'].to_numpy(dtype=object)
    soft_labels = df[['label_Basal', 'label_LumA', 'label_LumB']].to_numpy(dtype=np.float32)
    strat_labels = df['final_subtype'].to_numpy(dtype=object)
    groups = df['patient_id'].to_numpy(dtype=object)

    # diff12：可选列，缺失时填 -1.0（哨兵值，criterion 遇负值跳过降权）
    if 'diff12' in df.columns:
        diff12 = df['diff12'].to_numpy(dtype=np.float32)
        print(f"[Label] diff12 loaded  -> "
              f"mean={diff12.mean():.3f}  "
              f"min={diff12.min():.3f}  "
              f"max={diff12.max():.3f}")
    else:
        diff12 = np.full(len(slides), -1.0, dtype=np.float32)
        print("[Label] diff12 not found in CSV, filled with -1.0 "
              "(confidence weighting disabled)")

    # ── 统计打印 ─────────────────────────────────────────────────────────────
    subtype_counts = dict(Counter(strat_labels))
    print(f"[Label] slides={len(slides)} | unique_patients={len(set(groups))}")
    print(f"[Label] subtype distribution: {subtype_counts}")
    print(f"[Label] soft_label mean  -> "
          f"Basal={soft_labels[:, 0].mean():.3f}  "
          f"LumA={soft_labels[:, 1].mean():.3f}  "
          f"LumB={soft_labels[:, 2].mean():.3f}")
    print(f"[Label] soft_label range -> "
          f"[{soft_labels.min():.3f}, {soft_labels.max():.3f}]")

    return slides, soft_labels, strat_labels, groups, diff12






def _build_split_df(slides, labels, fold_id, split_name):
    """
    将单个 fold 的一个 split（train/val/test）展开成逐样本 DataFrame
    """
    slides = np.asarray(slides, dtype=object).reshape(-1)
    labels = np.asarray(labels, dtype=np.float32)

    base_cols = [
        'fold', 'split', 'slide_id',
        'label_Basal', 'label_LumA', 'label_LumB'
    ]

    if slides.size == 0:
        return pd.DataFrame(columns=base_cols)

    if labels.ndim != 2 or labels.shape[1] != 3:
        raise ValueError(
            f'labels must be shape [N, 3], got {labels.shape} in fold={fold_id}, split={split_name}'
        )
    if len(slides) != len(labels):
        raise ValueError(
            f'slides and labels length mismatch in fold={fold_id}, split={split_name}: '
            f'{len(slides)} vs {len(labels)}'
        )

    df = pd.DataFrame({
        'fold': [fold_id] * len(slides),
        'split': [split_name] * len(slides),
        'slide_id': slides.astype(str),
        'label_Basal': labels[:, 0],
        'label_LumA': labels[:, 1],
        'label_LumB': labels[:, 2],
    })

    df['sum3'] = df[['label_Basal', 'label_LumA', 'label_LumB']].sum(axis=1)

    soft_top1_idx = df[['label_Basal', 'label_LumA', 'label_LumB']].to_numpy().argmax(axis=1)
    idx2name = {0: 'Basal', 1: 'LumA', 2: 'LumB'}
    df['soft_top1_subtype'] = [idx2name[i] for i in soft_top1_idx]

    return df


def save_soft_split_history(
    model_path,
    train_p, train_l,
    val_p, val_l,
    test_p, test_l,
    label_df=None,
    prefix='split'
):
    """
    保存 soft-label k-fold 划分结果：
    1) {prefix}_manifest.csv  : 每个 slide 一行
    2) {prefix}_summary.csv   : 每折摘要统计
    3) {prefix}_payload.pkl   : 原始 list-of-ndarray 无损保存
    """

    os.makedirs(model_path, exist_ok=True)

    if not (
        len(train_p) == len(train_l) ==
        len(val_p) == len(val_l) ==
        len(test_p) == len(test_l)
    ):
        raise ValueError("train/val/test fold list lengths are inconsistent")

    n_folds = len(train_p)

    # ---------- 1) 展开为逐样本清单 ----------
    parts = []
    for fold_id in range(1, n_folds + 1):
        parts.append(_build_split_df(train_p[fold_id - 1], train_l[fold_id - 1], fold_id, 'train'))
        parts.append(_build_split_df(val_p[fold_id - 1],   val_l[fold_id - 1],   fold_id, 'val'))
        parts.append(_build_split_df(test_p[fold_id - 1],  test_l[fold_id - 1],  fold_id, 'test'))

    history_df = pd.concat(parts, ignore_index=True)

    # ---------- 2) 合并最终标签文件中的元信息 ----------
    if label_df is not None:
        label_df = label_df.copy()

        if 'slide_id' not in label_df.columns:
            raise ValueError("label_df must contain column 'slide_id'")

        meta_cols_preferred = [
            'slide_id', 'patient_id', 'final_subtype', 'diff12',
            'sample_weight', 'gene_id', 'status'
        ]
        meta_cols = [c for c in meta_cols_preferred if c in label_df.columns]

        meta_df = label_df[meta_cols].drop_duplicates(subset=['slide_id'])
        history_df = history_df.merge(
            meta_df,
            on='slide_id',
            how='left',
            validate='many_to_one'
        )

        # patient-level leakage 检查
        if 'patient_id' in history_df.columns:
            for fold_id in sorted(history_df['fold'].unique()):
                sub = history_df[history_df['fold'] == fold_id]

                def _get_patient_set(split_name):
                    x = sub[sub['split'] == split_name]['patient_id'].dropna().astype(str).unique()
                    return set(x.tolist())

                train_pat = _get_patient_set('train')
                val_pat   = _get_patient_set('val')
                test_pat  = _get_patient_set('test')

                if train_pat & val_pat:
                    raise RuntimeError(f'Patient leakage detected in fold {fold_id}: train ∩ val != empty')
                if train_pat & test_pat:
                    raise RuntimeError(f'Patient leakage detected in fold {fold_id}: train ∩ test != empty')
                # if val_pat & test_pat:
                #     raise RuntimeError(f'Patient leakage detected in fold {fold_id}: val ∩ test != empty')

    # 保存 manifest
    manifest_path = os.path.join(model_path, f'{prefix}_manifest.csv')
    history_df.to_csv(manifest_path, index=False)

    # ---------- 3) 生成摘要统计 ----------
    agg_dict = {
        'n_slides': ('slide_id', 'count'),
        'mean_label_Basal': ('label_Basal', 'mean'),
        'mean_label_LumA': ('label_LumA', 'mean'),
        'mean_label_LumB': ('label_LumB', 'mean'),
        'mean_sum3': ('sum3', 'mean'),
    }

    if 'patient_id' in history_df.columns:
        agg_dict['n_patients'] = ('patient_id', pd.Series.nunique)

    summary_df = (
        history_df
        .groupby(['fold', 'split'], as_index=False)
        .agg(**agg_dict)
    )

    # 如有 final_subtype，则补充硬标签计数
    if 'final_subtype' in history_df.columns:
        subtype_count_df = (
            history_df
            .groupby(['fold', 'split', 'final_subtype'])['slide_id']
            .count()
            .unstack(fill_value=0)
            .reset_index()
        )
        summary_df = summary_df.merge(
            subtype_count_df,
            on=['fold', 'split'],
            how='left'
        )

    summary_path = os.path.join(model_path, f'{prefix}_summary.csv')
    summary_df.to_csv(summary_path, index=False)

    # ---------- 4) 保存原始对象，便于精确重载 ----------
    payload = {
        'train_p': [np.asarray(x, dtype=object) for x in train_p],
        'train_l': [np.asarray(x, dtype=np.float32) for x in train_l],
        'val_p':   [np.asarray(x, dtype=object) for x in val_p],
        'val_l':   [np.asarray(x, dtype=np.float32) for x in val_l],
        'test_p':  [np.asarray(x, dtype=object) for x in test_p],
        'test_l':  [np.asarray(x, dtype=np.float32) for x in test_l],
    }

    payload_path = os.path.join(model_path, f'{prefix}_payload.pkl')
    with open(payload_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f'[Saved] manifest: {manifest_path}')
    print(f'[Saved] summary : {summary_path}')
    print(f'[Saved] payload : {payload_path}')
    # return history_df, summary_df


"""
loss.py
=======
PAM50 软标签回归损失函数

设计依据（TCGA-PRAD, N=369）
-----------------------------
类别分布：Basal 7.0%，LumA 77.5%，LumB 15.4%
软标签均值：Basal=0.058，LumA=0.432，LumB=0.096

类别权重推导（频率倒数 → 开方缩放 → 归一化至 LumA=1）：
  频率倒数：Basal=14.3，LumA=1.29，LumB=6.49
  开方：    Basal=3.78，LumA=1.14，LumB=2.55
  归一化：  α_Basal≈3.5，α_LumA=1.0，α_LumB≈2.2
  （开方缩放目的：避免 Basal 权重过大导致梯度爆炸）

损失公式：
  L = Σ_{c} α_c · w_sample · (pred_c - y_c)²

  其中 w_sample（可选）= diff12 置信度权重，对标签不确定样本降权
"""

class PAM50WeightedMSELoss(nn.Module):
    """
    PAM50 三类软标签加权 MSE 损失。

    特性
    ----
    1. 类别权重 alpha  : 对抗 LumA 77.5% 的类别不均衡
    2. 样本置信度权重  : 可选，传入 diff12 对标签不确定样本降权
    3. 支持 batch 输入 : pred/target shape 为 (3,) 或 (B, 3) 均可
    4. reduction 选项  : 'mean'（默认）/ 'sum' / 'none'

    Parameters
    ----------
    alpha_Basal : float, 默认 3.5
        Basal 类别权重（频率倒数开方归一化，详见模块 docstring）
    alpha_LumA  : float, 默认 1.0
        LumA 类别权重（基准）
    alpha_LumB  : float, 默认 2.2
        LumB 类别权重
    use_conf_weight : bool, 默认 True
        是否启用 diff12 置信度权重
    conf_weight_min : float, 默认 0.3
        diff12=0 时的最低权重（避免完全忽略 Ambiguous 样本）
    reduction : str, 默认 'mean'
        'mean' | 'sum' | 'none'
    """

    def __init__(
            self,
            alpha_Basal: float = 3.5,
            alpha_LumA: float = 1.0,
            alpha_LumB: float = 2.2,
            use_conf_weight: bool = True,
            conf_weight_min: float = 0.3,
            reduction: str = 'mean',
    ):
        super().__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got '{reduction}'")

        self.register_buffer(
            'alpha',
            torch.tensor([alpha_Basal, alpha_LumA, alpha_LumB], dtype=torch.float32),
        )
        self.use_conf_weight = use_conf_weight
        self.conf_weight_min = conf_weight_min
        self.reduction = reduction

    def forward(
            self,
            pred: torch.Tensor,  # (3,) 或 (B, 3)
            target: torch.Tensor,  # (3,) 或 (B, 3)
            diff12: torch.Tensor | None = None,  # (,) 或 (B,)，可选
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : Tensor (3,) 或 (B, 3)，三路 Sigmoid 输出
        target : Tensor (3,) 或 (B, 3)，原始 PAM50 软标签
        diff12 : Tensor (,) 或 (B,) 或 None
                 PAM50 分类器 top1-top2 分差，越大标签越可信
                 为 None 时不施加置信度权重

        Returns
        -------
        Tensor：标量（reduction='mean'|'sum'）或 (B,)（reduction='none'）
        """
        # ── 维度统一为 (B, 3) ────────────────────────────────────────────────
        squeeze_output = False
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)  # (1, 3)
            target = target.unsqueeze(0)
            if diff12 is not None and diff12.dim() == 0:
                diff12 = diff12.unsqueeze(0)
            squeeze_output = True

        B = pred.shape[0]

        # ── per-class 加权 MSE：(B, 3) ──────────────────────────────────────
        alpha = self.alpha.to(pred.device)  # (3,)
        mse_per_class = (pred - target) ** 2  # (B, 3)
        weighted = mse_per_class * alpha  # (B, 3)，广播

        # ── 样本级损失：(B,) ─────────────────────────────────────────────────
        loss_per_sample = weighted.sum(dim=1)  # (B,)

        # ── 置信度权重（可选） ────────────────────────────────────────────────
        if self.use_conf_weight and diff12 is not None:
            diff12 = diff12.to(pred.device).float()  # (B,)
            # 线性映射：diff12 ∈ [0,1] → w ∈ [conf_weight_min, 1.0]
            # diff12=0（完全 Ambiguous）→ w=conf_weight_min
            # diff12=1（极度确定）      → w=1.0
            conf_w = self.conf_weight_min + (1.0 - self.conf_weight_min) * diff12.clamp(0, 1)
            loss_per_sample = loss_per_sample * conf_w  # (B,)

        # ── reduction ────────────────────────────────────────────────────────
        if self.reduction == 'mean':
            loss = loss_per_sample.mean()
        elif self.reduction == 'sum':
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        if squeeze_output and self.reduction == 'none':
            loss = loss.squeeze(0)

        return loss

    def extra_repr(self) -> str:
        return (
            f"alpha=[{self.alpha[0]:.1f}, {self.alpha[1]:.1f}, {self.alpha[2]:.1f}]  "
            f"use_conf_weight={self.use_conf_weight}  "
            f"conf_weight_min={self.conf_weight_min}  "
            f"reduction='{self.reduction}'"
        )


#
# def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
#     """
#     dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
#     :param full_list:
#     :param ratio:
#     :param shuffle:
#     """
#     # select the val-set based on the label ratio
#     if label_balance_val and label is not None:
#         _label = label[full_list]
#         _label_uni = np.unique(_label)
#         sublist_1 = []
#         sublist_2 = []
#
#         for _l in _label_uni:
#             _list = full_list[_label == _l]
#             n_total = len(_list)
#             offset = int(n_total * ratio)
#             if shuffle:
#                 random.shuffle(_list)
#             sublist_1.extend(_list[:offset])
#             sublist_2.extend(_list[offset:])
#     else:
#         n_total = len(full_list)
#         offset = int(n_total * ratio)
#         if n_total == 0 or offset < 1:
#             return [], full_list
#         if shuffle:
#             random.shuffle(full_list)
#         val_set = full_list[:offset]
#         train_set = full_list[offset:]
#
#     return val_set, train_set

# def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
#     if k > 1:
#         skf = StratifiedKFold(n_splits=k)
#     else:
#         raise NotImplementedError
#     train_patients_list = []
#     train_labels_list = []
#     test_patients_list = []
#     test_labels_list = []
#     val_patients_list = []
#     val_labels_list = []
#     for train_index, test_index in skf.split(patients_array, labels_array):
#         if val_ratio != 0.:
#             # val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
#             val_index, train_index = data_split_strict(
#                 train_index,
#                 ratio=val_ratio,
#                 shuffle=True,
#                 label=labels_array,
#                 seed=42 + k
#             )
#             x_val, y_val = patients_array[val_index], labels_array[val_index]
#         else:
#             x_val, y_val = [],[]
#         x_train, x_test = patients_array[train_index], patients_array[test_index]
#         y_train, y_test = labels_array[train_index], labels_array[test_index]
#
#         train_patients_list.append(x_train)
#         train_labels_list.append(y_train)
#         test_patients_list.append(x_test)
#         test_labels_list.append(y_test)
#         val_patients_list.append(x_val)
#         val_labels_list.append(y_val)
#
#     # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
#     return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)
#


def data_split_strict(full_list, ratio, shuffle=True, label=None, seed=42):
    """
    Strict stratified split from full_list into (val_set, train_set).
    Guarantees (when possible):
      - For each class with n>=2: at least 1 sample in val and at least 1 in train
      - For class with n==1: keep it in train (val gets 0 from that class)
    No in-place modification on inputs.
    """
    full_list = np.array(full_list, dtype=np.int64)

    if ratio is None or ratio <= 0.0 or label is None:
        return np.array([], dtype=np.int64), full_list

    rng = np.random.default_rng(seed)

    labels = np.asarray(label)[full_list]
    classes = np.unique(labels)

    val_idx_parts = []
    train_idx_parts = []

    for c in classes:
        idx_c = full_list[labels == c]
        n = len(idx_c)
        if n == 0:
            continue

        if shuffle:
            idx_c = idx_c.copy()
            rng.shuffle(idx_c)

        n_val = int(round(n * ratio))

        if n >= 2:
            n_val = max(1, min(n_val, n - 1))
        else:
            n_val = 0  # only 1 sample -> keep it in train

        val_idx_parts.append(idx_c[:n_val])
        train_idx_parts.append(idx_c[n_val:])

    val_idx = np.concatenate(val_idx_parts) if len(val_idx_parts) else np.array([], dtype=np.int64)
    tr_idx  = np.concatenate(train_idx_parts) if len(train_idx_parts) else np.array([], dtype=np.int64)

    if shuffle:
        rng.shuffle(val_idx)
        rng.shuffle(tr_idx)

    return val_idx.astype(np.int64), tr_idx.astype(np.int64)


def get_kflod(
    k,
    patients_array,
    labels_array,
    val_ratio=0.0,
    shuffle=True,
    seed=42,
    label_balance_val=True,   # kept for backward compatibility; strict split is stratified by default
    return_class_counts=False,  # add per-fold train class counts (for sampler / LogitAdjustedCE)
):
    """
    Stratified K-Fold split with optional stratified validation split inside each train fold.

    Args:
      k: number of folds (k>1)
      patients_array: np array/list, shape [N], each is patient/slide id
      labels_array:   np array/list, shape [N], int labels (0..C-1)
      val_ratio: float in [0,1). If >0, split train_index into (val_index, train_index)
      shuffle: whether to shuffle folds
      seed: random seed
      label_balance_val: kept; if False, val split can be random (not recommended)
      return_class_counts: if True, also return per-fold class_counts on TRAIN split

    Returns (dtype=object arrays to match your original):
      train_patients_list: [k] of arrays
      train_labels_list  : [k] of arrays
      test_patients_list : [k] of arrays
      test_labels_list   : [k] of arrays
      val_patients_list  : [k] of arrays (empty arrays if val_ratio==0)
      val_labels_list    : [k] of arrays
      (optional) train_class_counts_list: [k] of arrays, np.bincount(y_train)
    """
    if k <= 1:
        raise NotImplementedError("k must be > 1")

    patients_array = np.asarray(patients_array, dtype=object)
    labels_array   = np.asarray(labels_array)

    # StratifiedKFold (recommended shuffle=True with fixed seed)
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)

    train_patients_list, train_labels_list = [], []
    test_patients_list,  test_labels_list  = [], []
    val_patients_list,   val_labels_list   = [], []
    train_class_counts_list = []

    for fold_id, (train_index, test_index) in enumerate(skf.split(patients_array, labels_array), start=1):
        train_index = np.asarray(train_index, dtype=np.int64)
        test_index  = np.asarray(test_index, dtype=np.int64)

        # ---- optional val split inside train
        if val_ratio and val_ratio > 0.0:
            if label_balance_val:
                val_index, train_index = data_split_strict(
                    train_index,
                    ratio=val_ratio,
                    shuffle=True,
                    label=labels_array,
                    seed=seed + fold_id
                )
            else:
                # random (NOT stratified) val split, kept for compatibility
                rng = np.random.default_rng(seed + fold_id)
                perm = train_index.copy()
                rng.shuffle(perm)
                n_val = int(round(len(perm) * val_ratio))
                n_val = max(1, min(n_val, len(perm) - 1))
                val_index = perm[:n_val]
                train_index = perm[n_val:]

            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val = np.array([], dtype=object)
            y_val = np.array([], dtype=labels_array.dtype)

        # ---- train/test
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)

        if return_class_counts:
            # ensure bincount length covers all classes
            n_classes = int(labels_array.max()) + 1 if labels_array.size > 0 else 0
            counts = np.bincount(y_train.astype(np.int64), minlength=n_classes)
            train_class_counts_list.append(counts)

    outputs = (
        np.array(train_patients_list, dtype=object),
        np.array(train_labels_list, dtype=object),
        np.array(test_patients_list, dtype=object),
        np.array(test_labels_list, dtype=object),
        np.array(val_patients_list, dtype=object),
        np.array(val_labels_list, dtype=object),
    )

    if return_class_counts:
        outputs = outputs + (np.array(train_class_counts_list, dtype=object),)

    return outputs


# ----------------------------
# Optional sanity check helper
# ----------------------------
def print_fold_class_counts(train_labels_list, val_labels_list, test_labels_list, n_classes=None):
    """
    Print class counts for each fold to ensure no class disappears.
    """
    k = len(train_labels_list)
    for i in range(k):
        y_tr = np.asarray(train_labels_list[i], dtype=np.int64)
        y_va = np.asarray(val_labels_list[i], dtype=np.int64) if len(val_labels_list[i]) else np.array([], dtype=np.int64)
        y_te = np.asarray(test_labels_list[i], dtype=np.int64)

        if n_classes is None:
            n_classes_ = int(max(y_tr.max() if y_tr.size else 0, y_te.max() if y_te.size else 0) + 1)
        else:
            n_classes_ = int(n_classes)

        tr_cnt = np.bincount(y_tr, minlength=n_classes_)
        va_cnt = np.bincount(y_va, minlength=n_classes_) if y_va.size else np.zeros(n_classes_, dtype=np.int64)
        te_cnt = np.bincount(y_te, minlength=n_classes_)

        print(f"Fold {i+1}: train {tr_cnt.tolist()} | val {va_cnt.tolist()} | test {te_cnt.tolist()}")



def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()
#
# class TCGADataset(Dataset):
#
#     def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True,keep_same_psize=0,is_train=False):
#         """
#         Args
#         :param images:
#         :param transform: optional transform to be applied on a sample
#         """
#         super(TCGADataset, self).__init__()
#
#         self.patient_name = file_name
#         self.patient_label = file_label
#         self.max_patch = max_patch
#         self.root = root
#         self.all_pts = os.listdir(os.path.join(self.root,'h5')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt'))
#         self.slide_name = []
#         self.slide_label = []
#         self.persistence = persistence
#         self.keep_same_psize = keep_same_psize
#         self.is_train = is_train
#
#         for i,_patient_name in enumerate(self.patient_name):
#             _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
#             _ids = np.where(_sides != '0')[0]
#             for _idx in _ids:
#                 if persistence:
#                     self.slide_name.append(torch.load(os.path.join(self.root,'pt',_sides[_idx])))
#                 else:
#                     self.slide_name.append(_sides[_idx])
#                 self.slide_label.append(self.patient_label[i])
#         self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]
#
#     def __len__(self):
#         return len(self.slide_name)
#
#     def __getitem__(self, idx):
#         """
#         Args
#         :param idx: the index of item
#         :return: image and its label
#         """
#         file_path = self.slide_name[idx]
#         label = self.slide_label[idx]
#
#         if self.persistence:
#             features = file_path
#         else:
#             features = torch.load(os.path.join(self.root,'pt',file_path))
#         return features , int(label)

###################  单pt文件 #####
class TCGADataset(Dataset):
    """
    TCGA-PRAD PAM50 软标签回归数据集。

    目录结构（args.dataset_root）
    ------------------------------
    dataset_root/
    └── pt/
        ├── TCGA-HI-7170-01Z-00-DX1.xxxx.pt   # Tensor(N,D) 或 {'features':Tensor}
        ├── TCGA-VP-A87E-01Z-00-DX1.xxxx.pt
        └── ...

    Parameters
    ----------
    file_name   : ndarray [M], slide_id（切片级 ID，与 pt 文件名前缀对应）
    file_label  : ndarray [M, 3], float32，软标签 [Basal, LumA, LumB]
    file_diff12 : ndarray [M] 或 None，float32，PAM50 置信度分差
                  为 None 时 __getitem__ 第 4 项返回 tensor(-1.)（哨兵值）
    max_patch   : int，每张切片最多使用的 patch 数（-1 表示不限）
    root        : str，dataset_root 路径
    persistence : bool
                  True  → __init__ 时预加载所有特征到内存（快，内存大）
                  False → __getitem__ 时按需加载（慢，内存省）
    is_train    : bool，预留给训练期数据增强（当前未使用）
    """

    def __init__(
            self,
            file_name: np.ndarray,
            file_label: np.ndarray,  # [M, 3] float32
            file_diff12: np.ndarray | None = None,
            max_patch: int = -1,
            root: str = None,
            persistence: bool = True,
            is_train: bool = False,
    ):
        super().__init__()

        self.max_patch = max_patch
        self.root = root
        self.persistence = persistence
        self.is_train = is_train

        # ── 验证输入 ─────────────────────────────────────────────────────────
        file_label = np.asarray(file_label, dtype=np.float32)
        if file_label.ndim != 2 or file_label.shape[1] != 3:
            raise ValueError(
                f'file_label must be shape [M, 3], got {file_label.shape}'
            )
        if len(file_name) != len(file_label):
            raise ValueError(
                f'file_name ({len(file_name)}) and file_label '
                f'({len(file_label)}) length mismatch'
            )

        # ── 扫描 pt 目录，建立文件名集合 ─────────────────────────────────────
        pt_dir = os.path.join(root, 'pt')
        all_pts = set(os.listdir(pt_dir))  # e.g. {'TCGA-HI-7170-01Z-...pt', ...}

        # ── 构建切片级索引 ────────────────────────────────────────────────────
        # 使用 slide_id 前缀匹配（startswith），避免子串误匹配
        self.slide_feat = []  # list[Tensor] (persistence) 或 list[str] (lazy)
        self.slide_label = []  # list of ndarray (3,)
        self.slide_diff12 = []  # list of float
        self.slide_name = []  # list[str]，slide_id（用于结果追踪）

        for i, slide_id in enumerate(file_name):
            # slide_id 示例：TCGA-HI-7170-01Z-00-DX1.a823f5e3-...
            matched = [f for f in all_pts if f.startswith(slide_id)]

            if len(matched) == 0:
                # pt 文件不存在，跳过并警告
                print(f'[TCGADataset] WARNING: no pt file found for {slide_id}, skipped.')
                continue

            for fname in matched:
                fpath = os.path.join(pt_dir, fname)

                if persistence:
                    feat = self._load_features(fpath, max_patch)
                    self.slide_feat.append(feat)
                else:
                    self.slide_feat.append(fpath)

                self.slide_label.append(file_label[i])  # ndarray (3,)
                self.slide_diff12.append(
                    float(file_diff12[i]) if file_diff12 is not None else -1.0
                )
                self.slide_name.append(slide_id)

        if len(self.slide_feat) == 0:
            raise RuntimeError(
                f'TCGADataset: no valid slides found. '
                f'Check dataset_root="{root}" and file_name contents.'
            )

    # ════════════════════════════════════════════════════════════════════════
    # 内部辅助
    # ════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _load_features(fpath: str, max_patch: int) -> torch.Tensor:
        """
        加载 .pt 文件，兼容两种格式：
          · dict {'features': Tensor(N, D), ...}  ← CLAM 标准格式
          · Tensor (N, D)                          ← 旧格式
        并按 max_patch 截断。
        """
        data = torch.load(fpath, map_location='cpu')

        if isinstance(data, dict):
            feat = data['features']  # Tensor (N, D)
        elif isinstance(data, torch.Tensor):
            feat = data
        else:
            raise TypeError(
                f'Unsupported feature file format: {type(data)} in {fpath}'
            )

        if max_patch > 0 and feat.shape[0] > max_patch:
            # 随机截断（训练 / 推断一致）
            idx = torch.randperm(feat.shape[0])[:max_patch]
            feat = feat[idx]

        return feat  # (N', D)

    # ════════════════════════════════════════════════════════════════════════
    # Dataset 接口
    # ════════════════════════════════════════════════════════════════════════

    def __len__(self) -> int:
        return len(self.slide_feat)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        features : Tensor (N, D)，patch 特征
        target   : Tensor (3,) float32，软标签 [Basal, LumA, LumB]
        name     : str，slide_id
        diff12   : Tensor scalar float32
                   diff12 ≥ 0 → 真实置信度分差（供 criterion 使用）
                   diff12 = -1 → 无 diff12 信息（criterion 内部会忽略）
        """
        # ── 特征加载 ─────────────────────────────────────────────────────────
        if self.persistence:
            features = self.slide_feat[idx]
        else:
            features = self._load_features(self.slide_feat[idx], self.max_patch)

        # ── 标签 ─────────────────────────────────────────────────────────────
        target = torch.tensor(self.slide_label[idx], dtype=torch.float32)  # (3,)
        diff12 = torch.tensor(self.slide_diff12[idx], dtype=torch.float32)  # scalar
        name = self.slide_name[idx]

        return features, target, name, diff12

    def __repr__(self) -> str:
        return (
            f'TCGADataset('
            f'n_slides={len(self)}, '
            f'persistence={self.persistence}, '
            f'max_patch={self.max_patch}, '
            f'root="{self.root}")'
        )

class C16Dataset(Dataset):

    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images:
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.file_name_list  = [str(_p) for _p in self.file_name]
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)

        label = int(self.slide_label[idx])
        patient_name = self.file_name_list[idx]

        return features , label , patient_name



#---------------------------------------------
#
# """
# loss.py
# =======
# PAM50 软标签回归损失函数
#
# 设计依据（TCGA-PRAD, N=369）
# -----------------------------
# 类别分布：Basal 7.0%，LumA 77.5%，LumB 15.4%
# 软标签均值：Basal=0.058，LumA=0.432，LumB=0.096
#
# 类别权重推导（频率倒数 → 开方缩放 → 归一化至 LumA=1）：
#   频率倒数：Basal=14.3，LumA=1.29，LumB=6.49
#   开方：    Basal=3.78，LumA=1.14，LumB=2.55
#   归一化：  α_Basal≈3.5，α_LumA=1.0，α_LumB≈2.2
#   （开方缩放目的：避免 Basal 权重过大导致梯度爆炸）
#
# 损失公式：
#   L = Σ_{c} α_c · w_sample · (pred_c - y_c)²
#
#   其中 w_sample（可选）= diff12 置信度权重，对标签不确定样本降权
# """
#
# class PAM50WeightedMSELoss(nn.Module):
#     """
#     PAM50 三类软标签加权 MSE 损失。
#
#     特性
#     ----
#     1. 类别权重 alpha  : 对抗 LumA 77.5% 的类别不均衡
#     2. 样本置信度权重  : 可选，传入 diff12 对标签不确定样本降权
#     3. 支持 batch 输入 : pred/target shape 为 (3,) 或 (B, 3) 均可
#     4. reduction 选项  : 'mean'（默认）/ 'sum' / 'none'
#
#     Parameters
#     ----------
#     alpha_Basal : float, 默认 3.5
#         Basal 类别权重（频率倒数开方归一化，详见模块 docstring）
#     alpha_LumA  : float, 默认 1.0
#         LumA 类别权重（基准）
#     alpha_LumB  : float, 默认 2.2
#         LumB 类别权重
#     use_conf_weight : bool, 默认 True
#         是否启用 diff12 置信度权重
#     conf_weight_min : float, 默认 0.3
#         diff12=0 时的最低权重（避免完全忽略 Ambiguous 样本）
#     reduction : str, 默认 'mean'
#         'mean' | 'sum' | 'none'
#     """
#
#     def __init__(
#         self,
#         alpha_Basal: float     = 3.5,
#         alpha_LumA: float      = 1.0,
#         alpha_LumB: float      = 2.2,
#         use_conf_weight: bool  = True,
#         conf_weight_min: float = 0.3,
#         reduction: str         = 'mean',
#     ):
#         super().__init__()
#
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(f"reduction must be 'mean'|'sum'|'none', got '{reduction}'")
#
#         self.register_buffer(
#             'alpha',
#             torch.tensor([alpha_Basal, alpha_LumA, alpha_LumB], dtype=torch.float32),
#         )
#         self.use_conf_weight = use_conf_weight
#         self.conf_weight_min = conf_weight_min
#         self.reduction       = reduction
#
#     def forward(
#         self,
#         pred:   torch.Tensor,                  # (3,) 或 (B, 3)
#         target: torch.Tensor,                  # (3,) 或 (B, 3)
#         diff12: torch.Tensor | None = None,    # (,) 或 (B,)，可选
#     ) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         pred   : Tensor (3,) 或 (B, 3)，三路 Sigmoid 输出
#         target : Tensor (3,) 或 (B, 3)，原始 PAM50 软标签
#         diff12 : Tensor (,) 或 (B,) 或 None
#                  PAM50 分类器 top1-top2 分差，越大标签越可信
#                  为 None 时不施加置信度权重
#
#         Returns
#         -------
#         Tensor：标量（reduction='mean'|'sum'）或 (B,)（reduction='none'）
#         """
#         # ── 维度统一为 (B, 3) ────────────────────────────────────────────────
#         squeeze_output = False
#         if pred.dim() == 1:
#             pred   = pred.unsqueeze(0)    # (1, 3)
#             target = target.unsqueeze(0)
#             if diff12 is not None and diff12.dim() == 0:
#                 diff12 = diff12.unsqueeze(0)
#             squeeze_output = True
#
#         B = pred.shape[0]
#
#         # ── per-class 加权 MSE：(B, 3) ──────────────────────────────────────
#         alpha = self.alpha.to(pred.device)               # (3,)
#         mse_per_class = (pred - target) ** 2             # (B, 3)
#         weighted      = mse_per_class * alpha            # (B, 3)，广播
#
#         # ── 样本级损失：(B,) ─────────────────────────────────────────────────
#         loss_per_sample = weighted.sum(dim=1)            # (B,)
#
#         # ── 置信度权重（可选） ────────────────────────────────────────────────
#         if self.use_conf_weight and diff12 is not None:
#             diff12 = diff12.to(pred.device).float()      # (B,)
#             # 线性映射：diff12 ∈ [0,1] → w ∈ [conf_weight_min, 1.0]
#             # diff12=0（完全 Ambiguous）→ w=conf_weight_min
#             # diff12=1（极度确定）      → w=1.0
#             conf_w = self.conf_weight_min + (1.0 - self.conf_weight_min) * diff12.clamp(0, 1)
#             loss_per_sample = loss_per_sample * conf_w   # (B,)
#
#         # ── reduction ────────────────────────────────────────────────────────
#         if self.reduction == 'mean':
#             loss = loss_per_sample.mean()
#         elif self.reduction == 'sum':
#             loss = loss_per_sample.sum()
#         else:
#             loss = loss_per_sample
#
#         if squeeze_output and self.reduction == 'none':
#             loss = loss.squeeze(0)
#
#         return loss
#
#     def extra_repr(self) -> str:
#         return (
#             f"alpha=[{self.alpha[0]:.1f}, {self.alpha[1]:.1f}, {self.alpha[2]:.1f}]  "
#             f"use_conf_weight={self.use_conf_weight}  "
#             f"conf_weight_min={self.conf_weight_min}  "
#             f"reduction='{self.reduction}'"
#         )
#
#
#
#
# class PAM50FocalMSELoss(nn.Module):
#     """
#     PAM50 Focal-MSE 损失：对大预测误差的样本自适应加权。
#
#     设计依据
#     --------
#     当前问题：Basal 真实值 80.5% 为 0，模型输出接近 0 即获得低 MSE，
#              导致 Basal 梯度信号极弱，训练收敛到 "全输出接近 0" 的局部极小值。
#              WeightedMSE 虽然放大了 Basal 的 alpha，但对"预测接近 0/真实也接近 0"
#              的 easy Basal 样本同样放大，梯度效率低。
#
#     Focal-MSE 公式（per class c，per sample i）：
#       L_focal = α_c · |pred - y|^γ · (pred - y)²
#                   ↑               ↑
#              类别权重      误差调制因子：误差越大权重越高
#
#     等价展开：L_focal = α_c · (pred - y)^(2+γ)
#       γ=0 → 退化为标准 WeightedMSE
#       γ=1 → 大误差样本权重 = 误差本身（三次方惩罚）
#       γ=2 → 四次方惩罚（更激进）
#
#     实验建议：γ=1.0 起步（稳定），效果不足再试 γ=1.5
#
#     与 PAM50WeightedMSELoss 的关系
#     --------------------------------
#     · 保留所有特性：alpha 类别权重、diff12 置信度降权、batch/单样本/reduction
#     · 新增 gamma 参数控制 Focal 强度
#     · gamma=0 时与 PAM50WeightedMSELoss 完全等价（向后兼容）
#
#     Parameters
#     ----------
#     alpha_Basal     : float, 默认 7.0（Focal 版本需要更高，因误差调制会放大小误差）
#     alpha_LumA      : float, 默认 1.0
#     alpha_LumB      : float, 默认 2.2
#     gamma           : float, 默认 1.0，Focal 调制强度
#                       0 → 标准 WeightedMSE
#                       1 → 中等 Focal（推荐起始值）
#                       2 → 强 Focal（对极端误差惩罚更重）
#     use_conf_weight : bool,  默认 True
#     conf_weight_min : float, 默认 0.3
#     reduction       : str,   'mean' | 'sum' | 'none'
#     per_class_gamma : bool,  默认 False
#                       True  → 每类用独立 gamma（Basal 用更高 gamma 更激进）
#                       False → 三类共用同一 gamma
#     gamma_Basal     : float, 仅 per_class_gamma=True 时生效，默认 1.5
#     gamma_LumA      : float, 默认 0.5（LumA 样本已充足，不需要太激进）
#     gamma_LumB      : float, 默认 1.0
#     """
#
#     def __init__(
#         self,
#         alpha_Basal:     float = 7.0,
#         alpha_LumA:      float = 1.0,
#         alpha_LumB:      float = 2.2,
#         gamma:           float = 1.0,
#         use_conf_weight: bool  = True,
#         conf_weight_min: float = 0.3,
#         reduction:       str   = 'mean',
#         per_class_gamma: bool  = False,
#         gamma_Basal:     float = 1.5,
#         gamma_LumA:      float = 0.5,
#         gamma_LumB:      float = 1.0,
#     ):
#         super().__init__()
#
#         if reduction not in ('mean', 'sum', 'none'):
#             raise ValueError(f"reduction must be 'mean'|'sum'|'none', got '{reduction}'")
#         if gamma < 0:
#             raise ValueError(f"gamma must be >= 0, got {gamma}")
#
#         self.register_buffer(
#             'alpha',
#             torch.tensor([alpha_Basal, alpha_LumA, alpha_LumB], dtype=torch.float32),
#         )
#         self.gamma           = float(gamma)
#         self.use_conf_weight = use_conf_weight
#         self.conf_weight_min = conf_weight_min
#         self.reduction       = reduction
#         self.per_class_gamma = per_class_gamma
#
#         if per_class_gamma:
#             self.register_buffer(
#                 'gamma_per_class',
#                 torch.tensor([gamma_Basal, gamma_LumA, gamma_LumB], dtype=torch.float32),
#             )
#
#     def forward(
#         self,
#         pred:   torch.Tensor,               # (3,) 或 (B, 3)
#         target: torch.Tensor,               # (3,) 或 (B, 3)
#         diff12: torch.Tensor | None = None, # (,) 或 (B,)，可选
#     ) -> torch.Tensor:
#         """
#         Parameters
#         ----------
#         pred   : Tensor (3,) 或 (B, 3)，三路 Sigmoid 输出
#         target : Tensor (3,) 或 (B, 3)，原始 PAM50 软标签
#         diff12 : Tensor (,) 或 (B,) 或 None
#
#         Returns
#         -------
#         Tensor：标量或 (B,)
#         """
#         # ── 维度统一为 (B, 3) ────────────────────────────────────────────────
#         squeeze_output = False
#         if pred.dim() == 1:
#             pred   = pred.unsqueeze(0)
#             target = target.unsqueeze(0)
#             if diff12 is not None and diff12.dim() == 0:
#                 diff12 = diff12.unsqueeze(0)
#             squeeze_output = True
#
#         B = pred.shape[0]
#         alpha = self.alpha.to(pred.device)          # (3,)
#
#         # ── Focal-MSE 核心计算 ───────────────────────────────────────────────
#         # abs_err: (B, 3)，绝对误差
#         # focal weight = |err|^gamma，gamma=0 → weight=1（退化为标准MSE）
#         abs_err = (pred - target).abs()             # (B, 3)
#
#         if self.per_class_gamma:
#             gamma_vec = self.gamma_per_class.to(pred.device)   # (3,)
#             # 每类用不同 gamma：Basal gamma 高 → Basal 大误差惩罚更重
#             focal_w = abs_err.pow(gamma_vec)                   # (B, 3)，广播
#         else:
#             focal_w = abs_err.pow(self.gamma)                  # (B, 3)
#
#         # Focal-MSE = α · |err|^γ · err²  =  α · |err|^(γ+2) · sign²  =  α · err^(γ+2)
#         # 注：err² = abs_err²，focal_w * abs_err² = abs_err^(γ+2)
#         focal_mse = focal_w * (pred - target).pow(2)          # (B, 3)
#         weighted  = focal_mse * alpha                          # (B, 3)
#
#         # ── 样本级损失：(B,) ─────────────────────────────────────────────────
#         loss_per_sample = weighted.sum(dim=1)                  # (B,)
#
#         # ── 置信度权重（可选） ────────────────────────────────────────────────
#         if self.use_conf_weight and diff12 is not None:
#             diff12 = diff12.to(pred.device).float()
#             # diff12 < 0 时为哨兵值（无 diff12 信息），跳过降权
#             valid_mask = diff12 >= 0                           # (B,)
#             conf_w = torch.ones_like(diff12)
#             conf_w[valid_mask] = (
#                 self.conf_weight_min
#                 + (1.0 - self.conf_weight_min) * diff12[valid_mask].clamp(0, 1)
#             )
#             loss_per_sample = loss_per_sample * conf_w
#
#         # ── reduction ────────────────────────────────────────────────────────
#         if self.reduction == 'mean':
#             loss = loss_per_sample.mean()
#         elif self.reduction == 'sum':
#             loss = loss_per_sample.sum()
#         else:
#             loss = loss_per_sample
#
#         if squeeze_output and self.reduction == 'none':
#             loss = loss.squeeze(0)
#
#         return loss
#
#     def extra_repr(self) -> str:
#         base = (
#             f"alpha=[{self.alpha[0]:.1f}, {self.alpha[1]:.1f}, {self.alpha[2]:.1f}]  "
#             f"gamma={self.gamma}  "
#             f"per_class_gamma={self.per_class_gamma}  "
#             f"use_conf_weight={self.use_conf_weight}  "
#             f"reduction='{self.reduction}'"
#         )
#         if self.per_class_gamma:
#             gc = self.gamma_per_class
#             base += f"  gamma_per_class=[{gc[0]:.1f}, {gc[1]:.1f}, {gc[2]:.1f}]"
#         return base
#
#


#---------------------------------

"""
loss.py
=======
PAM50 软标签回归损失函数

设计依据（TCGA-PRAD, N=369）
-----------------------------
类别分布：Basal 7.0%，LumA 77.5%，LumB 15.4%
软标签均值：Basal=0.058，LumA=0.432，LumB=0.096

类别权重推导（频率倒数 → 开方缩放 → 归一化至 LumA=1）：
  频率倒数：Basal=14.3，LumA=1.29，LumB=6.49
  开方：    Basal=3.78，LumA=1.14，LumB=2.55
  归一化：  α_Basal≈3.5，α_LumA=1.0，α_LumB≈2.2
  （开方缩放目的：避免 Basal 权重过大导致梯度爆炸）

损失公式：
  L = Σ_{c} α_c · w_sample · (pred_c - y_c)²

  其中 w_sample（可选）= diff12 置信度权重，对标签不确定样本降权
"""

class PAM50WeightedMSELoss(nn.Module):
    """
    PAM50 三类软标签加权 MSE 损失。

    特性
    ----
    1. 类别权重 alpha  : 对抗 LumA 77.5% 的类别不均衡
    2. 样本置信度权重  : 可选，传入 diff12 对标签不确定样本降权
    3. 支持 batch 输入 : pred/target shape 为 (3,) 或 (B, 3) 均可
    4. reduction 选项  : 'mean'（默认）/ 'sum' / 'none'

    Parameters
    ----------
    alpha_Basal : float, 默认 3.5
        Basal 类别权重（频率倒数开方归一化，详见模块 docstring）
    alpha_LumA  : float, 默认 1.0
        LumA 类别权重（基准）
    alpha_LumB  : float, 默认 2.2
        LumB 类别权重
    use_conf_weight : bool, 默认 True
        是否启用 diff12 置信度权重
    conf_weight_min : float, 默认 0.3
        diff12=0 时的最低权重（避免完全忽略 Ambiguous 样本）
    reduction : str, 默认 'mean'
        'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha_Basal: float     = 3.5,
        alpha_LumA: float      = 1.0,
        alpha_LumB: float      = 2.2,
        use_conf_weight: bool  = True,
        conf_weight_min: float = 0.3,
        reduction: str         = 'mean',
    ):
        super().__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got '{reduction}'")

        self.register_buffer(
            'alpha',
            torch.tensor([alpha_Basal, alpha_LumA, alpha_LumB], dtype=torch.float32),
        )
        self.use_conf_weight = use_conf_weight
        self.conf_weight_min = conf_weight_min
        self.reduction       = reduction

    def forward(
        self,
        pred:   torch.Tensor,                  # (3,) 或 (B, 3)
        target: torch.Tensor,                  # (3,) 或 (B, 3)
        diff12: torch.Tensor | None = None,    # (,) 或 (B,)，可选
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : Tensor (3,) 或 (B, 3)，三路 Sigmoid 输出
        target : Tensor (3,) 或 (B, 3)，原始 PAM50 软标签
        diff12 : Tensor (,) 或 (B,) 或 None
                 PAM50 分类器 top1-top2 分差，越大标签越可信
                 为 None 时不施加置信度权重

        Returns
        -------
        Tensor：标量（reduction='mean'|'sum'）或 (B,)（reduction='none'）
        """
        # ── 维度统一为 (B, 3) ────────────────────────────────────────────────
        squeeze_output = False
        if pred.dim() == 1:
            pred   = pred.unsqueeze(0)    # (1, 3)
            target = target.unsqueeze(0)
            if diff12 is not None and diff12.dim() == 0:
                diff12 = diff12.unsqueeze(0)
            squeeze_output = True

        B = pred.shape[0]

        # ── per-class 加权 MSE：(B, 3) ──────────────────────────────────────
        alpha = self.alpha.to(pred.device)               # (3,)
        mse_per_class = (pred - target) ** 2             # (B, 3)
        weighted      = mse_per_class * alpha            # (B, 3)，广播

        # ── 样本级损失：(B,) ─────────────────────────────────────────────────
        loss_per_sample = weighted.sum(dim=1)            # (B,)

        # ── 置信度权重（可选） ────────────────────────────────────────────────
        if self.use_conf_weight and diff12 is not None:
            diff12 = diff12.to(pred.device).float()      # (B,)
            # 线性映射：diff12 ∈ [0,1] → w ∈ [conf_weight_min, 1.0]
            # diff12=0（完全 Ambiguous）→ w=conf_weight_min
            # diff12=1（极度确定）      → w=1.0
            conf_w = self.conf_weight_min + (1.0 - self.conf_weight_min) * diff12.clamp(0, 1)
            loss_per_sample = loss_per_sample * conf_w   # (B,)

        # ── reduction ────────────────────────────────────────────────────────
        if self.reduction == 'mean':
            loss = loss_per_sample.mean()
        elif self.reduction == 'sum':
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        if squeeze_output and self.reduction == 'none':
            loss = loss.squeeze(0)

        return loss

    def extra_repr(self) -> str:
        return (
            f"alpha=[{self.alpha[0]:.1f}, {self.alpha[1]:.1f}, {self.alpha[2]:.1f}]  "
            f"use_conf_weight={self.use_conf_weight}  "
            f"conf_weight_min={self.conf_weight_min}  "
            f"reduction='{self.reduction}'"
        )




class PAM50FocalMSELoss(nn.Module):
    """
    PAM50 Focal-MSE 损失：对大预测误差的样本自适应加权。

    设计依据
    --------
    当前问题：Basal 真实值 80.5% 为 0，模型输出接近 0 即获得低 MSE，
             导致 Basal 梯度信号极弱，训练收敛到 "全输出接近 0" 的局部极小值。
             WeightedMSE 虽然放大了 Basal 的 alpha，但对"预测接近 0/真实也接近 0"
             的 easy Basal 样本同样放大，梯度效率低。

    Focal-MSE+ 公式（per class c，per sample i）：
      L = α_c · (1 + γ · |pred - y|) · (pred - y)²
               ↑                   ↑
          基础权重1       误差调制加法项（大误差额外惩罚）

    设计原理（关键修正）
    -------------------
    原版 Focal-MSE：α · |err|^γ · err²
      → 当 err ∈ [0,1]（PAM50 软标签误差必然在此范围），
        |err|^γ ∈ [0,1]，所有损失值被「缩小」而非放大
      → 梯度整体减小，模型收敛变慢，实验证实性能下降

    Focal-MSE+（本实现）：α · (1 + γ·|err|) · err²
      → err=0:   weight = 1.0（等于标准 MSE，easy 样本不惩罚）
      → err=0.5: weight = 1.5（大50%惩罚）
      → err=0.9: weight = 1.9（大90%惩罚）
      → γ=0 严格退化为标准 WeightedMSE ✓
      → alpha_Basal=7 无需重新校准 ✓
      → 所有样本梯度不减小，大误差额外增大 ✓

    实验建议：γ=1.0 起步（稳定），效果不足再试 γ=2.0

    与 PAM50WeightedMSELoss 的关系
    --------------------------------
    · 保留所有特性：alpha 类别权重、diff12 置信度降权、batch/单样本/reduction
    · 新增 gamma 参数控制 Focal 强度
    · gamma=0 时与 PAM50WeightedMSELoss 完全等价（向后兼容）

    Parameters
    ----------
    alpha_Basal     : float, 默认 7.0（Focal 版本需要更高，因误差调制会放大小误差）
    alpha_LumA      : float, 默认 1.0
    alpha_LumB      : float, 默认 2.2
    gamma           : float, 默认 1.0，Focal 调制强度
                      0 → 标准 WeightedMSE
                      1 → 中等 Focal（推荐起始值）
                      2 → 强 Focal（对极端误差惩罚更重）
    use_conf_weight : bool,  默认 True
    conf_weight_min : float, 默认 0.3
    reduction       : str,   'mean' | 'sum' | 'none'
    per_class_gamma : bool,  默认 False
                      True  → 每类用独立 gamma（Basal 用更高 gamma 更激进）
                      False → 三类共用同一 gamma
    gamma_Basal     : float, 仅 per_class_gamma=True 时生效，默认 1.5
    gamma_LumA      : float, 默认 0.5（LumA 样本已充足，不需要太激进）
    gamma_LumB      : float, 默认 1.0
    """

    def __init__(
        self,
        alpha_Basal:     float = 7.0,
        alpha_LumA:      float = 1.0,
        alpha_LumB:      float = 2.2,
        gamma:           float = 1.0,
        use_conf_weight: bool  = True,
        conf_weight_min: float = 0.3,
        reduction:       str   = 'mean',
        per_class_gamma: bool  = False,
        gamma_Basal:     float = 1.5,
        gamma_LumA:      float = 0.5,
        gamma_LumB:      float = 1.0,
    ):
        super().__init__()

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction must be 'mean'|'sum'|'none', got '{reduction}'")
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")

        self.register_buffer(
            'alpha',
            torch.tensor([alpha_Basal, alpha_LumA, alpha_LumB], dtype=torch.float32),
        )
        self.gamma           = float(gamma)
        self.use_conf_weight = use_conf_weight
        self.conf_weight_min = conf_weight_min
        self.reduction       = reduction
        self.per_class_gamma = per_class_gamma

        if per_class_gamma:
            self.register_buffer(
                'gamma_per_class',
                torch.tensor([gamma_Basal, gamma_LumA, gamma_LumB], dtype=torch.float32),
            )

    def forward(
        self,
        pred:   torch.Tensor,               # (3,) 或 (B, 3)
        target: torch.Tensor,               # (3,) 或 (B, 3)
        diff12: torch.Tensor | None = None, # (,) 或 (B,)，可选
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred   : Tensor (3,) 或 (B, 3)，三路 Sigmoid 输出
        target : Tensor (3,) 或 (B, 3)，原始 PAM50 软标签
        diff12 : Tensor (,) 或 (B,) 或 None

        Returns
        -------
        Tensor：标量或 (B,)
        """
        # ── 维度统一为 (B, 3) ────────────────────────────────────────────────
        squeeze_output = False
        if pred.dim() == 1:
            pred   = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            if diff12 is not None and diff12.dim() == 0:
                diff12 = diff12.unsqueeze(0)
            squeeze_output = True

        B = pred.shape[0]
        alpha = self.alpha.to(pred.device)          # (3,)

        # ── Focal-MSE+ 核心计算 ──────────────────────────────────────────────
        # 公式：L = α · (1 + γ·|err|) · err²
        # 当 err ∈ [0,1]（PAM50 误差范围）：
        #   γ=0 → weight=1，严格等于标准 MSE
        #   γ>0 → weight=1+γ|err|>1，大误差额外惩罚，梯度只增不减
        abs_err = (pred - target).abs()             # (B, 3)
        err_sq  = (pred - target).pow(2)            # (B, 3)

        if self.per_class_gamma:
            gamma_vec = self.gamma_per_class.to(pred.device)   # (3,)
            # 每类用不同 gamma：Basal gamma 高 → Basal 大误差额外惩罚更重
            focal_w = 1.0 + gamma_vec * abs_err                # (B, 3)，广播
        else:
            focal_w = 1.0 + self.gamma * abs_err               # (B, 3)

        # focal_mse = (1 + γ·|err|) · err²
        focal_mse = focal_w * err_sq                           # (B, 3)
        weighted  = focal_mse * alpha                          # (B, 3)

        # ── 样本级损失：(B,) ─────────────────────────────────────────────────
        loss_per_sample = weighted.sum(dim=1)                  # (B,)

        # ── 置信度权重（可选） ────────────────────────────────────────────────
        if self.use_conf_weight and diff12 is not None:
            diff12 = diff12.to(pred.device).float()
            # diff12 < 0 时为哨兵值（无 diff12 信息），跳过降权
            valid_mask = diff12 >= 0                           # (B,)
            conf_w = torch.ones_like(diff12)
            conf_w[valid_mask] = (
                self.conf_weight_min
                + (1.0 - self.conf_weight_min) * diff12[valid_mask].clamp(0, 1)
            )
            loss_per_sample = loss_per_sample * conf_w

        # ── reduction ────────────────────────────────────────────────────────
        if self.reduction == 'mean':
            loss = loss_per_sample.mean()
        elif self.reduction == 'sum':
            loss = loss_per_sample.sum()
        else:
            loss = loss_per_sample

        if squeeze_output and self.reduction == 'none':
            loss = loss.squeeze(0)

        return loss

    def extra_repr(self) -> str:
        base = (
            f"Focal-MSE+: α·(1+γ|err|)·err²  "
            f"alpha=[{self.alpha[0]:.1f}, {self.alpha[1]:.1f}, {self.alpha[2]:.1f}]  "
            f"gamma={self.gamma}  "
            f"per_class_gamma={self.per_class_gamma}  "
            f"use_conf_weight={self.use_conf_weight}  "
            f"reduction='{self.reduction}'"
        )
        if self.per_class_gamma:
            gc = self.gamma_per_class
            base += f"  gamma_per_class=[{gc[0]:.1f}, {gc[1]:.1f}, {gc[2]:.1f}]"
        return base



# loss.py 新增 ProtoContrastLoss
class ProtoContrastLoss(nn.Module):
    """
    原型对比损失（Prototype NCE）
    文献：ProtoMIL MICCAI 2024, CLAM Nature BME 2021

    在每个 batch 内，用当前 batch 的同类样本聚合向量作为原型，
    最大化 anchor 与同类原型的相似度，最小化与异类原型相似度。

    特别设计：对 Basal 样本施加 2× 权重，强化 Basal 分离度。
    """

    def __init__(self, temperature: float = 0.07,
                 basal_weight: float = 2.0):
        super().__init__()
        self.tau = temperature
        self.w_bas = basal_weight

    def forward(self,
                z_B: torch.Tensor,  # [B, 512]  Basal 聚合向量
                z_LA: torch.Tensor,  # [B, 512]
                z_LB: torch.Tensor,  # [B, 512]
                targets: torch.Tensor  # [B, 3]  软标签
                ) -> torch.Tensor:
        """
        用硬标签（argmax）确定每个样本的"主类"，
        在三路聚合向量拼接的空间中做 SupCon loss。
        """
        labels = targets.argmax(dim=1)  # [B]  0=Basal,1=LumA,2=LumB
        B = z_B.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=z_B.device)

        # 取各路对应主类的聚合向量作为 anchor
        # anchor[i] = z_label[i] 对应通路的向量
        anchors = torch.stack([
            z_B[i] if labels[i] == 0
            else z_LA[i] if labels[i] == 1
            else z_LB[i]
            for i in range(B)
        ], dim=0)  # [B, 512]

        anchors = F.normalize(anchors, dim=-1)

        # 计算相似度矩阵
        sim = anchors @ anchors.T / self.tau  # [B, B]

        # 屏蔽自身
        mask_self = torch.eye(B, dtype=torch.bool, device=z_B.device)
        sim = sim.masked_fill(mask_self, float('-inf'))

        # 正例 mask：同类
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [B,B]
        mask_pos = mask_pos & ~mask_self

        # InfoNCE loss
        log_prob = F.log_softmax(sim, dim=-1)  # [B, B]
        n_pos = mask_pos.float().sum(dim=-1).clamp(min=1)
        loss_per = -(log_prob * mask_pos.float()).sum(dim=-1) / n_pos

        # Basal 加权
        w = torch.where(labels == 0,
                        torch.full_like(loss_per, self.w_bas),
                        torch.ones_like(loss_per))
        return (loss_per * w).mean()


def basal_mixup(features: torch.Tensor,
                targets: torch.Tensor,
                alpha: float = 0.4) -> tuple:
    """
    Basal intra-class MixUp 数据增强。
    仅对当前 batch 内的 Basal 样本做类内混合，非 Basal 样本不受影响。

    Parameters
    ----------
    features : Tensor [B, N, D]，已在 device 上
    targets  : Tensor [B, 3]，软标签 [Basal, LumA, LumB]，已在 device 上
    alpha    : float，Beta 分布参数，控制混合比例的集中程度
               0.4 → lam 集中在 0.3~0.7，混合效果适中
               0.1 → lam 集中在极端值（接近原样本），保守

    Returns
    -------
    features : Tensor [B, N, D]，混合后（Basal 行已替换，其余行不变）
    targets  : Tensor [B, 3]，混合后软标签
    """
    # ── 保护：确保输入是 3D tensor ─────────────────────────────────────────
    if features.dim() == 2:
        # bag 是单样本 [N, D]，无法做 batch 级 mixup，直接返回
        return features, targets

    # ── 找 Basal 样本在当前 batch 中的索引 ────────────────────────────────
    # targets.argmax(dim=1) == 0 表示软标签中 Basal 值最大（硬标签为 Basal）
    is_basal = (targets.argmax(dim=1) == 0)
    idx_b = is_basal.nonzero(as_tuple=False).flatten()  # [n_basal]

    if len(idx_b) < 2:
        # 当前 batch 内 Basal 样本不足 2 例，跳过 mixup
        return features, targets

    # ── 在 Basal 子集内随机配对（Bug 1 修复）──────────────────────────────
    # perm 是 [0, n_basal) 内的随机排列，而非 batch 全局索引
    n_basal = len(idx_b)
    perm = torch.randperm(n_basal, device=features.device)
    idx_b2 = idx_b[perm]  # 配对的 Basal 样本在 batch 中的索引

    # ── 采样混合系数（Bug 2 修复：直接用 torch.distributions）──────────────
    # lam ∈ (0, 1)，shape [n_basal]
    # 使用 float32 避免 AMP 下的 dtype 冲突
    lam = torch.from_numpy(
        __import__('numpy').random.beta(alpha, alpha, size=n_basal)
    ).to(dtype=torch.float32, device=features.device)

    # ── 执行 mixup（在原 tensor 的副本上操作）─────────────────────────────
    features = features.clone()
    targets = targets.clone()

    lam_f = lam.view(n_basal, 1, 1)  # [n_basal, 1, 1] → broadcast [n_basal, N, D]
    lam_t = lam.view(n_basal, 1)  # [n_basal, 1]    → broadcast [n_basal, 3]

    features[idx_b] = (lam_f * features[idx_b]
                       + (1.0 - lam_f) * features[idx_b2])
    targets[idx_b] = (lam_t * targets[idx_b]
                      + (1.0 - lam_t) * targets[idx_b2])

    return features, targets


