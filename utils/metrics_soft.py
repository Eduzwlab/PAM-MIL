
CLASS_NAMES = ['Basal', 'LumA', 'LumB']   # 列索引 0 / 1 / 2

def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:

    preds   = np.asarray(preds,   dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    assert preds.shape == targets.shape == (len(preds), 3), \
        f"shape mismatch: preds={preds.shape}, targets={targets.shape}"

    result = {}

    per_pearson, per_spearman, per_mae, per_rmse = [], [], [], []

    for i, name in enumerate(CLASS_NAMES):
        y_pred = preds[:, i]
        y_true = targets[:, i]

        # Pearson r（当标准差为 0 时置 0，避免 NaN）
        if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
            r = 0.0
        else:
            r, _ = pearsonr(y_true, y_pred)

        # Spearman ρ
        rho, _ = spearmanr(y_true, y_pred)

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

        result[f'pearson_{name}']  = float(r)
        result[f'spearman_{name}'] = float(rho)
        result[f'mae_{name}']      = float(mae)
        result[f'rmse_{name}']     = float(rmse)

        per_pearson.append(r)
        per_spearman.append(rho)
        per_mae.append(mae)
        per_rmse.append(rmse)

    result['pearson_macro_overall']  = float(np.mean(per_pearson))
    result['spearman_macro_overall'] = float(np.mean(per_spearman))
    result['mae_macro_overall']      = float(np.mean(per_mae))
    result['rmse_macro_overall']     = float(np.mean(per_rmse))

    flat_pred   = preds.reshape(-1)
    flat_target = targets.reshape(-1)

    if flat_target.std() < 1e-8 or flat_pred.std() < 1e-8:
        result['pearson_flat_overall']  = 0.0
        result['spearman_flat_overall'] = 0.0
    else:
        r_flat,   _ = pearsonr(flat_target, flat_pred)
        rho_flat, _ = spearmanr(flat_target, flat_pred)
        result['pearson_flat_overall']  = float(r_flat)
        result['spearman_flat_overall'] = float(rho_flat)

    return result

def compute_classification_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:

    preds   = np.asarray(preds,   dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)

    # argmax → 整数类别（0=Basal, 1=LumA, 2=LumB）
    true_cls = np.argmax(targets, axis=1)   # [N]
    pred_cls = np.argmax(preds,   axis=1)   # [N]

    result = {}

    n_classes = len(CLASS_NAMES)
    true_onehot = np.zeros((len(true_cls), n_classes), dtype=np.float64)
    for i, c in enumerate(true_cls):
        true_onehot[i, c] = 1.0

    try:
        result['auroc_ovr_macro'] = float(
            roc_auc_score(true_onehot, preds, multi_class='ovr', average='macro')
        )
    except ValueError:
        result['auroc_ovr_macro'] = float('nan')

    try:
        result['auroc_ovr_micro'] = float(
            roc_auc_score(true_onehot, preds, multi_class='ovr', average='micro')
        )
    except ValueError:
        result['auroc_ovr_micro'] = float('nan')

    result['acc']          = float(accuracy_score(true_cls, pred_cls))
    result['balanced_acc'] = float(balanced_accuracy_score(true_cls, pred_cls))
    result['f1_macro']     = float(f1_score(true_cls, pred_cls,
                                             average='macro', zero_division=0))

    return result
═══════════════════════════════════════════════════════════════

def compute_all_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:

    reg  = compute_regression_metrics(preds, targets)
    cls  = compute_classification_metrics(preds, targets)
    return {**reg, **cls}

_REGRESSION_KEYS = [
    'pearson_Basal',  'pearson_LumA',  'pearson_LumB',
    'spearman_Basal', 'spearman_LumA', 'spearman_LumB',
    'mae_Basal',      'mae_LumA',      'mae_LumB',
    'rmse_Basal',     'rmse_LumA',     'rmse_LumB',

    'pearson_macro_overall',  'pearson_flat_overall',
    'spearman_macro_overall', 'spearman_flat_overall',
    'mae_macro_overall',
    'rmse_macro_overall',
]

_CLASSIFICATION_KEYS = [
    'auroc_ovr_macro',
    'auroc_ovr_micro',
    'acc',
    'balanced_acc',
    'f1_macro',
]

_RESULT_KEYS = ['p_name', 'bag_l', 'bag_pre', 'pred_cls']

ALL_METRIC_KEYS = _REGRESSION_KEYS + _CLASSIFICATION_KEYS

def init_ckc_metric() -> dict:

    return {k: [] for k in ALL_METRIC_KEYS + _RESULT_KEYS}


def append_fold_metrics(ckc_metric: dict, fold_result: dict) -> None:
    for k in ALL_METRIC_KEYS:
        if k in fold_result:
            ckc_metric[k].append(fold_result[k])


def resume_ckc_metric(ckc_metric: dict, ckp_path: str) -> int:

    import torch, os
    ckp = torch.load(ckp_path)
    fold_start   = ckp.get('k', 0)
    saved_metric = ckp.get('ckc_metric', {})
    for key in ALL_METRIC_KEYS:
        if key in saved_metric:
            ckc_metric[key].extend(saved_metric[key])
    return fold_start


def _ms(lst) -> tuple:
    """返回 (mean, std)，空列表时返回 (nan, nan)。"""
    arr = np.array(lst, dtype=float)
    if len(arr) == 0:
        return float('nan'), float('nan')
    return float(np.mean(arr)), float(np.std(arr))

def save_predictions(ckc_metric: dict, save_path: str) -> None:

    import pandas as pd

    all_names = [s for fold in ckc_metric['p_name']  for s in fold]
    all_true  = np.concatenate(ckc_metric['bag_l'],   axis=0)   # [N_all, 3]
    all_pred  = np.concatenate(ckc_metric['bag_pre'], axis=0)   # [N_all, 3]
    all_cls   = [c for fold in ckc_metric['pred_cls'] for c in fold]

    datasave = {
        'slide_id':      all_names,
        'true_Basal':    all_true[:, 0],
        'true_LumA':     all_true[:, 1],
        'true_LumB':     all_true[:, 2],
        'pred_Basal':    all_pred[:, 0],
        'pred_LumA':     all_pred[:, 1],
        'pred_LumB':     all_pred[:, 2],
        'true_decision': [CLASS_NAMES[int(np.argmax(t))] for t in all_true],
        'pred_decision': [CLASS_NAMES[c] for c in all_cls],
    }
    pd.DataFrame(datasave).to_csv(save_path, index=False)
    print(f'[Save] predictions → {save_path}')

def log_wandb(ckc_metric: dict, wandb) -> None:
    """将所有 K 折指标的 mean/std 写入 wandb。"""
    log_dict = {}

    reg_groups = {
        'pearson':  ['pearson_Basal',  'pearson_LumA',  'pearson_LumB',
                     'pearson_macro_overall', 'pearson_flat_overall'],
        'spearman': ['spearman_Basal', 'spearman_LumA', 'spearman_LumB',
                     'spearman_macro_overall', 'spearman_flat_overall'],
        'mae':      ['mae_Basal',  'mae_LumA',  'mae_LumB',  'mae_macro_overall'],
        'rmse':     ['rmse_Basal', 'rmse_LumA', 'rmse_LumB', 'rmse_macro_overall'],
    }
    for prefix, keys in reg_groups.items():
        for k in keys:
            suffix = k.replace(f'{prefix}_', '')
            m, s = _ms(ckc_metric[k])
            log_dict[f'cross_val/{prefix}_{suffix}_mean'] = m
            log_dict[f'cross_val/{prefix}_{suffix}_std']  = s

    for k in _CLASSIFICATION_KEYS:
        m, s = _ms(ckc_metric[k])
        log_dict[f'cross_val/{k}_mean'] = m
        log_dict[f'cross_val/{k}_std']  = s

    wandb.log(log_dict)

def print_cv_summary(ckc_metric: dict) -> None:

    W = 68
    print('\n' + '═' * W)
    print(' Cross Validation Summary — PAM50 Soft-Label Regression')
    print('═' * W)

    print('\n【Regression Metrics】')
    col_w = 17
    header = f"  {'Metric':<18}" + ''.join(
        f"{c:>{col_w}}" for c in ['Basal', 'LumA', 'LumB', 'Macro-OA', 'Flat-OA']
    )
    print(header)
    print('  ' + '─' * (W - 2))

    reg_rows = [
        ('Pearson r',
         ['pearson_Basal',  'pearson_LumA',  'pearson_LumB',
          'pearson_macro_overall', 'pearson_flat_overall']),
        ('Spearman ρ',
         ['spearman_Basal', 'spearman_LumA', 'spearman_LumB',
          'spearman_macro_overall', 'spearman_flat_overall']),
        ('MAE',
         ['mae_Basal', 'mae_LumA', 'mae_LumB',
          'mae_macro_overall', None]),
        ('RMSE',
         ['rmse_Basal', 'rmse_LumA', 'rmse_LumB',
          'rmse_macro_overall', None]),
    ]

    for label, keys in reg_rows:
        row = f"  {label:<18}"
        for k in keys:
            if k is None:
                row += f"{'—':>{col_w}}"
            else:
                m, s = _ms(ckc_metric[k])
                row += f"{f'{m:.3f}±{s:.3f}':>{col_w}}"
        print(row)

    print()
    print('  Note: Macro-OA = macro-average of 3 classes (equal weight, Basal unmasked)')
    print('        Flat-OA  = Pearson/Spearman on [N×3] flattened vector')
    print('                   (Flat vs Macro gap → class-level systematic bias)')

    print('\n【Classification Decision Metrics】')
    cls_rows = [
        ('AUROC OvR Macro ★', 'auroc_ovr_macro',
         '← 主指标，对类别不平衡最鲁棒'),
        ('AUROC OvR Micro',   'auroc_ovr_micro',
         '← 受 LumA 主导，参考用'),
        ('Accuracy',          'acc',            ''),
        ('Balanced Accuracy', 'balanced_acc',
         '← 等价于 Macro-Recall，Basal 敏感'),
        ('F1 Macro',          'f1_macro',       ''),
    ]
    print(f"  {'Metric':<22} {'Mean±Std':>12}   Note")
    print('  ' + '─' * (W - 2))
    for label, key, note in cls_rows:
        m, s = _ms(ckc_metric[key])
        print(f"  {label:<22} {f'{m:.3f}±{s:.3f}':>12}   {note}")

    print('═' * W + '\n')

if __name__ == '__main__':
    import pandas as pd

    rng = np.random.default_rng(42)
    N   = 369

    df = pd.read_csv('/mnt/user-data/uploads/Matched_Result_unique_slides_final_label.csv')
    targets = df[['label_Basal', 'label_LumA', 'label_LumB']].values.astype(np.float64)


    preds = np.clip(targets + rng.normal(0, 0.08, targets.shape), 0, 1)

    print('=== compute_all_metrics (single fold mock) ===')
    result = compute_all_metrics(preds, targets)
    for k, v in result.items():
        print(f'  {k:<35} {v:.4f}')

    print('\n=== K-fold summary mock (3 folds) ===')
    ckc = init_ckc_metric()
    for _ in range(3):
        noise = rng.normal(0, 0.05, targets.shape)
        p = np.clip(targets + noise, 0, 1)
        fold_res = compute_all_metrics(p, targets)
        append_fold_metrics(ckc, fold_res)

        ckc['p_name'].append([f'slide_{i}' for i in range(N)])
        ckc['bag_l'].append(targets)
        ckc['bag_pre'].append(p)
        ckc['pred_cls'].append(np.argmax(p, axis=1).tolist())

    print_cv_summary(ckc)
