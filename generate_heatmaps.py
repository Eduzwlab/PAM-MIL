
import os
import glob
import argparse
import h5py
import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from PIL import Image
from scipy.ndimage import gaussian_filter, zoom
from scipy.stats import rankdata
import cv2
import openslide



plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "figure.titlesize": 12.0,
    "legend.fontsize": 8.6,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "axes.titleweight": "regular",
    "axes.labelweight": "regular",
    "font.weight": "regular",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

WSI_EXTS = ('.ndpi', '.svs', '.tiff', '.tif', '.mrxs', '.scn', '.vms', '.vmu', '.bif', '.czi')
SUBTYPES = ["Basal", "LumA", "LumB"]
SINGLE_CLASS_STYLE = "jet"

MULTI_CLASS_COLORS = {
    "Basal": np.array([242, 201, 76], dtype=np.float32),
    "LumA":  np.array([0, 168, 143], dtype=np.float32),
    "LumB":  np.array([47, 91, 234], dtype=np.float32),
}

LOW_CONF_TISSUE_COLOR = np.array([233, 236, 240], dtype=np.float32)
BACKGROUND_COLOR = np.array([255, 255, 255], dtype=np.float32)


def find_wsi(slides_dir, slide_id):
    for ext in WSI_EXTS:
        p = os.path.join(slides_dir, slide_id + ext)
        if os.path.isfile(p):
            return p

    for f in glob.glob(os.path.join(slides_dir, "**", "*"), recursive=True):
        if os.path.splitext(os.path.basename(f))[0] == slide_id and os.path.splitext(f)[1].lower() in WSI_EXTS:
            return f
    return None


def find_pt(data_root, slide_id):
    for sub in ("pt_files", "pt"):
        p = os.path.join(data_root, sub, slide_id + ".pt")
        if os.path.isfile(p):
            return p

    for f in glob.glob(os.path.join(data_root, "**", slide_id + ".pt"), recursive=True):
        return f
    return None


def find_h5(data_root, slide_id):
    for sub in ("h5_files", "h5"):
        p = os.path.join(data_root, sub, slide_id + ".h5")
        if os.path.isfile(p):
            return p

    for f in glob.glob(os.path.join(data_root, "**", slide_id + ".h5"), recursive=True):
        return f
    return None


def load_model(model_root, in_dim, device, mamba_type="spatial_mamba", fold=0):
    ckpt_path = os.path.join(model_root, f'fold_{fold}_model_best.pt')

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件: {ckpt_path}")

    print(f"[INFO] 加载模型: {ckpt_path}")

    from modules.PAM_MIL import PAM_MIL

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        model = PAM_MIL(in_dim=in_dim, type=mamba_type)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if len(missing) > 0:
            print(f"[WARN] missing keys: {len(missing)}")
        if len(unexpected) > 0:
            print(f"[WARN] unexpected keys: {len(unexpected)}")
    else:
        model = ckpt

    model.to(device).eval()

    if hasattr(model, "preproc"):
        model.preproc.nmax = 0
        model.preproc.enable_train_sampling = False
        model.preproc.enable_eval_mc = False
        print("[INFO] SlidePreprocessor 已禁用采样")

    return model


def load_features_and_coords(data_root, slide_id, device):
    pt_path = find_pt(data_root, slide_id)
    h5_path = find_h5(data_root, slide_id)

    if pt_path is None:
        raise FileNotFoundError(f"找不到 pt: {slide_id}")
    if h5_path is None:
        raise FileNotFoundError(f"找不到 h5: {slide_id}")

    feat = torch.load(pt_path, map_location="cpu")
    if isinstance(feat, dict):
        feat = feat.get("features", list(feat.values())[0])

    if not isinstance(feat, torch.Tensor):
        feat = torch.tensor(feat, dtype=torch.float32)

    feat = feat.float()

    with h5py.File(h5_path, "r") as f:
        coords = np.array(f["coords"])

    assert len(feat) == len(coords), f"N不匹配: features={len(feat)} vs coords={len(coords)}"
    return feat.to(device), coords

def to_percentiles_01(scores):
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(scores) == 0:
        return scores
    if len(scores) == 1:
        return np.ones_like(scores, dtype=np.float32)

    ranks = rankdata(scores, method="average").astype(np.float32)
    return (ranks - 1.0) / max(len(scores) - 1.0, 1.0)


def resize_float_map(arr, out_w, out_h):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] == out_h and arr.shape[1] == out_w:
        return arr.astype(np.float32)

    if HAS_CV2:
        return cv2.resize(arr.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    zy = out_h / max(arr.shape[0], 1)
    zx = out_w / max(arr.shape[1], 1)
    return zoom(arr.astype(np.float32), (zy, zx), order=1)


def make_tissue_mask_from_thumbnail(thumbnail):
    if thumbnail is None:
        return None

    img = np.array(thumbnail)
    if img.ndim != 3:
        return None

    if HAS_CV2:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        _, s, v = cv2.split(hsv)

        mask = ((s > 18) & (v < 245)) | (img.mean(axis=2) < 235)
        mask = mask.astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        min_area = max(16, int(mask.shape[0] * mask.shape[1] * 0.00002))
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean[labels == i] = 255

        return (clean > 0).astype(np.float32)

    gray = img.mean(axis=2)
    mask = (gray < 235).astype(np.float32)
    mask = gaussian_filter(mask, sigma=1.0)
    return (mask > 0.2).astype(np.float32)


def bilinear_splat(num_map, den_map, xs, ys, vals):
    H, W = num_map.shape

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    dx = (xs - x0).astype(np.float32)
    dy = (ys - y0).astype(np.float32)

    corners = [
        (x0, y0, (1.0 - dx) * (1.0 - dy)),
        (x1, y0, dx * (1.0 - dy)),
        (x0, y1, (1.0 - dx) * dy),
        (x1, y1, dx * dy),
    ]

    vals = vals.astype(np.float32)

    for xx, yy, ww in corners:
        m = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H) & (ww > 1e-8)
        if np.any(m):
            np.add.at(num_map, (yy[m], xx[m]), vals[m] * ww[m])
            np.add.at(den_map, (yy[m], xx[m]), ww[m])


def rasterize_rect_support_ul(x0, y0, patch_w, patch_h, W, H):
    diff = np.zeros((H + 1, W + 1), dtype=np.float32)

    x1 = np.floor(x0).astype(np.int32)
    y1 = np.floor(y0).astype(np.int32)
    x2 = np.ceil(x0 + patch_w).astype(np.int32)
    y2 = np.ceil(y0 + patch_h).astype(np.int32)

    x1 = np.clip(x1, 0, W)
    x2 = np.clip(x2, 0, W)
    y1 = np.clip(y1, 0, H)
    y2 = np.clip(y2, 0, H)

    m = (x2 > x1) & (y2 > y1)
    if np.any(m):
        np.add.at(diff, (y1[m], x1[m]), 1.0)
        np.add.at(diff, (y2[m], x1[m]), -1.0)
        np.add.at(diff, (y1[m], x2[m]), -1.0)
        np.add.at(diff, (y2[m], x2[m]), 1.0)

    support = diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    return support.astype(np.float32)


def rasterize_rect_sum_count_ul(x0, y0, patch_w, patch_h, vals, W, H):
    sum_diff = np.zeros((H + 1, W + 1), dtype=np.float32)
    cnt_diff = np.zeros((H + 1, W + 1), dtype=np.float32)

    x1 = np.floor(x0).astype(np.int32)
    y1 = np.floor(y0).astype(np.int32)
    x2 = np.ceil(x0 + patch_w).astype(np.int32)
    y2 = np.ceil(y0 + patch_h).astype(np.int32)

    x1 = np.clip(x1, 0, W)
    x2 = np.clip(x2, 0, W)
    y1 = np.clip(y1, 0, H)
    y2 = np.clip(y2, 0, H)

    vals = np.asarray(vals, dtype=np.float32).reshape(-1)
    m = (x2 > x1) & (y2 > y1)

    if np.any(m):
        np.add.at(sum_diff, (y1[m], x1[m]), vals[m])
        np.add.at(sum_diff, (y2[m], x1[m]), -vals[m])
        np.add.at(sum_diff, (y1[m], x2[m]), -vals[m])
        np.add.at(sum_diff, (y2[m], x2[m]), vals[m])

        np.add.at(cnt_diff, (y1[m], x1[m]), 1.0)
        np.add.at(cnt_diff, (y2[m], x1[m]), -1.0)
        np.add.at(cnt_diff, (y1[m], x2[m]), -1.0)
        np.add.at(cnt_diff, (y2[m], x2[m]), 1.0)

    sum_map = sum_diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    cnt_map = cnt_diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1]
    return sum_map.astype(np.float32), cnt_map.astype(np.float32)


def pil_to_np(img):
    return np.array(img.convert("RGB"))


def fmt_score(x, ndigits=3):
    if x is None:
        return "NA"
    try:
        x = float(x)
        if np.isnan(x):
            return "NA"
        return f"{x:.{ndigits}f}"
    except Exception:
        return "NA"


def safe_to_float(x, default=np.nan):
    try:
        x = float(x)
        return x
    except Exception:
        return default


def safe_get_gt(gt_dict, key):
    if gt_dict is None:
        return None
    v = gt_dict.get(key, None)
    if v is None:
        return None
    try:
        v = float(v)
        if np.isnan(v):
            return None
        return v
    except Exception:
        return None


def short_slide_name(slide_id, n=16):
    return str(slide_id)[:n]


def infer_discrete_label_from_scores(score_dict):
    vals = [safe_to_float(score_dict.get(s, np.nan)) for s in SUBTYPES]
    if any([np.isnan(v) for v in vals]):
        return None
    return SUBTYPES[int(np.argmax(vals))]


def tensor_to_numpy_1d(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.float32).reshape(-1)

    return x


def format_patch_position_from_coord(coord):
    x = int(round(float(coord[0])))
    y = int(round(float(coord[1])))
    return f"_{x}; {y}_"


def save_patch_attention_scores_csv(
        slide_id,
        coords,
        attn_dict,
        save_dir,
        float_format="%.10g",
):
    coords = np.asarray(coords)

    n_patches = coords.shape[0]

    attn_scores = {}
    for s in SUBTYPES:

        arr = tensor_to_numpy_1d(attn_dict[s], name=f"{s}_attention")


        attn_scores[s] = arr.astype(np.float32)

    attn_mat = np.stack(
        [attn_scores["Basal"], attn_scores["LumA"], attn_scores["LumB"]],
        axis=1
    )

    attn_mat_for_argmax = np.where(np.isnan(attn_mat), -np.inf, attn_mat)
    pred_idx = np.argmax(attn_mat_for_argmax, axis=1)

    all_invalid = ~np.isfinite(attn_mat_for_argmax).any(axis=1)
    patch_pred_subtypes = []
    for i, invalid in zip(pred_idx, all_invalid):
        if invalid:
            patch_pred_subtypes.append("NA")
        else:
            patch_pred_subtypes.append(SUBTYPES[int(i)])

    patch_positions = [
        format_patch_position_from_coord(c) for c in coords[:, :2]
    ]

    df_patch = pd.DataFrame({
        "patch_position": patch_positions,
        "Basal_attention": attn_scores["Basal"],
        "LumA_attention": attn_scores["LumA"],
        "LumB_attention": attn_scores["LumB"],
        "pred_subtype": patch_pred_subtypes,
    })

    os.makedirs(save_dir, exist_ok=True)

    out_csv = os.path.join(
        save_dir,
        f"{slide_id}_patch_attention_scores.csv"
    )

    df_patch.to_csv(out_csv, index=False, float_format=float_format)

    print(f"  [PATCH CSV] saved: {out_csv}")
    print(f"  [PATCH CSV] n_patches={len(df_patch)}")

    return out_csv


def add_vertical_colorbar(
        fig,
        ax,
        style="jet",
        width='3.0%',
        height='36%',
        loc='center right',
        bbox_to_anchor=(0.0, 0.0, 1.0, 1.0),
        fontsize=7.2,
        show_ticks=True,
        tick_values=(0.0, 0.5, 1.0),
):
    acb = inset_axes(
        ax,
        width=width,
        height=height,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    sm = ScalarMappable(cmap=plt.get_cmap(style), norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=acb, orientation='vertical')

    if show_ticks:
        cb.set_ticks(list(tick_values))
        cb.set_ticklabels([f"{t:g}" for t in tick_values])
        cb.ax.tick_params(labelsize=fontsize, length=1.8, pad=1.2)
    else:
        cb.set_ticks([])
        cb.ax.tick_params(length=0)

    cb.outline.set_linewidth(0.55)
    cb.outline.set_edgecolor('#c9c9c9')
    return cb


def style_legend(legend_obj):
    if legend_obj is None:
        return
    frame = legend_obj.get_frame()
    frame.set_facecolor((1, 1, 1, 0.97))
    frame.set_edgecolor("#cccccc")
    frame.set_linewidth(0.65)


def build_multiclass_semantic_map_image(
        thumbnail,
        raw_maps,
        tissue_mask=None,
        min_strength=0.12,
        min_margin=0.025,
        gamma=0.90,
        support_thresh=0.015,
        class_colors=None,
        low_conf_color=None,
        background_color=None,
):

    if class_colors is None:
        class_colors = MULTI_CLASS_COLORS
    if low_conf_color is None:
        low_conf_color = LOW_CONF_TISSUE_COLOR
    if background_color is None:
        background_color = BACKGROUND_COLOR

    thumb_np = np.array(thumbnail.convert("RGB"), dtype=np.float32)
    H, W, _ = thumb_np.shape

    value_list = []
    alpha_list = []
    strength_list = []

    for s in SUBTYPES:
        value = np.asarray(raw_maps[s]["value"], dtype=np.float32)
        alpha = np.asarray(raw_maps[s]["alpha"], dtype=np.float32)

        value = np.clip(value, 0.0, 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)

        strength = np.clip(value * alpha, 0.0, 1.0)
        strength = np.power(strength, gamma)

        value_list.append(value)
        alpha_list.append(alpha)
        strength_list.append(strength)

    value_stack = np.stack(value_list, axis=-1)      # (H, W, 3)
    alpha_stack = np.stack(alpha_list, axis=-1)      # (H, W, 3)
    strength_stack = np.stack(strength_list, axis=-1)

    winner_idx = np.argmax(strength_stack, axis=-1)
    top1 = np.max(strength_stack, axis=-1)
    top2 = np.partition(strength_stack, -2, axis=-1)[..., -2]
    margin = top1 - top2

    support_mask = np.max(alpha_stack, axis=-1) > support_thresh

    if tissue_mask is not None:
        tissue_mask = np.asarray(tissue_mask, dtype=np.float32)
        support_mask = support_mask & (tissue_mask > 0.08)

    confident_mask = support_mask & (top1 >= min_strength) & (margin >= min_margin)
    low_conf_mask = support_mask & (~confident_mask)

    out = np.zeros((H, W, 3), dtype=np.float32)
    out[:] = background_color[None, None, :]

    out[low_conf_mask] = low_conf_color

    for i, s in enumerate(SUBTYPES):
        out[confident_mask & (winner_idx == i)] = class_colors[s]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out), winner_idx, top1, confident_mask, low_conf_mask

class _PatchBasedHeatmapGeneratorBase(object):
    THUMBNAIL_SIZE_SCALE_UPPER_LIMIT = 0.5
    THUMBNAIL_SIZE_SCALE_LOWER_LIMIT = 0.01
    THUMBNAIL_MAX_SIZE = 4096
    THUMBNAIL_MIN_SIZE = 256

    AVAILABLE_HEATMAP_STYLE = ('coolwarm', 'hot', 'bwr', 'Spectral', 'seismic', 'jet', 'viridis', 'turbo', 'cividis')
    AVAILABLE_NORMALIZE_METHOD = ('close', 'sigmod', 'sigmoid', 'rank')

    def __init__(self, slide_path, patch_level, coordinates, scores, patch_size, slide_obj=None):

        self._slide = slide_obj if slide_obj is not None else openslide.open_slide(slide_path)
        self._owns_slide = slide_obj is None
        self._slide_path = slide_path
        self._patch_level = int(patch_level)
        self._coordinates = np.asarray(coordinates, dtype=np.float64)
        self._scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        self._patch_size = patch_size if isinstance(patch_size, (tuple, list)) else (patch_size, patch_size)

        assert self._coordinates.ndim == 2 and self._coordinates.shape[1] >= 2, "coordinates must be [N,2]"
        assert len(self._coordinates) == len(self._scores), "scores 和 coordinates discordance"

    def __del__(self):
        try:
            if getattr(self, "_owns_slide", False) and getattr(self, "_slide", None) is not None:
                self._slide.close()
        except Exception:
            pass

    def set_scores(self, scores):
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        assert len(scores) == len(
            self._coordinates), f"set_scores Fail：scores={len(scores)} 与 coords={len(self._coordinates)} discordance"
        self._scores = scores

    def _normalize_scores(self, normalize_method):
        scores = self._scores.astype(np.float32).copy()

        if normalize_method in ("sigmod", "sigmoid"):
            scores = to_percentiles_01(scores)
        elif normalize_method == "rank":
            scores = rankdata(scores, method="average").astype(np.float32)
            scores = (scores - 1.0) / max(len(scores) - 1.0, 1.0)
        elif normalize_method == "close":
            smin, smax = float(np.min(scores)), float(np.max(scores))
            if smax - smin > 1e-8:
                scores = (scores - smin) / (smax - smin)
            else:
                scores = np.zeros_like(scores, dtype=np.float32)
        else:
            raise ValueError(f"normalize_method={normalize_method} no support")

        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def _compute_thumbnail_size(self, thumbnail_size_scale):
        slide_level_count = self._slide.level_count
        assert 0 <= self._patch_level < slide_level_count

        slide_size = self._slide.level_dimensions[self._patch_level]
        W0, H0 = map(int, slide_size)

        assert len(thumbnail_size_scale) == 2
        sx_req, sy_req = thumbnail_size_scale

        def _solve_one_dim(dim, req_scale):
            req_scale = float(np.clip(
                req_scale,
                self.THUMBNAIL_SIZE_SCALE_LOWER_LIMIT,
                self.THUMBNAIL_SIZE_SCALE_UPPER_LIMIT,
            ))

            feasible_max = max(1, int(round(dim * self.THUMBNAIL_SIZE_SCALE_UPPER_LIMIT)))
            target = max(1, int(round(dim * req_scale)))

            effective_min = min(self.THUMBNAIL_MIN_SIZE, feasible_max)
            effective_max = min(self.THUMBNAIL_MAX_SIZE, feasible_max)

            if effective_min > effective_max:
                effective_min = effective_max

            out_dim = int(np.clip(target, effective_min, effective_max))
            eff_scale = out_dim / float(dim)
            return eff_scale, out_dim

        sx, width = _solve_one_dim(W0, sx_req)
        sy, height = _solve_one_dim(H0, sy_req)

        print(
            f"[THUMB] level_size=({W0}, {H0}) "
            f"req_scale=({sx_req:.4f}, {sy_req:.4f}) "
            f"used_scale=({sx:.4f}, {sy:.4f}) "
            f"thumb_size=({width}, {height})"
        )

        return slide_size, (sx, sy), (width, height)

    def prepare_context(self, thumbnail_size_scale=(0.125, 0.125), use_tissue_mask=True):
        slide_size, used_scale, thumb_size = self._compute_thumbnail_size(thumbnail_size_scale)

        thumbnail = self._slide.get_thumbnail(thumb_size).convert("RGB")
        thumb_w, thumb_h = thumbnail.size

        sx_eff = thumb_w / float(slide_size[0])
        sy_eff = thumb_h / float(slide_size[1])

        tissue_mask = make_tissue_mask_from_thumbnail(thumbnail) if use_tissue_mask else None

        print(
            f"[CTX] patch_level={self._patch_level} "
            f"slide_size={slide_size} thumb_size=({thumb_w},{thumb_h}) "
            f"sx_eff={sx_eff:.6f} sy_eff={sy_eff:.6f} "
            f"use_tissue_mask={use_tissue_mask}"
        )

        return {
            "slide_size": slide_size,
            "used_scale": used_scale,
            "thumb_size": (thumb_w, thumb_h),
            "thumbnail": thumbnail,
            "sx_eff": sx_eff,
            "sy_eff": sy_eff,
            "tissue_mask": tissue_mask,
        }

    def _compose_heatmap_image(self, thumbnail, value, alpha_map, style="jet", alpha=0.5):
        assert style in self.AVAILABLE_HEATMAP_STYLE, f"style={style} no support"
        assert 0.0 <= alpha <= 1.0, "alpha must be between [0~1] "

        color_map = plt.get_cmap(style)
        color = (color_map(value)[..., :3] * 255).astype(np.uint8)

        thumbnail_np = np.array(thumbnail, dtype=np.float32)
        color_np = color.astype(np.float32)

        blend_alpha = np.clip(alpha * alpha_map, 0.0, 1.0)[..., None]
        heatmap_np = thumbnail_np * (1.0 - blend_alpha) + color_np * blend_alpha
        heatmap_np = np.clip(heatmap_np, 0, 255).astype(np.uint8)
        return Image.fromarray(heatmap_np)

    def generate_heatmap(self, *args, **kwargs):
        raise NotImplementedError

class PatchBasedHeatmapGeneratorBlock(_PatchBasedHeatmapGeneratorBase):
    def _build_block_map(self, thumb_w, thumb_h, sx_eff, sy_eff, scores, tissue_mask=None):
        patch_w_thumb = max(1.0, float(self._patch_size[0]) * sx_eff)
        patch_h_thumb = max(1.0, float(self._patch_size[1]) * sy_eff)

        x0 = self._coordinates[:, 0] * sx_eff
        y0 = self._coordinates[:, 1] * sy_eff

        score_sum_map, count_map = rasterize_rect_sum_count_ul(
            x0=x0,
            y0=y0,
            patch_w=patch_w_thumb,
            patch_h=patch_h_thumb,
            vals=scores,
            W=thumb_w,
            H=thumb_h,
        )

        value = np.zeros_like(score_sum_map, dtype=np.float32)
        valid = count_map > 0
        value[valid] = score_sum_map[valid] / np.maximum(count_map[valid], 1e-6)

        alpha_map = valid.astype(np.float32)
        if tissue_mask is not None:
            alpha_map *= tissue_mask.astype(np.float32)
            value[alpha_map <= 0] = 0.0

        coverage = float((alpha_map > 0).sum()) / float(alpha_map.size) * 100.0
        print(
            f"[BLOCK] patch_thumb={patch_w_thumb:.2f}x{patch_h_thumb:.2f} "
            f"coverage={coverage:.1f}% count_max={float(count_map.max()):.1f}"
        )

        return value.astype(np.float32), alpha_map.astype(np.float32), count_map.astype(np.float32)

    def generate_heatmap(
            self,
            thumbnail_size_scale=(0.125, 0.125),
            style=SINGLE_CLASS_STYLE,
            alpha=0.5,
            normalize_method="sigmod",
            use_tissue_mask=True,
            shared_ctx=None,
            return_raw=False,
    ):
        assert normalize_method in self.AVAILABLE_NORMALIZE_METHOD, f"normalize_method={normalize_method} 不支持"

        if shared_ctx is None:
            shared_ctx = self.prepare_context(thumbnail_size_scale=thumbnail_size_scale,
                                              use_tissue_mask=use_tissue_mask)

        thumbnail = shared_ctx["thumbnail"]
        thumb_w, thumb_h = shared_ctx["thumb_size"]
        sx_eff = shared_ctx["sx_eff"]
        sy_eff = shared_ctx["sy_eff"]
        tissue_mask = shared_ctx["tissue_mask"]

        scores = self._normalize_scores(normalize_method)

        value, alpha_map, count_map = self._build_block_map(
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            sx_eff=sx_eff,
            sy_eff=sy_eff,
            scores=scores,
            tissue_mask=tissue_mask,
        )

        heatmap_image = self._compose_heatmap_image(
            thumbnail=thumbnail,
            value=value,
            alpha_map=alpha_map,
            style=style,
            alpha=alpha,
        )

        if return_raw:
            return thumbnail, heatmap_image, {"value": value, "alpha": alpha_map, "count": count_map}
        return thumbnail, heatmap_image


class PatchBasedHeatmapGeneratorRegion(_PatchBasedHeatmapGeneratorBase):
    def _build_region_map(
            self,
            thumb_w,
            thumb_h,
            sx_eff,
            sy_eff,
            scores,
            tissue_mask=None,
            upsample=3.0,
            sigma_ratio=0.55,
            alpha_sigma_ratio=0.85,
            contrast_percentile=(2, 98),
            max_latent_side=7000,
    ):
        coords = self._coordinates.astype(np.float64)

        patch_w_thumb = max(float(self._patch_size[0]) * sx_eff, 1.5)
        patch_h_thumb = max(float(self._patch_size[1]) * sy_eff, 1.5)

        upsample_eff = float(upsample)
        if thumb_w * upsample_eff > max_latent_side:
            upsample_eff = max_latent_side / float(max(thumb_w, 1))
        if thumb_h * upsample_eff > max_latent_side:
            upsample_eff = min(upsample_eff, max_latent_side / float(max(thumb_h, 1)))
        upsample_eff = max(1.0, upsample_eff)

        hi_w = int(max(round(thumb_w * upsample_eff), 8))
        hi_h = int(max(round(thumb_h * upsample_eff), 8))

        patch_w_hi = patch_w_thumb * upsample_eff
        patch_h_hi = patch_h_thumb * upsample_eff

        x0_hi = coords[:, 0] * sx_eff * upsample_eff
        y0_hi = coords[:, 1] * sy_eff * upsample_eff
        cx_hi = x0_hi + patch_w_hi * 0.5
        cy_hi = y0_hi + patch_h_hi * 0.5

        num_map = np.zeros((hi_h, hi_w), dtype=np.float32)
        den_map = np.zeros((hi_h, hi_w), dtype=np.float32)
        bilinear_splat(num_map, den_map, cx_hi, cy_hi, scores)

        sigma_x = max(patch_w_hi * sigma_ratio, 1.2)
        sigma_y = max(patch_h_hi * sigma_ratio, 1.2)

        num_blur = gaussian_filter(num_map, sigma=(sigma_y, sigma_x), mode="nearest")
        den_blur = gaussian_filter(den_map, sigma=(sigma_y, sigma_x), mode="nearest")

        with np.errstate(invalid="ignore", divide="ignore"):
            value_hi = np.where(den_blur > 1e-6, num_blur / den_blur, 0.0).astype(np.float32)

        value_hi = gaussian_filter(
            value_hi,
            sigma=(max(sigma_y * 0.20, 0.8), max(sigma_x * 0.20, 0.8)),
            mode="nearest",
        )

        support = rasterize_rect_support_ul(
            x0=x0_hi,
            y0=y0_hi,
            patch_w=patch_w_hi,
            patch_h=patch_h_hi,
            W=hi_w,
            H=hi_h,
        )
        support_bin = (support > 0).astype(np.float32)

        alpha_hi = gaussian_filter(
            support_bin,
            sigma=(max(patch_h_hi * alpha_sigma_ratio, 1.0), max(patch_w_hi * alpha_sigma_ratio, 1.0)),
            mode="nearest",
        )

        if tissue_mask is not None:
            tissue_mask_hi = resize_float_map(tissue_mask.astype(np.float32), hi_w, hi_h)
            alpha_hi *= tissue_mask_hi

        valid = alpha_hi > 0.05
        if np.any(valid):
            vv = value_hi[valid]
            lo = np.percentile(vv, contrast_percentile[0])
            hi = np.percentile(vv, contrast_percentile[1])
            value_hi = np.clip((value_hi - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            value_hi = np.power(value_hi, 0.95)

        pos = alpha_hi > 0
        if np.any(pos):
            ref = np.percentile(alpha_hi[pos], 99.5)
            alpha_hi = np.clip(alpha_hi / max(ref, 1e-6), 0.0, 1.0)

        alpha_hi[alpha_hi < 0.01] = 0.0
        value_hi[alpha_hi <= 0] = 0.0

        value = resize_float_map(value_hi, thumb_w, thumb_h)
        alpha_map = resize_float_map(alpha_hi, thumb_w, thumb_h)

        if np.max(alpha_map) > 1e-6:
            alpha_map = alpha_map / np.max(alpha_map)

        alpha_map = np.clip(alpha_map, 0.0, 1.0)
        alpha_map[alpha_map < 0.015] = 0.0
        value[alpha_map <= 0] = 0.0

        coverage = float((alpha_map > 0.02).sum()) / float(alpha_map.size) * 100.0
        print(
            f"[REGION] latent={hi_w}x{hi_h} upsample={upsample_eff:.2f} "
            f"patch_thumb={patch_w_thumb:.2f}x{patch_h_thumb:.2f} "
            f"sigma=({sigma_x:.2f},{sigma_y:.2f}) coverage={coverage:.1f}%"
        )

        return value.astype(np.float32), alpha_map.astype(np.float32)

    def generate_heatmap(
            self,
            thumbnail_size_scale=(0.125, 0.125),
            style=SINGLE_CLASS_STYLE,
            alpha=0.5,
            normalize_method="sigmod",
            use_tissue_mask=True,
            shared_ctx=None,
            return_raw=False,
            upsample=3.0,
            sigma_ratio=0.55,
            alpha_sigma_ratio=0.85,
            contrast_percentile=(2, 98),
            max_latent_side=7000,
    ):
        assert normalize_method in self.AVAILABLE_NORMALIZE_METHOD, f"normalize_method={normalize_method} no support"

        if shared_ctx is None:
            shared_ctx = self.prepare_context(thumbnail_size_scale=thumbnail_size_scale,
                                              use_tissue_mask=use_tissue_mask)

        thumbnail = shared_ctx["thumbnail"]
        thumb_w, thumb_h = shared_ctx["thumb_size"]
        sx_eff = shared_ctx["sx_eff"]
        sy_eff = shared_ctx["sy_eff"]
        tissue_mask = shared_ctx["tissue_mask"]

        scores = self._normalize_scores(normalize_method)

        value, alpha_map = self._build_region_map(
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            sx_eff=sx_eff,
            sy_eff=sy_eff,
            scores=scores,
            tissue_mask=tissue_mask,
            upsample=upsample,
            sigma_ratio=sigma_ratio,
            alpha_sigma_ratio=alpha_sigma_ratio,
            contrast_percentile=contrast_percentile,
            max_latent_side=max_latent_side,
        )

        heatmap_image = self._compose_heatmap_image(
            thumbnail=thumbnail,
            value=value,
            alpha_map=np.power(alpha_map, 0.90),
            style=style,
            alpha=alpha,
        )

        if return_raw:
            return thumbnail, heatmap_image, {"value": value, "alpha": alpha_map}
        return thumbnail, heatmap_image


def save_single_subtype_figure(
        slide_id,
        subtype,
        image,
        pred,
        gt,
        output_path,
        style=SINGLE_CLASS_STYLE,
        dpi=240,
):
    slide_short = short_slide_name(slide_id, n=18)
    w, h = image.size
    ratio = h / max(w, 1)

    fig_w = 7.1
    fig_h = fig_w * ratio + 0.75

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.028, right=0.905, top=0.905, bottom=0.028)
    ax = fig.add_subplot(gs[0, 0])

    ax.imshow(pil_to_np(image))
    ax.axis("off")

    fig.text(
        0.5, 0.965,
        slide_short,
        ha="center", va="top",
        fontsize=11.4, fontweight="regular", color="#1f1f1f"
    )
    fig.text(
        0.5, 0.938,
        f"{subtype} attention map  |  GT = {fmt_score(gt)}  |  Pred = {fmt_score(pred)}",
        ha="center", va="top",
        fontsize=9.8, fontweight="regular", color="#3a3a3a"
    )

    add_vertical_colorbar(
        fig,
        ax,
        style=style,
        width='3.0%',
        height='36%',
        loc='center right',
        bbox_to_anchor=(0.012, 0.0, 1.0, 1.0),
        show_ticks=True,
        fontsize=7.3,
        tick_values=(0.0, 0.5, 1.0),
    )

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.045)
    plt.close(fig)


def save_multiclass_figure(
        slide_id,
        image,
        preds_dict,
        gt_dict,
        output_path,
        gt_subtype=None,
        dpi=240,
        class_colors=None,
):
    if class_colors is None:
        class_colors = MULTI_CLASS_COLORS

    slide_short = short_slide_name(slide_id, n=18)
    pred_subtype = infer_discrete_label_from_scores(preds_dict)

    w, h = image.size
    ratio = h / max(w, 1)

    fig_w = 7.25
    fig_h = fig_w * ratio + 0.78

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = gridspec.GridSpec(1, 1, figure=fig, left=0.026, right=0.985, top=0.905, bottom=0.028)
    ax = fig.add_subplot(gs[0, 0])

    ax.imshow(pil_to_np(image))
    ax.axis("off")

    fig.text(
        0.5, 0.965,
        slide_short,
        ha="center", va="top",
        fontsize=11.4, fontweight="regular", color="#1f1f1f"
    )
    fig.text(
        0.5, 0.938,
        f"Subtype map  |  GT: {gt_subtype if gt_subtype else 'NA'}  |  Pred: {pred_subtype if pred_subtype else 'NA'}",
        ha="center", va="top",
        fontsize=9.8, fontweight="regular", color="#3a3a3a"
    )

    handles = [
        Patch(facecolor=class_colors[s] / 255.0, edgecolor="white", linewidth=0.7, label=s)
        for s in SUBTYPES
    ]
    handles.append(
        Patch(facecolor=LOW_CONF_TISSUE_COLOR / 255.0, edgecolor="white", linewidth=0.7, label="Low confidence")
    )

    leg = ax.legend(
        handles=handles,
        loc="lower right",
        fontsize=8.5,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        borderpad=0.55,
        handlelength=1.3,
        handletextpad=0.55,
        labelspacing=0.42,
    )
    style_legend(leg)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.045)
    plt.close(fig)


def save_panel4_figure(
        slide_id,
        output_path,
        basal_img,
        luma_img,
        lumb_img,
        multiclass_img,
        gt_dict,
        preds_dict,
        gt_subtype=None,
        style=SINGLE_CLASS_STYLE,
        dpi=260,
        class_colors=None,
):
    if class_colors is None:
        class_colors = MULTI_CLASS_COLORS

    slide_short = short_slide_name(slide_id, n=18)
    pred_subtype = infer_discrete_label_from_scores(preds_dict)

    imgs = [basal_img, luma_img, lumb_img, multiclass_img]
    titles = [
        f"Basal\nGT={fmt_score(safe_get_gt(gt_dict, 'Basal'))} | Pred={fmt_score(preds_dict.get('Basal'))}",
        f"LumA\nGT={fmt_score(safe_get_gt(gt_dict, 'LumA'))} | Pred={fmt_score(preds_dict.get('LumA'))}",
        f"LumB\nGT={fmt_score(safe_get_gt(gt_dict, 'LumB'))} | Pred={fmt_score(preds_dict.get('LumB'))}",
        f"Subtype map\nGT: {gt_subtype if gt_subtype else 'NA'} | Pred: {pred_subtype if pred_subtype else 'NA'}",
    ]
    panel_letters = ["A", "B", "C", "D"]

    w, h = imgs[0].size
    ratio = h / max(w, 1)

    fig = plt.figure(figsize=(16.8, 4.25 * ratio + 1.25), facecolor="white")
    gs = gridspec.GridSpec(1, 4, figure=fig, left=0.012, right=0.988, top=0.84, bottom=0.055, wspace=0.030)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

    for i, (ax, img, title, letter) in enumerate(zip(axes, imgs, titles, panel_letters)):
        ax.imshow(pil_to_np(img))
        ax.axis("off")
        ax.set_title(title, fontsize=9.6, pad=8.5, color="#2c2c2c", fontweight="regular")
        ax.text(
            0.018, 0.982, letter,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10.4, color="#111111", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#d5d5d5", linewidth=0.55, alpha=0.96)
        )

    fig.text(
        0.5, 0.955,
        slide_short,
        ha="center", va="top",
        fontsize=12.0, fontweight="regular", color="#1a1a1a"
    )

    for ax in axes[:3]:
        add_vertical_colorbar(
            fig,
            ax,
            style=style,
            width='3.0%',
            height='24%',
            loc='lower right',
            bbox_to_anchor=(-0.008, 0.030, 1.0, 1.0),
            show_ticks=True,
            fontsize=6.7,
            tick_values=(0.0, 0.5, 1.0),
        )

    handles = [
        Patch(facecolor=class_colors[s] / 255.0, edgecolor="white", linewidth=0.7, label=s)
        for s in SUBTYPES
    ]
    handles.append(
        Patch(facecolor=LOW_CONF_TISSUE_COLOR / 255.0, edgecolor="white", linewidth=0.7, label="Low confidence")
    )

    leg = axes[3].legend(
        handles=handles,
        loc="lower right",
        fontsize=8.2,
        frameon=True,
        fancybox=True,
        framealpha=0.97,
        borderpad=0.48,
        handlelength=1.2,
        handletextpad=0.50,
        labelspacing=0.38,
    )
    style_legend(leg)

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.045)
    plt.close(fig)


def process_slide(
        slide_id,
        data_root,
        slides_dir,
        model,
        row,
        output_dir,
        patch_size,
        patch_level,
        device,
        thumb_scale=(0.125, 0.125),
        style="cividis",
        alpha=0.5,
        normalize_method="sigmod",
        heatmap_mode="region",
        upsample=3.0,
        sigma_ratio=0.55,
        alpha_sigma_ratio=0.85,
        dpi=240,
        use_tissue_mask=True,
        save_patch_csv=True,
):
    print(f"\n{'=' * 60}\n[SLIDE] {slide_id}")

    features, coords = load_features_and_coords(data_root, slide_id, device)
    print(f"  patches={len(features)} feat_dim={features.shape[1]}")

    with torch.no_grad():
        if not hasattr(model, "get_attention_weights"):
            raise AttributeError("model 缺少 get_attention_weights(features) 接口")
        preds_1d, attn_dict = model.get_attention_weights(features)

    preds_np = np.asarray(preds_1d.detach().cpu().numpy()).reshape(-1)
    preds_dict = {s: float(preds_np[i]) for i, s in enumerate(SUBTYPES)}
    print(f"  preds → {preds_dict}")

    gt = {s: np.nan for s in SUBTYPES}
    gt_sub = None
    if row is not None:
        for s in SUBTYPES:
            gt[s] = safe_to_float(row.get(f"label_{s}", np.nan), default=np.nan)
        gt_sub = str(row.get("final_subtype", "")).strip()
        if gt_sub.lower() in ("", "nan", "none"):
            gt_sub = None

    pred_sub = infer_discrete_label_from_scores(preds_dict)

    wsi_path = find_wsi(slides_dir, slide_id)

    save_dir = os.path.join(output_dir, slide_id)
    os.makedirs(save_dir, exist_ok=True)

    if save_patch_csv:
        save_patch_attention_scores_csv(
            slide_id=slide_id,
            coords=coords,
            attn_dict=attn_dict,
            save_dir=save_dir,
        )


    shared_slide = openslide.open_slide(wsi_path)

    heatmaps = {}
    raw_maps = {}
    thumbnail_img = None
    tissue_mask = None

    try:
        if heatmap_mode == "block":
            HeatmapGenerator = PatchBasedHeatmapGeneratorBlock
        elif heatmap_mode == "region":
            HeatmapGenerator = PatchBasedHeatmapGeneratorRegion
        else:
            raise ValueError(f"heatmap_mode={heatmap_mode} no supported")

        generator = HeatmapGenerator(
            slide_path=wsi_path,
            patch_level=patch_level,
            coordinates=coords,
            scores=np.zeros(len(coords), dtype=np.float32),
            patch_size=(patch_size, patch_size),
            slide_obj=shared_slide,
        )

        shared_ctx = generator.prepare_context(
            thumbnail_size_scale=thumb_scale,
            use_tissue_mask=use_tissue_mask,
        )
        thumbnail_img = shared_ctx["thumbnail"]
        tissue_mask = shared_ctx["tissue_mask"]

        for s in SUBTYPES:
            raw_scores = np.asarray(attn_dict[s].detach().cpu().numpy()).reshape(-1).astype(np.float32)
            generator.set_scores(raw_scores)

            if heatmap_mode == "block":
                thumb_img, heatmap_img, raw_dict = generator.generate_heatmap(
                    shared_ctx=shared_ctx,
                    style=SINGLE_CLASS_STYLE,
                    alpha=alpha,
                    normalize_method=normalize_method,
                    return_raw=True,
                )
            else:
                thumb_img, heatmap_img, raw_dict = generator.generate_heatmap(
                    shared_ctx=shared_ctx,
                    style=SINGLE_CLASS_STYLE,
                    alpha=alpha,
                    normalize_method=normalize_method,
                    upsample=upsample,
                    sigma_ratio=sigma_ratio,
                    alpha_sigma_ratio=alpha_sigma_ratio,
                    return_raw=True,
                )

            _ = thumb_img
            heatmaps[s] = heatmap_img
            raw_maps[s] = raw_dict

            save_single_subtype_figure(
                slide_id=slide_id,
                subtype=s,
                image=heatmap_img,
                pred=preds_dict.get(s),
                gt=safe_get_gt(gt, s),
                output_path=os.path.join(save_dir, f"{slide_id}_{s}_single.png"),
                style=SINGLE_CLASS_STYLE,
                dpi=dpi,
            )

        multiclass_img, winner_idx, winner_strength, confident_mask, low_conf_mask = build_multiclass_semantic_map_image(
            thumbnail=thumbnail_img,
            raw_maps=raw_maps,
            tissue_mask=tissue_mask,
            min_strength=0.12,
            min_margin=0.025,
            gamma=0.90,
            support_thresh=0.015,
            class_colors=MULTI_CLASS_COLORS,
            low_conf_color=LOW_CONF_TISSUE_COLOR,
            background_color=BACKGROUND_COLOR,
        )
        _ = winner_idx, winner_strength, confident_mask, low_conf_mask

        save_multiclass_figure(
            slide_id=slide_id,
            image=multiclass_img,
            preds_dict=preds_dict,
            gt_dict=gt,
            output_path=os.path.join(save_dir, f"{slide_id}_multiclass.png"),
            gt_subtype=gt_sub,
            dpi=dpi,
            class_colors=MULTI_CLASS_COLORS,
        )

        save_panel4_figure(
            slide_id=slide_id,
            output_path=os.path.join(save_dir, f"{slide_id}_panel4.png"),
            basal_img=heatmaps["Basal"],
            luma_img=heatmaps["LumA"],
            lumb_img=heatmaps["LumB"],
            multiclass_img=multiclass_img,
            gt_dict=gt,
            preds_dict=preds_dict,
            gt_subtype=gt_sub,
            style=SINGLE_CLASS_STYLE,
            dpi=max(dpi, 260),
            class_colors=MULTI_CLASS_COLORS,
        )

    finally:
        shared_slide.close()

    print(f"  [finish] {save_dir} | pred_subtype={pred_sub}")
    return preds_dict


def save_csv(results, output_dir):
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "predictions_summary.csv"), index=False)



def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", required=True)
    p.add_argument("--slides_dir", required=True)
    p.add_argument("--model_root", required=True)
    p.add_argument("--label_csv", required=True)
    p.add_argument("--output_dir", default="./heatmap_output")

    p.add_argument("--in_dim", type=int, default=1536)
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--patch_level", type=int, default=0)
    p.add_argument("--dpi", type=int, default=240)
    p.add_argument("--fold", type=int, default=0, choices=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda")

    p.add_argument("--slide_ids", nargs="*", default=None)
    p.add_argument("--mamba_type", default="spatial_mamba", choices=["SRMamba", "spatial_mamba", "Mamba", "BiMamba"])

    p.add_argument("--thumb_scale", type=float, nargs=2, default=(0.125, 0.125))
    p.add_argument("--style", default="jet",
                   choices=['coolwarm', 'hot', 'bwr', 'Spectral', 'seismic', 'jet', 'viridis', 'turbo', 'cividis'])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--normalize_method", default="sigmod", choices=["close", "sigmod", "sigmoid", "rank"])

    p.add_argument("--heatmap_mode", default="region", choices=["block", "region"],)
    p.add_argument("--upsample", type=float, default=3.0)
    p.add_argument("--sigma_ratio", type=float, default=0.55)
    p.add_argument("--alpha_sigma_ratio", type=float, default=0.85)
    p.add_argument("--disable_tissue_mask", action="store_true", default=False)

    p.add_argument("--disable_patch_csv", action="store_true", default=False)

    return p.parse_args()


def main():
    args = parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"[INFO] device={dev}")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.label_csv)
    if "slide_id" not in df.columns:
        raise KeyError("label_csv loss [slide_id] ")
    df["slide_id"] = df["slide_id"].astype(str)
    lmap = {r["slide_id"]: r for _, r in df.iterrows()}

    ids = [str(s) for s in args.slide_ids] if args.slide_ids else list(df["slide_id"])
    print(f"[INFO] 共 {len(ids)} 个 slide")
    print(f"[INFO] heatmap_mode={args.heatmap_mode}")
    print(f"[INFO] save_patch_csv={not args.disable_patch_csv}")

    model = load_model(args.model_root, args.in_dim, dev, args.mamba_type, args.fold)

    results = []
    failed = []

    for sid in ids:
        try:
            row = lmap.get(sid)
            preds = process_slide(
                slide_id=sid,
                data_root=args.data_root,
                slides_dir=args.slides_dir,
                model=model,
                row=row,
                output_dir=args.output_dir,
                patch_size=args.patch_size,
                patch_level=args.patch_level,
                device=dev,
                thumb_scale=tuple(args.thumb_scale),
                style=args.style,
                alpha=args.alpha,
                normalize_method=args.normalize_method,
                heatmap_mode=args.heatmap_mode,
                upsample=args.upsample,
                sigma_ratio=args.sigma_ratio,
                alpha_sigma_ratio=args.alpha_sigma_ratio,
                dpi=args.dpi,
                use_tissue_mask=not args.disable_tissue_mask,
                save_patch_csv=not args.disable_patch_csv,
            )

            r = {
                "slide_id": sid,
                "pred_Basal": preds["Basal"],
                "pred_LumA": preds["LumA"],
                "pred_LumB": preds["LumB"],
                "pred_subtype": infer_discrete_label_from_scores(preds),
            }

            if row is not None:
                r.update({
                    "gt_Basal": safe_to_float(row.get("label_Basal", np.nan), default=np.nan),
                    "gt_LumA": safe_to_float(row.get("label_LumA", np.nan), default=np.nan),
                    "gt_LumB": safe_to_float(row.get("label_LumB", np.nan), default=np.nan),
                    "gt_subtype": str(row.get("final_subtype", "")),
                })

            results.append(r)

        except Exception as e:
            print(f"[ERROR] {sid}: {e}")
            import traceback
            traceback.print_exc()
            failed.append(sid)

    if results:
        save_csv(results, args.output_dir)



if __name__ == "__main__":
    main()