#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import cv2
import glob
import time
import json
import argparse
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = THIS_DIR


def _natural_key(p):
    name = os.path.basename(p)
    parts = re.split(r'(\d+)', name)
    return [int(s) if s.isdigit() else s for s in parts]


def prepare_symlinks(img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imgs = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
        imgs.extend(glob.glob(os.path.join(img_dir, pat)))
    imgs = sorted(set(imgs), key=_natural_key)
    for i, p in enumerate(imgs):
        dst = os.path.join(out_dir, f"{i}.jpg")
        if os.path.islink(dst) or os.path.exists(dst):
            try:
                os.unlink(dst)
            except Exception:
                pass
        try:
            os.symlink(os.path.abspath(p), dst)
        except OSError:
            import shutil
            shutil.copy2(p, dst)
    return len(imgs)


def prepare_symlinks_from_list(img_paths, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    imgs = list(img_paths)
    for i, p in enumerate(imgs):
        dst = os.path.join(out_dir, f"{i}.jpg")
        if os.path.islink(dst) or os.path.exists(dst):
            try:
                os.unlink(dst)
            except Exception:
                pass
        try:
            os.symlink(os.path.abspath(p), dst)
        except OSError:
            import shutil
            shutil.copy2(p, dst)
    return len(imgs)


def read_lasot_gt_all(gt_path):
    arr = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                vals = [float(x) for x in line.split(',')]
            else:
                vals = [float(x) for x in re.split(r"[\s,]+", line) if x]
            if len(vals) >= 8:
                xs = [vals[0], vals[2], vals[4], vals[6]]
                ys = [vals[1], vals[3], vals[5], vals[7]]
                x1, y1 = min(xs), min(ys)
                x2, y2 = max(xs), max(ys)
                x, y, w, h = x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)
            else:
                x, y, w, h = vals[:4]
            arr.append([x, y, w, h])
    return np.array(arr, dtype=np.float32)


def _read_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _uavdt_list_images(img_root, prefix):
    pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG")
    all_imgs = []
    subdir = os.path.join(img_root, prefix)
    if os.path.isdir(subdir):
        for pat in pats:
            all_imgs.extend(glob.glob(os.path.join(subdir, pat)))
    if not all_imgs:
        for pat in pats:
            all_imgs.extend(glob.glob(os.path.join(img_root, f"{prefix}_*{pat.split('*')[-1]}")))
    all_imgs = [p for p in all_imgs if os.path.basename(p).startswith(f"{prefix}_")]
    all_imgs = sorted(set(all_imgs), key=_natural_key)
    return all_imgs


def _uavdt_extract_objs(ann_obj):
    objs = []
    try:
        size = ann_obj.get('size', {})
        H = int(size.get('height', 0) or 0)
        W = int(size.get('width', 0) or 0)
        for o in ann_obj.get('objects', []) is not None and ann_obj.get('objects', []) or []:
            cls = o.get('classTitle', '')
            pts = o.get('points', {}).get('exterior', None)
            if not pts or len(pts) < 2:
                continue
            (x1, y1), (x2, y2) = pts[0], pts[1]
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            tid = None
            for k in ('target_id', 'track_id', 'tracking_id', 'id'):
                if k in o and o[k] is not None:
                    try:
                        tid = int(o[k])
                        break
                    except Exception:
                        pass
            if tid is None:
                for tg in o.get('tags', []) or []:
                    name = str(tg.get('name', '')).lower()
                    if name in ('target id', 'target_id', 'id', 'track id', 'track_id', 'tracking id', 'tracking_id'):
                        try:
                            tid = int(tg.get('value'))
                        except Exception:
                            tid = None
                        break
            objs.append(dict(classTitle=cls, bbox=[x1, y1, w, h], target_id=tid))
        return H, W, objs
    except Exception:
        return 0, 0, []


def _uavdt_find_ann_path(ann_root, img_path):
    name = os.path.basename(img_path)
    seq_dir = os.path.basename(os.path.dirname(img_path))
    candidates = []
    if os.path.isdir(os.path.join(ann_root, seq_dir)):
        dir1 = os.path.join(ann_root, seq_dir)
        candidates.append(os.path.join(dir1, f"{name}.json"))
        candidates.append(os.path.join(dir1, f"{os.path.splitext(name)[0]}.json"))
    candidates.append(os.path.join(ann_root, f"{name}.json"))
    candidates.append(os.path.join(ann_root, f"{os.path.splitext(name)[0]}.json"))
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _expand_desired_classes(desired):
    if desired is None:
        return set(['car', 'vehicle'])
    if isinstance(desired, (list, tuple, set)):
        tokens = [str(x).strip().lower() for x in desired if str(x).strip()]
    else:
        tokens = [t.strip().lower() for t in re.split(r"[\s,]+", str(desired)) if t.strip()]
    expanded = set()
    for t in tokens:
        if t == 'car':
            expanded.update(['car', 'vehicle'])
        elif t == 'vehicle':
            expanded.update(['vehicle', 'car'])
        elif t == 'truck':
            expanded.update(['truck', 'lorry'])
        elif t == 'bus':
            expanded.update(['bus', 'coach'])
        else:
            expanded.add(t)
    return expanded if len(expanded) > 0 else set(['car', 'vehicle'])


def _uavdt_select_initial(img_list, ann_root, desired_class='car'):
    desired_set = _expand_desired_classes(desired_class)
    for idx, img_path in enumerate(img_list):
        jpath = _uavdt_find_ann_path(ann_root, img_path)
        ann = _read_json(jpath) if jpath else None
        if ann is None:
            continue
        H, W, objs = _uavdt_extract_objs(ann)
        cars = [o for o in objs if str(o.get('classTitle', '')).lower() in desired_set]
        if not cars:
            continue
        with_id = [o for o in cars if o.get('target_id') is not None]
        if with_id:
            sel = sorted(with_id, key=lambda o: int(o['target_id']))[0]
        else:
            sel = sorted(cars, key=lambda o: (o['bbox'][2] * o['bbox'][3]), reverse=True)[0]
        return idx, sel.get('target_id'), sel.get('bbox'), (H, W)
    return None, None, None, (None, None)


def _uavdt_build_gt(img_list, ann_root, start_idx, target_id, desired_class='car',
                    init_bbox=None, iou_link=True, iou_thresh=0.3):
    desired_set = _expand_desired_classes(desired_class)
    gt = []
    valid_idx = []
    prev = list(init_bbox) if (init_bbox is not None) else None

    def _iou_xywh_local(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / float(union) if union > 0 else 0.0

    for i in range(start_idx, len(img_list)):
        jpath = _uavdt_find_ann_path(ann_root, img_list[i])
        ann = _read_json(jpath) if jpath else None
        if ann is None:
            gt.append([0.0, 0.0, 0.0, 0.0])
            continue
        _, _, objs = _uavdt_extract_objs(ann)
        cands = [o for o in objs if str(o.get('classTitle', '')).lower() in desired_set]
        cand = None
        if target_id is not None:
            for o in cands:
                if o.get('target_id') == target_id:
                    cand = o
                    break
            if cand is None and len(cands) > 0:
                if iou_link and prev is not None:
                    best = max(cands, key=lambda o: _iou_xywh_local(prev, o['bbox']))
                    cand = best
                    prev = list(best['bbox'])
                else:
                    cand = cands[0]
                    prev = list(cand['bbox'])
        else:
            if iou_link and prev is not None and len(cands) > 0:
                best = max(cands, key=lambda o: _iou_xywh_local(prev, o['bbox']))
                if _iou_xywh_local(prev, best['bbox']) >= float(iou_thresh):
                    cand = best
                else:
                    cand = best
                prev = list(best['bbox'])
            elif len(cands) > 0:
                cand = cands[0]
                prev = list(cand['bbox'])

        if cand is None:
            gt.append([0.0, 0.0, 0.0, 0.0])
            continue
        b = cand.get('bbox', [0.0, 0.0, 0.0, 0.0])
        gt.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
        valid_idx.append(i - start_idx)
    return np.array(gt, dtype=np.float32), valid_idx


def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def eval_metrics(gt, pr, img_hw=None):
    N = min(len(gt), len(pr))
    if N == 0:
        return dict(auc=0.0, pnorm=0.0, p=0.0, n=0)
    gt = gt[:N]
    pr = pr[:N]
    ious = np.array([iou_xywh(gt[i], pr[i]) for i in range(N)], dtype=np.float32)
    ths = np.linspace(0, 1, 101)
    succ = [(ious >= t).mean() for t in ths]
    auc = float(np.mean(succ))

    def center(b):
        return np.array([b[0] + b[2] / 2.0, b[1] + b[3] / 2.0], dtype=np.float32)

    cle = np.linalg.norm(np.stack([center(gt[i]) - center(pr[i]) for i in range(N)], axis=0), axis=1)
    p = float((cle <= 20.0).mean())

    if img_hw is not None:
        H, W = img_hw
        diag = float(np.sqrt(W * W + H * H))
        diag = max(diag, 1e-6)
        norm_err = cle / diag
    else:
        gt_diag = np.sqrt(np.maximum(gt[:, 2], 1e-6) ** 2 + np.maximum(gt[:, 3], 1e-6) ** 2)
        norm_err = cle / gt_diag

    ths_norm = np.linspace(0.0, 0.5, 101)
    prec_norm_curve = [(norm_err <= t).mean() for t in ths_norm]
    pnorm = float(np.mean(prec_norm_curve))
    return dict(auc=auc, pnorm=pnorm, p=p, n=N)


def eval_region_metric(gt, pr, img_hw=None, ignore_zero_gt=True):
    N = min(len(gt), len(pr))
    if N == 0 or img_hw is None:
        return dict(region_acc=0.0, correct=0, n=0)
    H, W = img_hw
    if H is None or W is None or H <= 0 or W <= 0:
        return dict(region_acc=0.0, correct=0, n=0)

    mask = np.ones(N, dtype=bool)
    if ignore_zero_gt:
        mask = (gt[:N, 2] > 0) & (gt[:N, 3] > 0)

    def region_id(b):
        cx = float(b[0]) + float(b[2]) / 2.0
        cy = float(b[1]) + float(b[3]) / 2.0
        left = cx < (W / 2.0)
        top = cy < (H / 2.0)
        if top and left:
            return 0
        if top and (not left):
            return 1
        if (not top) and left:
            return 2
        return 3

    gt = gt[:N][mask]
    pr = pr[:N][mask]
    N = len(gt)
    if N == 0:
        return dict(region_acc=0.0, correct=0, n=0)
    correct = 0
    for i in range(N):
        if region_id(gt[i]) == region_id(pr[i]):
            correct += 1
    acc = float(correct) / float(max(1, N))
    return dict(region_acc=acc, correct=int(correct), n=int(N))


def eval_region_with_nan_policy(gt, pr, states, img_hw=None, ignore_zero_gt=True):
    N = min(len(gt), len(pr), len(states))
    if N == 0 or img_hw is None:
        return dict(region_acc=0.0, correct=0, n=0)
    H, W = img_hw
    if H is None or W is None or H <= 0 or W <= 0:
        return dict(region_acc=0.0, correct=0, n=0)

    def is_nan_box(b):
        return np.any(np.isnan(b[:4]))

    def region_id(b):
        cx = float(b[0]) + float(b[2]) / 2.0
        cy = float(b[1]) + float(b[3]) / 2.0
        left = cx < (W / 2.0)
        top = cy < (H / 2.0)
        if top and left:
            return 0
        if top and (not left):
            return 1
        if (not top) and left:
            return 2
        return 3

    correct = 0
    count = 0
    for i in range(N):
        g = gt[i]
        p = pr[i]
        st = states[i] if i < len(states) else 'predict'
        if is_nan_box(g):
            count += 1
            verdict = 'success' if st != 'track' else 'failure'
            try:
                print(f"[VOT-NaN] Frame {i}: state={st} -> {verdict}")
            except Exception:
                pass
            if st != 'track':
                correct += 1
            continue
        if ignore_zero_gt and (g[2] <= 0 or g[3] <= 0):
            continue
        count += 1
        if region_id(g) == region_id(p):
            correct += 1
    acc = float(correct) / float(max(1, count))
    return dict(region_acc=acc, correct=int(correct), n=int(count))


def _infer_action_from_error(dx, dy, img_w, img_h):
    W2 = max(1.0, float(img_w) / 2.0)
    H2 = max(1.0, float(img_h) / 2.0)
    dx_n = float(dx) / W2
    dy_n = -float(dy) / H2
    if abs(dx_n) >= abs(dy_n):
        return "Right" if dx_n > 0 else "Left"
    else:
        return "Forward" if dy_n > 0 else "Backward"


def plot_gt_vs_pred_action_scatter(gt_all, pr_all, img_hw, save_path, title=None,
                                   max_points_per_class=20, marker_size=50,
                                   font_family='Times New Roman', font_ttf=None):
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib import font_manager as fm

    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams['svg.fonttype'] = 'none'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'

    bold_prop = None
    reg_prop = None
    try:
        if font_ttf and isinstance(font_ttf, str) and len(font_ttf) > 0 and os.path.isfile(font_ttf):
            fm.fontManager.addfont(font_ttf)
            reg_prop = fm.FontProperties(fname=font_ttf)
            fam = reg_prop.get_name() or font_family
            rcParams['font.family'] = fam
            font_dir = os.path.dirname(font_ttf)
            candidates = [
                os.path.join(font_dir, 'Times_New_Roman_Bold.ttf'),
                os.path.join(font_dir, 'timesbd.ttf'),
                os.path.join(font_dir, 'Times New Roman Bold.ttf')
            ]
            for bp in candidates:
                if os.path.isfile(bp):
                    fm.fontManager.addfont(bp)
                    bold_prop = fm.FontProperties(fname=bp)
                    break
            if bold_prop is None:
                bold_prop = fm.FontProperties(family=fam, weight='bold')
        else:
            rcParams['font.family'] = font_family
            rcParams['font.serif'] = [font_family, 'Times', 'Times New Roman', 'Nimbus Roman', 'Liberation Serif', 'DejaVu Serif']
            reg_prop = fm.FontProperties(family=font_family)
            bold_prop = fm.FontProperties(family=font_family, weight='bold')
        rcParams['mathtext.default'] = 'regular'
    except Exception:
        reg_prop = fm.FontProperties(family=font_family)
        bold_prop = fm.FontProperties(family=font_family, weight='bold')

    H, W = img_hw
    if not (H and W) or H <= 0 or W <= 0:
        return

    cx, cy = W / 2.0, H / 2.0
    xs = {"Forward": [], "Backward": [], "Left": [], "Right": []}
    ys = {"Forward": [], "Backward": [], "Left": [], "Right": []}
    all_dx = []
    all_dy = []

    N = min(len(gt_all), len(pr_all))
    for i in range(N):
        g = gt_all[i]
        if np.any(np.isnan(g[:4])) or g[2] <= 0 or g[3] <= 0:
            continue
        p = pr_all[i]

        gcx = float(g[0]) + float(g[2]) / 2.0
        gcy = float(g[1]) + float(g[3]) / 2.0
        dx_gt, dy_gt = gcx - cx, cy - gcy

        pcx = float(p[0]) + float(p[2]) / 2.0
        pcy = float(p[1]) + float(p[3]) / 2.0
        dx_pr, dy_pr = pcx - cx, pcy - cy
        act = _infer_action_from_error(dx_pr, dy_pr, W, H)

        xs[act].append(dx_gt)
        ys[act].append(dy_gt)
        all_dx.append(dx_gt)
        all_dy.append(dy_gt)

    FIG_H = 5.0
    fig_w = max(4.0, float(FIG_H) * (float(W) / float(max(1, H))))
    plt.figure(figsize=(fig_w, FIG_H))
    colors = {"Forward": "#e41a1c", "Backward": "#377eb8", "Left": "#4daf4a", "Right": "#ff7f00"}
    order = ["Forward", "Backward", "Left", "Right"]
    import numpy as _np
    for act in order:
        if xs[act]:
            nx = len(xs[act])
            m = int(max(1, max_points_per_class))
            if nx > m:
                step = int(_np.ceil(nx / float(m)))
                keep_idx = list(range(0, nx, step))[:m]
                xs_plot = [xs[act][i] for i in keep_idx]
                ys_plot = [ys[act][i] for i in keep_idx]
            else:
                xs_plot = xs[act]
                ys_plot = ys[act]
            plt.scatter(xs_plot, ys_plot, s=marker_size, c=colors[act], label=act,
                        alpha=0.85, edgecolors='none')

    try:
        if len(all_dx) > 0:
            dx_abs = _np.abs(_np.asarray(all_dx, dtype=float))
            base_x = _np.percentile(dx_abs, 95.0)
            limx = min(W / 2.0, max(30.0, float(base_x) * 1.15))
        else:
            limx = W / 2.0
        if len(all_dy) > 0:
            dy_abs = _np.abs(_np.asarray(all_dy, dtype=float))
            base_y = _np.percentile(dy_abs, 95.0)
            limy = min(H / 2.0, max(30.0, float(base_y) * 1.15))
        else:
            limy = H / 2.0
    except Exception:
        limx, limy = W / 2.0, H / 2.0

    plt.axhline(0, color="gray", lw=0.8, ls="--")
    plt.axvline(0, color="gray", lw=0.8, ls="--")

    plt.xlim(-limx, limx)
    plt.ylim(-limy, limy)
    ax = plt.gca()
    plt.xlabel("Target Center Position (px)", fontproperties=bold_prop)
    plt.ylabel("Target Center Position (py)", fontproperties=bold_prop)
    leg = plt.legend(title="Predicted Action", loc="best", framealpha=0.9,
                     prop=bold_prop, title_fontproperties=bold_prop)
    try:
        leg.get_title().set_fontproperties(bold_prop)
        for t in leg.get_texts():
            t.set_fontproperties(bold_prop)
    except Exception:
        pass

    try:
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontproperties(bold_prop)
    except Exception:
        pass
    plt.grid(True, alpha=0.2)

    root, ext = os.path.splitext(save_path)
    if str(ext).lower() not in ('.pdf', '.svg', '.eps'):
        save_path = root + '.pdf'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    fmt = os.path.splitext(save_path)[1][1:].lower()
    plt.savefig(save_path, format=fmt)
    try:
        plt.show()
    except Exception:
        pass


def run_ortrack_with_reacq_and_enhance(img_dir, gt_all, out_dir,
                                       or_prj_path, or_model_name, or_checkpoint,
                                       yolo_model_path,
                                       ref_from_gt=True, ref_mask_from_yolo=True, ref_yolo_model=None, ref_yolo_conf=0.25,
                                       or_lost_thresh=0.6, or_max_lost=3, or_template=128, or_search=256,
                                       kf_history_len=8, kf_max_predict_frames=15,
                                       reacq_interval=2, reacq_crop_expand=0.4, reacq_iou_with_pred=0.2, reacq_match_thresh=0.70, reacq_fullframe_after_n=3,
                                       online_enhance=True, enhance_interval=20, enhance_alpha=0.2, enhance_iou_thresh=0.3,
                                       save_images_to="",
                                       fps_log_frames=100,
                                       classes_prompts=None,
                                       manual_ref_paths=None,
                                       kf_mode='standard'):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = None
    if isinstance(save_images_to, str) and len(save_images_to) > 0:
        vis_dir = save_images_to
        try:
            os.makedirs(vis_dir, exist_ok=True)
        except Exception:
            vis_dir = None

    sys.path.insert(0, PROJ_ROOT)

    from detectors import initialize_ortrack_tracker, initialize_yolo_model
    from dinov3_feature_extractor import DINOv3FeatureExtractor
    import torch

    tracker = initialize_ortrack_tracker(
        ortrack_prj_path=or_prj_path,
        model_name=or_model_name,
        checkpoint_path=or_checkpoint,
        lost_threshold=float(or_lost_thresh),
        max_lost_frames=int(or_max_lost),
        keep_last_position=True,
        template_size=int(or_template),
        search_size=int(or_search)
    )

    target_classes_list = None
    if classes_prompts is not None:
        if isinstance(classes_prompts, str):
            target_classes_list = str(classes_prompts).strip().split()
        elif isinstance(classes_prompts, (list, tuple)):
            target_classes_list = [str(x).strip() for x in classes_prompts if str(x).strip()]

    detector = initialize_yolo_model(model_path=yolo_model_path, target_classes=target_classes_list)

    feature_extractor = DINOv3FeatureExtractor(device=('cuda' if torch.cuda.is_available() else 'cpu'))

    _img_list = []
    for _pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
        _img_list.extend(glob.glob(os.path.join(img_dir, _pat)))
    imgs = sorted(set(_img_list), key=_natural_key)
    if len(imgs) == 0:
        raise FileNotFoundError(f"No images under {img_dir}")
    if len(gt_all) < 1:
        raise RuntimeError("Empty groundtruth for sequence")

    first_bgr = cv2.imread(imgs[0], cv2.IMREAD_COLOR)
    if first_bgr is None:
        raise FileNotFoundError(imgs[0])
    init_bbox = [int(round(v)) for v in gt_all[0][:4]]

    preds = []
    states = []

    pred_csv_path = os.path.join(out_dir, 'pred.csv')
    os.makedirs(out_dir, exist_ok=True)
    pred_fh = open(pred_csv_path, 'w', encoding='utf-8')
    pred_fh.write('frame,x,y,w,h,conf,state\n')

    def _append_pred(frame_idx, bbox, conf, state):
        bx, by, bw, bh = [int(round(float(v))) for v in bbox]
        preds.append([float(bx), float(by), float(bw), float(bh)])
        states.append(str(state) if state is not None else 'track')
        try:
            pred_fh.write(f"{int(frame_idx)},{bx},{by},{bw},{bh},{float(conf):.6f},{states[-1]}\n")
        except Exception:
            pass

    class _ImageSequenceVideo:
        def __init__(self, image_paths):
            self.image_paths = list(image_paths)
            self.idx = 0

        def read(self):
            if self.idx >= len(self.image_paths):
                return False, None
            p = self.image_paths[self.idx]
            self.idx += 1
            fr = cv2.imread(p, cv2.IMREAD_COLOR)
            if fr is None:
                return False, None
            return True, fr

    def step_cb(frame_idx, frame_bgr, track_bbox, predict_bbox, det_bboxes, det_best_idx, is_lost, state, conf):
        if int(frame_idx) == 0:
            _append_pred(0, init_bbox, conf=1.0, state='init')
            return
        if track_bbox is not None:
            _append_pred(frame_idx, track_bbox, conf=(conf if conf is not None else 1.0), state=(state or 'track'))
        elif predict_bbox is not None:
            _append_pred(frame_idx, predict_bbox, conf=(conf if conf is not None else 0.0), state=(state or 'predict'))
        else:
            last = preds[-1] if preds else init_bbox
            _append_pred(frame_idx, last, conf=(conf if conf is not None else 0.0), state=(state or 'missing'))

    cfg = {
        'path_to_video': img_dir,
        'desired_height': int(first_bgr.shape[0]),
        'desired_width': int(first_bgr.shape[1]),
        'fps': 0,
        'wait_key': 1,
        'plot_visualizations': False,
        'save_images_to': (vis_dir if vis_dir else False),
        'save_video_to': '',
        'video_writer_fps': None,
        'unreal_seed': None,
        'eval_table1': False,
        'episodes': 1,
        'distractors': None,
        'target_appearance_id': 2,
        'distractor_appearance_id': 4,
        'text_query': (" ".join(target_classes_list) if target_classes_list else ''),
        'yolo_model_path': yolo_model_path,
        'yolo_confidence_thresh': float(ref_yolo_conf),
        'max_detection_attempts': 1,
        'ortrack_prj_path': or_prj_path,
        'ortrack_model_name': or_model_name,
        'ortrack_checkpoint_path': or_checkpoint,
        'ortrack_lost_thresh': float(or_lost_thresh),
        'ortrack_max_lost_frames': int(or_max_lost),
        'ortrack_template_size': int(or_template),
        'ortrack_search_size': int(or_search),
        'reference_feature_path': None,
        'match_similarity_thresh': float(reacq_match_thresh),
        'dinov3_model_name': 'dinov3_vith16plus',
        'dinov3_checkpoint_path': None,
        'online_enhance': bool(online_enhance),
        'enhance_interval': int(enhance_interval),
        'enhance_alpha': float(enhance_alpha),
        'enhance_iou_thresh': float(enhance_iou_thresh),
        'enhance_crop_expand': 0.2,
        'reacq_parallel': True,
        'reacq_interval': int(reacq_interval),
        'reacq_crop_expand': float(reacq_crop_expand),
        'reacq_iou_with_pred': float(reacq_iou_with_pred),
        'reacq_match_thresh': float(reacq_match_thresh),
        'reacq_fullframe_after_n': int(reacq_fullframe_after_n),
        'fly_drone': False,
        'auto_takeoff': False,
        'target_area_ratio': 0.12,
        'area_tolerance_ratio': 0.20,
        'debug_control': False,
        'ctrl_yaw_kp': 0.45,
        'ctrl_yaw_ki': 0.0,
        'ctrl_yaw_kd': 0.1,
        'ctrl_vz_kp': 0.5,
        'ctrl_vz_ki': 0.0,
        'ctrl_vz_kd': 0.05,
        'ctrl_vy_kp': 0.35,
        'ctrl_vy_ki': 0.0,
        'ctrl_vy_kd': 0.15,
        'ctrl_dt': 0.1,
        'ctrl_max_speed': 40,
        'ctrl_low_battery': 15,
        'ctrl_vy_min_output': 0.8,
        'kf_history_len': int(kf_history_len),
        'kf_max_predict_frames': int(kf_max_predict_frames),
        'kf_mode': str(kf_mode),
        'dp_enable': False,
        'dp_repo_root': '',
        'dp_checkpoint': '',
        'dp_action_exec_steps': None,
        'dp_max_plans': 0,
        'dp_dt': 1,
    }

    from tracking.ortrack_core import track_object_with_ortrack

    video = _ImageSequenceVideo(imgs)
    try:
        track_object_with_ortrack(
            tracker,
            init_bbox,
            cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB),
            video,
            cfg,
            detector=detector,
            feature_extractor=feature_extractor,
            reference_feature=None,
            vehicle=None,
            ref_store=None,
            enhancer=None,
            step_cb=step_cb,
        )
    finally:
        try:
            pred_fh.close()
        except Exception:
            pass

    return np.array(preds, dtype=np.float32), states


def main():
    ap = argparse.ArgumentParser(description="Evaluate on DBT70/VOT-like: ORTrack (GT-init) with reacq/enhance + quadrant success")
    ap.add_argument("--dat_root", type=str, default=None, help="DBT70 根目录，如 /home/ws0/gh/follow/nips_data/DBT70")
    ap.add_argument("--testing_set", type=str, default=None, help="testing_set.txt 路径（不传则只跑 --sequence/--category/--vot_root）")
    ap.add_argument("--sequence", type=str, default=None, help="VOT: 指定序列名（支持逗号/空格分隔多个），不填则遍历全部")
    ap.add_argument("--category", type=str, default=None, help="DBT70: 类别名，自动遍历子序列")
    ap.add_argument("--vot_root", type=str, default=None, help="VOT2018 根目录（含 ann/ 与 img/），如 /home/ws0/gh/follow/nips_data/VOT2018")
    ap.add_argument("--uavdt_root", type=str, default=None, help="UAVDT 根目录（含 test/train），如 /home/ws0/gh/follow/nips_data/UAVDT")
    ap.add_argument("--uavdt_split", type=str, default="test", help="UAVDT split: test 或 train")
    ap.add_argument("--uavdt_sequences", type=str, default="", help="逗号分隔的视频前缀列表，如 S1603,S0201 ...")
    ap.add_argument("--uavdt_class", type=str, default="car", help="UAVDT 选取的类别，缺省 car")
    ap.add_argument("--uavdt_iou_link", action="store_true", default=True,
                    help="UAVDT: 无 target id 时用 IoU 将 GT 跨帧关联（默认开）")
    ap.add_argument("--uavdt_iou_thresh", type=float, default=0.3,
                    help="UAVDT: IoU 关联阈值（仅在无 id 时使用）")
    ap.add_argument("--ignore_zero_gt", action="store_true", default=True,
                    help="评测时忽略 GT 为零框的帧")
    ap.add_argument("--follow2_py", type=str, default=os.path.join(PROJ_ROOT, "main.py"))
    ap.add_argument("--out_root", type=str, default=os.path.join(PROJ_ROOT, "out_dat_follow2"))
    ap.add_argument("--yolo_weight", type=str, default=os.path.join(PROJ_ROOT, "ORTrack/weights/yoloe-11l-seg.pt"))
    ap.add_argument("--text_query", type=str, default="", help='如 "person car"，可留空')
    ap.add_argument("--ref_feature", type=str, default="", help="手动参考特征文件路径，支持逗号分隔多个（.pt 或 .npy）")
    ap.add_argument("--seed_ref_from_gt", action="store_true", default=True, help="用首帧GT提取DINO参考特征并传给follow2")
    ap.add_argument("--save_video", action="store_true", default=False)
    ap.add_argument("--ref_mask_from_yolo", action="store_true", default=True,
                    help="首帧参考特征：用YOLOE分割掩码与GT相交做masked提特征（默认开启，失败自动回退）")
    ap.add_argument("--ref_yolo_model", type=str, default=os.path.join(PROJ_ROOT, "ORTrack/weights/yoloe-11l-seg.pt"),
                    help="YOLOE 分割模型权重路径")
    ap.add_argument("--ref_yolo_conf", type=float, default=0.25, help="YOLOE 置信度阈值")
    ap.add_argument("--kf_history_len", type=int, default=8)
    ap.add_argument("--kf_max_predict_frames", type=int, default=15)
    ap.add_argument("--reacq_interval", type=int, default=2)
    ap.add_argument("--reacq_crop_expand", type=float, default=0.4)
    ap.add_argument("--reacq_iou_with_pred", type=float, default=0.2)
    ap.add_argument("--reacq_match_thresh", type=float, default=0.70)
    ap.add_argument("--reacq_fullframe_after_n", type=int, default=3)
    ap.add_argument("--online_enhance", action="store_true", default=True)
    ap.add_argument("--enhance_interval", type=int, default=20)
    ap.add_argument("--enhance_alpha", type=float, default=0.2)
    ap.add_argument("--enhance_iou_thresh", type=float, default=0.3)
    ap.add_argument("--save_images_to", type=str, default="", help="保存可视化图像目录（为空则不保存）")
    ap.add_argument("--scatter_max_points", type=int, default=20, help="散点图每类最多点数（默认20）")
    ap.add_argument("--font_ttf", type=str, default="", help="Times New Roman 的 .ttf 路径（可选，确保字体一致）")
    ap.add_argument("--fps_log_frames", type=int, default=100, help="每N帧打印一次平均FPS（<=0 关闭）")
    ap.add_argument("--class_name", type=str, default=None, help="覆盖类别名，用于YOLO文本提示（如 car/person/...）")
    ap.add_argument("--or_prj_path", type=str, default=os.path.join(PROJ_ROOT, "ORTrack"), help="ORTrack 项目根目录")
    ap.add_argument("--or_model_name", type=str, default="deit_tiny_patch16_224", help="ORTrack 模型名")
    ap.add_argument("--or_checkpoint", type=str, default=None, help="ORTrack checkpoint（留空用默认路径）")
    ap.add_argument("--or_lost_thresh", type=float, default=0.6, help="ORTrack 丢失阈值")
    ap.add_argument("--or_max_lost", type=int, default=3, help="ORTrack 最大连续丢失帧")
    ap.add_argument("--or_template", type=int, default=128, help="模板尺寸")
    ap.add_argument("--or_search", type=int, default=256, help="搜索尺寸")

    args = ap.parse_args()

    seq_list = []
    if args.uavdt_sequences:
        for s in [t.strip() for t in re.split(r"[\s,]+", args.uavdt_sequences) if t.strip()]:
            seq_list.append(f"UAVDT::{s}")

    if args.sequence:
        for s in [t.strip() for t in re.split(r"[\s,]+", args.sequence) if t.strip()]:
            seq_list.append(s)

    if not seq_list and args.dat_root and args.category:
        base = os.path.join(args.dat_root, args.category)
        if os.path.isdir(base):
            for name in sorted(os.listdir(base)):
                seq_dir = os.path.join(base, name)
                if not os.path.isdir(seq_dir):
                    continue
                if os.path.isdir(os.path.join(seq_dir, 'img')) and os.path.isfile(os.path.join(seq_dir, 'groundtruth_rect.txt')):
                    seq_list.append(f"{args.category}/{name}")

    if not seq_list and args.vot_root:
        ann_dir = os.path.join(args.vot_root, 'ann')
        img_root = os.path.join(args.vot_root, 'img')
        if not (os.path.isdir(ann_dir) and os.path.isdir(img_root)):
            print(f"[ERR] 缺少 ann/ 或 img/ 目录: {args.vot_root}")
            sys.exit(1)
        for name in sorted(os.listdir(ann_dir)):
            if not os.path.isfile(os.path.join(ann_dir, name, 'groundtruth.txt')):
                continue
            if not os.path.isdir(os.path.join(img_root, name)):
                continue
            seq_list.append(name)

    if not seq_list:
        print("[ERR] 未提供任何有效序列。可使用 --sequence、或 --category 配合 --dat_root、或 --vot_root，或 --uavdt_sequences。")
        sys.exit(1)

    os.makedirs(args.out_root, exist_ok=True)
    all_logs = []
    for rel in seq_list:
        if rel.startswith('UAVDT::'):
            seq = rel.split('::', 1)[1]
            if os.path.isdir(os.path.join(args.uavdt_root, 'ann')) and os.path.isdir(os.path.join(args.uavdt_root, 'img')):
                base = args.uavdt_root
            else:
                base = os.path.join(args.uavdt_root, args.uavdt_split)
            ann_root = os.path.join(base, 'ann')
            img_root = os.path.join(base, 'img')
            img_list = _uavdt_list_images(img_root, prefix=seq)
            if not img_list:
                print(f"[SKIP] 无图像: {seq}")
                continue
            desired_classes = _expand_desired_classes(args.uavdt_class or 'car')
            start_idx, target_id, bbox0, (H0, W0) = _uavdt_select_initial(img_list, ann_root, desired_class=desired_classes)
            if start_idx is None:
                human_cls = ",".join(sorted(list(desired_classes)))
                print(f"[SKIP] 未找到类别 {human_cls} 的目标: {seq}")
                continue
            try:
                print(f"[UAVDT] {seq}: total_imgs={len(img_list)}, start_idx={start_idx}, HxW={H0}x{W0}, target_id={target_id}")
            except Exception:
                pass
            link_dir = os.path.join('/tmp', f"uavdt_{seq}_symlink")
            n_frames = prepare_symlinks_from_list(img_list[start_idx:], link_dir)
            try:
                print(f"[UAVDT] {seq}: symlink_frames={n_frames}")
            except Exception:
                pass
            out_dir = os.path.join(args.out_root, f"uavdt_{seq}")
            os.makedirs(out_dir, exist_ok=True)
            gt_xywh, _valid_idx = _uavdt_build_gt(
                img_list, ann_root, start_idx, target_id,
                desired_class=desired_classes,
                init_bbox=bbox0,
                iou_link=args.uavdt_iou_link,
                iou_thresh=args.uavdt_iou_thresh
            )
            pr_xywh, _states = run_ortrack_with_reacq_and_enhance(
                img_dir=link_dir,
                gt_all=gt_xywh,
                out_dir=out_dir,
                or_prj_path=args.or_prj_path,
                or_model_name=args.or_model_name,
                or_checkpoint=args.or_checkpoint,
                yolo_model_path=args.yolo_weight,
                ref_from_gt=True,
                ref_mask_from_yolo=True,
                ref_yolo_model=args.ref_yolo_model,
                ref_yolo_conf=args.ref_yolo_conf,
                or_lost_thresh=args.or_lost_thresh,
                or_max_lost=args.or_max_lost,
                or_template=args.or_template,
                or_search=args.or_search,
                kf_history_len=args.kf_history_len,
                kf_max_predict_frames=args.kf_max_predict_frames,
                reacq_interval=args.reacq_interval,
                reacq_crop_expand=args.reacq_crop_expand,
                reacq_iou_with_pred=args.reacq_iou_with_pred,
                reacq_match_thresh=args.reacq_match_thresh,
                reacq_fullframe_after_n=args.reacq_fullframe_after_n,
                online_enhance=args.online_enhance,
                enhance_interval=args.enhance_interval,
                enhance_alpha=args.enhance_alpha,
                enhance_iou_thresh=args.enhance_iou_thresh,
                save_images_to=args.save_images_to,
                fps_log_frames=args.fps_log_frames,
                classes_prompts=(args.class_name.split() if args.class_name else list(desired_classes)),
                manual_ref_paths=[p.strip() for p in str(args.ref_feature).split(',') if p.strip()],
                kf_mode='confaware'
            )
            try:
                plot_gt_vs_pred_action_scatter(
                    gt_xywh, pr_xywh, (H0, W0),
                    save_path=os.path.join(out_dir, "gt_vs_action_scatter.pdf"),
                    title=None,
                    max_points_per_class=int(args.scatter_max_points),
                    font_ttf=args.font_ttf
                )
            except Exception:
                pass
            region = eval_region_metric(gt_xywh, pr_xywh, img_hw=(H0, W0) if (H0 and W0) else None, ignore_zero_gt=args.ignore_zero_gt)
            log = dict(sequence=seq, **region)
            all_logs.append(log)
            print(json.dumps(log, ensure_ascii=False))
            continue

        if args.vot_root and '/' not in rel:
            seq = rel
            cls = 'vot'
            img_dir = os.path.join(args.vot_root, 'img', seq)
            gt_file = os.path.join(args.vot_root, 'ann', seq, 'groundtruth.txt')
        else:
            parts = rel.split('/')
            if len(parts) < 2:
                print(f"[SKIP] 非法序列名: {rel}")
                continue
            cls, seq = parts[0], parts[1]
            seq_dir = os.path.join(args.dat_root, rel)
            img_dir = os.path.join(seq_dir, "img")
            gt_file = os.path.join(seq_dir, "groundtruth_rect.txt")

        if not os.path.isdir(img_dir) or not os.path.isfile(gt_file):
            print(f"[SKIP] 缺失: {img_dir} 或 {gt_file}")
            continue

        _ = prepare_symlinks(img_dir, os.path.join("/tmp", f"{cls}_{seq}_symlink"))
        out_dir = os.path.join(args.out_root, f"{cls}_{seq}")
        os.makedirs(out_dir, exist_ok=True)

        gt_xywh = read_lasot_gt_all(gt_file)

        key = (args.class_name.lower() if args.class_name else cls.lower())
        synonyms = {
            "drone": "drone quadcopter uav",
            "airplane": "airplane plane jet",
            "car": "car vehicle",
            "person": "person human",
            "ship": "ship boat vessel",
            "boat": "boat ship",
            "bicycle": "bicycle bike",
            "motorbike": "motorbike motorcycle",
            "motorcycle": "motorcycle",
            "bus": "bus coach",
            "truck": "truck lorry",
            "vot": "car vehicle",
        }
        classes_prompts = synonyms.get(key, key)

        pr_xywh, states = run_ortrack_with_reacq_and_enhance(
            img_dir=img_dir,
            gt_all=gt_xywh,
            out_dir=out_dir,
            or_prj_path=args.or_prj_path,
            or_model_name=args.or_model_name,
            or_checkpoint=args.or_checkpoint,
            yolo_model_path=args.yolo_weight,
            ref_from_gt=args.seed_ref_from_gt,
            ref_mask_from_yolo=args.ref_mask_from_yolo,
            ref_yolo_model=args.ref_yolo_model,
            ref_yolo_conf=args.ref_yolo_conf,
            or_lost_thresh=args.or_lost_thresh,
            or_max_lost=args.or_max_lost,
            or_template=args.or_template,
            or_search=args.or_search,
            kf_history_len=args.kf_history_len,
            kf_max_predict_frames=args.kf_max_predict_frames,
            reacq_interval=args.reacq_interval,
            reacq_crop_expand=args.reacq_crop_expand,
            reacq_iou_with_pred=args.reacq_iou_with_pred,
            reacq_match_thresh=args.reacq_match_thresh,
            reacq_fullframe_after_n=args.reacq_fullframe_after_n,
            online_enhance=args.online_enhance,
            enhance_interval=args.enhance_interval,
            enhance_alpha=args.enhance_alpha,
            enhance_iou_thresh=args.enhance_iou_thresh,
            save_images_to=args.save_images_to,
            fps_log_frames=args.fps_log_frames,
            classes_prompts=classes_prompts.split() if isinstance(classes_prompts, str) else classes_prompts,
            manual_ref_paths=[p.strip() for p in str(args.ref_feature).split(',') if p.strip()],
            kf_mode='confaware'
        )

        try:
            _img_list2 = []
            for _pat in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
                _img_list2.extend(glob.glob(os.path.join(img_dir, _pat)))
            _imgs_sorted2 = sorted(set(_img_list2), key=_natural_key)
            first_img = cv2.imread(_imgs_sorted2[0], cv2.IMREAD_COLOR) if _imgs_sorted2 else None
            H0, W0 = first_img.shape[:2] if first_img is not None else (None, None)
        except Exception:
            H0, W0 = None, None

        try:
            if H0 is not None and W0 is not None:
                plot_gt_vs_pred_action_scatter(
                    gt_xywh, pr_xywh, (H0, W0),
                    save_path=os.path.join(out_dir, "gt_vs_action_scatter.pdf"),
                    title=None,
                    max_points_per_class=int(args.scatter_max_points),
                    font_ttf=args.font_ttf
                )
        except Exception:
            pass

        region = eval_region_with_nan_policy(gt_xywh, pr_xywh, states, img_hw=(H0, W0) if (H0 is not None and W0 is not None) else None, ignore_zero_gt=args.ignore_zero_gt)
        log = dict(sequence=rel, **region)
        all_logs.append(log)
        print(json.dumps(log, ensure_ascii=False))

    if len(all_logs):
        mean_region_acc = float(np.mean([x['region_acc'] for x in all_logs]))
        summary = dict(mean_region_acc=mean_region_acc, count=len(all_logs))
        print("==== SUMMARY ====")
        print(json.dumps(summary, ensure_ascii=False))
        with open(os.path.join(args.out_root, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(dict(per_seq=all_logs, summary=summary), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
