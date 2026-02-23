#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import glob
from typing import List, Tuple, Iterator, Optional, Dict
import re
from collections import defaultdict

import cv2
import numpy as np
import zarr

# Allow running from anywhere
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

DP_DIR = os.path.join(ROOT_DIR, 'diffusion_policy-main')
if DP_DIR not in sys.path:
    sys.path.append(DP_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer


def _iter_records_from_labels(input_root: str) -> Iterator[Tuple[str, dict]]:
    """Yield (episode_dir, record_dict) from labels.jsonl under input_root."""
    pattern = os.path.join(input_root, '**', 'labels.jsonl')
    for labels_path in glob.glob(pattern, recursive=True):
        ep_dir = os.path.dirname(labels_path)
        try:
            with open(labels_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict) and 'frame' in rec and 'traj_xyz' in rec and (
                            'bbox_xywh' in rec or 'bbox_xyxy' in rec or 'bbox' in rec
                        ):
                            yield ep_dir, rec
                    except Exception:
                        continue
        except Exception:
            continue


# Fallback: legacy per-sample json files
def _find_all_json_records(input_root: str) -> List[str]:
    # recursively search for json files that look like per-sample records
    patterns = [
        os.path.join(input_root, '**', '*.json'),
        os.path.join(input_root, '*.json')
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    # filter out non-sample json (e.g., settings). Heuristics: must contain keys we need.
    result = []
    for f in files:
        try:
            with open(f, 'r') as fp:
                rec = json.load(fp)
            if isinstance(rec, dict) and (
                ('frame' in rec) or ('img' in rec)
            ) and (
                ('bbox_xywh' in rec) or ('bbox_xyxy' in rec) or ('bbox' in rec)
            ) and (
                ('traj_xyz' in rec) or ('trajectory' in rec)
            ):
                result.append(f)
        except Exception:
            continue
    return sorted(result)


def _load_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"image not found: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if image_size is not None:
        w, h = image_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _normalize_bbox_xywh(bbox_xywh: List[float], img_w: int, img_h: int) -> np.ndarray:
    x, y, w, h = bbox_xywh
    x = float(np.clip(x, 0, img_w - 1))
    y = float(np.clip(y, 0, img_h - 1))
    w = float(max(0.0, min(w, img_w)))
    h = float(max(0.0, min(h, img_h)))
    return np.array([x / img_w, y / img_h, w / img_w, h / img_h], dtype=np.float32)


def _normalize_bbox_xyxy_to_xywh(bbox_xyxy: List[float], img_w: int, img_h: int) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    x1 = float(np.clip(x1, 0, img_w - 1))
    y1 = float(np.clip(y1, 0, img_h - 1))
    x2 = float(np.clip(x2, 0, img_w - 1))
    y2 = float(np.clip(y2, 0, img_h - 1))
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return np.array([x1 / img_w, y1 / img_h, w / img_w, h / img_h], dtype=np.float32)


def _extract_bbox_norm(rec: dict, img_w: int, img_h: int) -> np.ndarray:
    if 'bbox_xywh' in rec:
        return _normalize_bbox_xywh(rec['bbox_xywh'], img_w=img_w, img_h=img_h)
    if 'bbox_xyxy' in rec:
        return _normalize_bbox_xyxy_to_xywh(rec['bbox_xyxy'], img_w=img_w, img_h=img_h)
    if 'bbox' in rec:
        b = rec['bbox']
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if b.shape[0] != 4:
            raise ValueError('bbox must have 4 elements')
        # Heuristic: if already normalized
        if np.max(b) <= 1.5:
            return b.astype(np.float32)
        # assume pixel xywh
        return _normalize_bbox_xywh(b.tolist(), img_w=img_w, img_h=img_h)
    raise KeyError('bbox not found')


def _ensure_action_traj(traj_xyz: np.ndarray, horizon: int) -> np.ndarray:
    # traj_xyz expected shape (T, D), pad/trim to horizon
    traj = np.asarray(traj_xyz, dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"traj_xyz must be 2D (T, D). Got shape={traj.shape}")
    T, D = traj.shape
    if T == horizon:
        return traj
    if T > horizon:
        return traj[:horizon]
    # pad with last row
    pad = np.repeat(traj[-1:], repeats=horizon - T, axis=0)
    return np.concatenate([traj, pad], axis=0)


def convert(
    input_root: str,
    output_zarr: str,
    image_width: int = 96,
    image_height: int = 96,
    horizon: int = 8,
    max_per_level: int = None,
    per_level_counts: Optional[Dict[str, int]] = None,
    level_regex: str = r'level\d+',
):
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)

    # build in-memory zarr and save at the end to avoid in-place overwrite issues
    rb = ReplayBuffer.create_empty_zarr()

    # prefer labels.jsonl format
    records = list(_iter_records_from_labels(input_root))
    legacy_json_files: List[str] = []
    if len(records) == 0:
        legacy_json_files = _find_all_json_records(input_root)
        if len(legacy_json_files) == 0:
            raise RuntimeError(f"no sample found under: {input_root}")

    # cache for occ2d per-episode if available on disk
    def _load_occ2d_for_episode(ep_dir: str):
        import os, json, numpy as np
        # support epXXXX/occupancy2d.npz
        p = os.path.join(ep_dir, 'occupancy2d.npz')
        if not os.path.isfile(p):
            # support episodes/epXXXX/occupancy2d.npz
            parent = os.path.dirname(ep_dir)
            name = os.path.basename(ep_dir)
            cand = os.path.join(parent, 'episodes', name, 'occupancy2d.npz')
            p = cand if os.path.isfile(cand) else None
        if p is None or (not os.path.isfile(p)):
            return None
        try:
            npz = np.load(p, allow_pickle=True)
            occ = npz.get('occ') if hasattr(npz, 'get') else npz['occ']
            meta = npz.get('meta') if hasattr(npz, 'get') else npz['meta']
            if isinstance(meta, np.ndarray) and meta.dtype == object and meta.size == 1:
                meta = meta.item()
            if isinstance(meta, (bytes, str)):
                try:
                    meta = json.loads(meta)
                except Exception:
                    pass
            return occ, meta
        except Exception:
            return None

    occ_list = []  # list of (occ, meta) or None, per-episode

    # setup per-level limiter
    pat = re.compile(level_regex) if level_regex else None
    count_by_level = defaultdict(int)
    def _extract_level_name(path: str) -> str:
        try:
            for seg in os.path.normpath(path).split(os.sep):
                if pat and pat.fullmatch(seg):
                    return seg
        except Exception:
            pass
        return None
    def _get_limit_for_level(level_name: Optional[str]) -> Optional[int]:
        if (per_level_counts is not None) and (level_name is not None):
            if level_name in per_level_counts:
                return int(per_level_counts[level_name])
        if max_per_level is not None:
            return int(max_per_level)
        return None

    if records:
        for ep_dir, rec in records:
            # per-level limiting
            level_name = _extract_level_name(ep_dir)
            _limit = _get_limit_for_level(level_name)
            if (_limit is not None) and (level_name is not None) and (count_by_level[level_name] >= _limit):
                continue
            img_name = rec.get('frame') or rec.get('img')
            if img_name is None:
                continue
            img_path = img_name if os.path.isabs(img_name) else os.path.join(ep_dir, img_name)
            img = _load_image(img_path, image_size=(image_width, image_height))
            H, W = img.shape[:2]

            bbox_norm_xywh = _extract_bbox_norm(rec, img_w=W, img_h=H)

            traj_xyz = rec.get('traj_xyz') or rec.get('trajectory')
            if traj_xyz is None:
                continue
            traj = _ensure_action_traj(np.asarray(traj_xyz, dtype=np.float32), horizon=horizon)

            imgs = np.repeat(img[None, ...], repeats=horizon, axis=0)
            bboxes = np.repeat(bbox_norm_xywh[None, ...], repeats=horizon, axis=0)
            actions = traj

            data = {
                'img': imgs.astype(np.uint8),
                'bbox': bboxes.astype(np.float32),
                'action': actions.astype(np.float32)
            }
            rb.add_episode(data)
            if level_name is not None:
                count_by_level[level_name] += 1
            # try to load occ2d for this episode
            occ_item = _load_occ2d_for_episode(ep_dir)
            occ_list.append(occ_item)
    else:
        # legacy: per-sample json files
        ep_seen = dict()  # map ep_dir->episode index
        for jf in legacy_json_files:
            with open(jf, 'r') as fp:
                rec = json.load(fp)
            img_rel = rec.get('frame') or rec.get('img')
            if img_rel is None:
                continue
            ep_dir = os.path.dirname(jf)
            # per-level limiting
            level_name = _extract_level_name(ep_dir)
            _limit = _get_limit_for_level(level_name)
            if (_limit is not None) and (level_name is not None) and (count_by_level[level_name] >= _limit):
                continue
            img_path = img_rel if os.path.isabs(img_rel) else os.path.join(ep_dir, img_rel)
            img = _load_image(img_path, image_size=(image_width, image_height))
            H, W = img.shape[:2]

            bbox_norm_xywh = _extract_bbox_norm(rec, img_w=W, img_h=H)

            traj_xyz = rec.get('traj_xyz') or rec.get('trajectory')
            if traj_xyz is None:
                continue
            traj = _ensure_action_traj(np.asarray(traj_xyz, dtype=np.float32), horizon=horizon)

            imgs = np.repeat(img[None, ...], repeats=horizon, axis=0)
            bboxes = np.repeat(bbox_norm_xywh[None, ...], repeats=horizon, axis=0)
            actions = traj

            data = {
                'img': imgs.astype(np.uint8),
                'bbox': bboxes.astype(np.float32),
                'action': actions.astype(np.float32)
            }
            rb.add_episode(data)
            if level_name is not None:
                count_by_level[level_name] += 1
            if ep_dir not in ep_seen:
                occ_item = _load_occ2d_for_episode(ep_dir)
                occ_list.append(occ_item)
                ep_seen[ep_dir] = True

    # write meta for convenience
    # write meta for convenience
    rb.update_meta({
        'image_shape': np.array([image_height, image_width, 3], dtype=np.int64),
        'horizon': np.array(horizon, dtype=np.int64)
    })

    # persist occ2d arrays into zarr meta if available
    try:
        n_ep = rb.n_episodes
        occ2d = None
        origin = None
        cell = None
        valid = None
        if len(occ_list) == n_ep:
            # allocate arrays based on first non-None
            for item in occ_list:
                if item is not None:
                    occ0, meta0 = item
                    H0, W0 = occ0.shape
                    occ2d = np.zeros((n_ep, H0, W0), dtype=np.uint8)
                    origin = np.zeros((n_ep, 2), dtype=np.float32)
                    cell = np.zeros((n_ep,), dtype=np.float32)
                    valid = np.zeros((n_ep,), dtype=np.uint8)
                    break
            if occ2d is not None:
                for i, item in enumerate(occ_list):
                    if item is None:
                        continue
                    occ_i, meta_i = item
                    try:
                        occ2d[i, ...] = occ_i.astype(np.uint8)
                        ox, oy = float(meta_i['origin'][0]), float(meta_i['origin'][1])
                        origin[i, 0] = ox; origin[i, 1] = oy
                        cell[i] = float(meta_i.get('cell_size', 50.0))
                        valid[i] = 1
                    except Exception:
                        continue
                rb.update_meta({
                    'occ2d': occ2d,
                    'occ2d_origin': origin,
                    'occ2d_cell_size': cell,
                    'occ2d_valid': valid
                })
    except Exception:
        pass

    rb.save_to_path(output_zarr)
    print(f"saved zarr: {output_zarr}")
    print(f"episodes: {rb.n_episodes}, steps: {rb.n_steps}")


def main():
    parser = argparse.ArgumentParser(description='Convert UnrealCV collected samples to Zarr ReplayBuffer')
    parser.add_argument('--input-root', type=str, required=True,
                        help='Root directory containing episode folders (with labels.jsonl)')
    parser.add_argument('--output-zarr', type=str, required=True,
                        help='Output zarr directory path (will be overwritten)')
    parser.add_argument('--image-size', type=int, nargs=2, default=[96, 96],
                        help='Resize images to (W H)')
    parser.add_argument('--horizon', type=int, default=8,
                        help='Episode length (must match planned traj_horizon)')
    parser.add_argument('--max-per-level', type=int, default=None,
                        help='max number of samples per level (e.g., level3/level4)')
    parser.add_argument('--per-level-counts', type=str, default=None,
                        help='per-level limits, e.g., "level3=100,level4=50"')
    parser.add_argument('--level-regex', type=str, default=r'level\d+',
                        help='regex used to identify level directory name (fullmatch)')

    args = parser.parse_args()
    w, h = args.image_size
    # parse per-level-counts
    plc = None
    if args.per_level_counts:
        plc = {}
        for seg in args.per_level_counts.split(','):
            seg = seg.strip()
            if not seg:
                continue
            if '=' not in seg:
                continue
            k, v = seg.split('=', 1)
            k = k.strip(); v = v.strip()
            try:
                plc[k] = int(v)
            except Exception:
                continue
    convert(
        input_root=os.path.abspath(args.input_root),
        output_zarr=os.path.abspath(args.output_zarr),
        image_width=w,
        image_height=h,
        horizon=args.horizon,
        max_per_level=args.max_per_level,
        per_level_counts=plc,
        level_regex=args.level_regex,
    )


if __name__ == '__main__':
    main() 