from typing import Dict
import os
import re
import json
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from threadpoolctl import threadpool_limits

class UnrealBBoxImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path,
            shape_meta: dict,
            horizon=16,
            pad_before=0,
            pad_after=0,
            n_obs_steps=2,
            seed=42,
            val_ratio=0.02,
            max_train_episodes=None
            ):
        super().__init__()
        # img: (T,H,W,C) uint8
        # bbox: (T,4) float32 normalized [0,1]
        # action: (T,D) float32
        # 使用磁盘后端懒加载，避免将整库复制到内存导致初始化卡顿/内存占用过大
        self.replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode='r')
        self.zarr_path = zarr_path
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # 仅对观测（img、bbox）加载前 n_obs_steps，减少 I/O 与内存
        key_first_k = {'img': n_obs_steps, 'bbox': n_obs_steps, 'uav_pose_before': 1}

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        # 懒加载 zarr occ2d 句柄（避免初始化时解压整块）
        self._occ2d_handles = None

        # 建立 step->episode 的映射，便于按样本索引找到对应 episode
        try:
            self._step_to_episode_idx = self.replay_buffer.get_episode_idxs()
        except Exception:
            self._step_to_episode_idx = None

        # 预加载每个 episode 的 2D 占用图 (occ, meta)
        self.occupancy_maps = self._load_occupancy_maps()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k={'img': self.n_obs_steps, 'bbox': self.n_obs_steps}
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # image range [0,1] expected by imagenet_norm
        normalizer['image'] = get_image_range_normalizer()
        # bbox already in [0,1], identity normalizer
        from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
        normalizer['bbox'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # 仅取前 n_obs_steps 的观测，动作保留全长（由策略内部处理 To 与 T 的对应关系）
        T_slice = slice(self.n_obs_steps)
        # imgs: (T,H,W,C) -> (T,C,H,W) and [0,1]
        image = np.moveaxis(sample['img'][T_slice], -1, 1) / 255.0
        bbox = sample['bbox'][T_slice].astype(np.float32)
        action = sample['action'].astype(np.float32)

        data = {
            'obs': {
                'image': image,  # T, 3, H, W
                'bbox': bbox,    # T, 4
            },
            'action': action    # T, D
        }
        # 可选：无人机起始绝对位姿（若存在则加入 obs）
        try:
            if 'uav_pose_before' in sample:
                upb = sample['uav_pose_before']
                if upb is not None and len(upb) > 0:
                    # 取序列首帧（或被 pad 的首帧）
                    data['obs']['uav_pose_before'] = upb[0].astype(np.float32)
        except Exception:
            pass
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 限制底层线程，避免过多并行带来的抖动
        threadpool_limits(1)
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # 附加 episode 级占用图（不转为 torch，保持 Python 对象 (occ, meta)）
        try:
            occ_tuple = None
            if self._step_to_episode_idx is not None:
                buffer_start_idx = int(self.sampler.indices[idx][0])
                ep_idx = int(self._step_to_episode_idx[buffer_start_idx])
                # 先尝试预加载列表（磁盘读取结果）
                if isinstance(self.occupancy_maps, list) and ep_idx < len(self.occupancy_maps):
                    occ_tuple = self.occupancy_maps[ep_idx]
                # 懒加载 zarr meta（只读取该 episode 的一小块）
                if occ_tuple is None and self._occ2d_handles is not None:
                    try:
                        occ2d = self._occ2d_handles['occ2d']
                        origin = self._occ2d_handles['origin']
                        cs = self._occ2d_handles['cell_size']
                        valid = self._occ2d_handles.get('valid', None)
                        if valid is None or bool(valid[ep_idx]):
                            occ_i = occ2d[ep_idx, ...]
                            meta_i = {
                                'origin': [float(origin[ep_idx, 0]), float(origin[ep_idx, 1])],
                                'cell_size': float(cs[ep_idx]),
                                'shape': tuple(occ_i.shape)
                            }
                            occ_tuple = (np.asarray(occ_i), meta_i)
                    except Exception:
                        occ_tuple = None
            # 转 torch 仅对数值型
            torch_data = dict_apply(data, torch.from_numpy)
            if occ_tuple is not None:
                torch_data['occupancy_map'] = occ_tuple
            return torch_data
        except Exception:
            torch_data = dict_apply(data, torch.from_numpy)
            return torch_data

    # ----------------- helper -----------------
    def _load_occupancy_maps(self):
        """
        优先从磁盘扫描每个 episode 的 occupancy2d.npz；若磁盘未找到且 zarr meta
        中存在 occ2d，则记录 zarr 句柄，采用按需懒加载。返回列表长度与 n_episodes 对齐。
        磁盘布局兼容：
        - <base>/epXXXX/occupancy2d.npz
        - <base>/episodes/epXXXX/occupancy2d.npz
        若无法匹配 episode 数量，则返回 [None] * n_episodes。
        """
        n_ep = int(self.replay_buffer.n_episodes)
        # 1) 先磁盘扫描
        base = os.path.abspath(os.path.expanduser(str(self.zarr_path)))
        # 如果 zarr 目录本身是以 .zarr 结尾，向上取其父目录更合理
        base_dir = base
        try:
            if os.path.isdir(base) and base.endswith('.zarr'):
                base_dir = os.path.dirname(base)
        except Exception:
            pass

        cand_paths = []
        # 先查找 base_dir/epXXXX/occupancy2d.npz
        for name in sorted(os.listdir(base_dir)):
            ep_dir = os.path.join(base_dir, name)
            if os.path.isdir(ep_dir) and re.match(r'^ep\d+$', name):
                p = os.path.join(ep_dir, 'occupancy2d.npz')
                if os.path.isfile(p):
                    cand_paths.append(p)
        # 若没找到，再尝试 base_dir/episodes/epXXXX/occupancy2d.npz
        if len(cand_paths) == 0:
            epi_root = os.path.join(base_dir, 'episodes')
            if os.path.isdir(epi_root):
                for name in sorted(os.listdir(epi_root)):
                    ep_dir = os.path.join(epi_root, name)
                    if os.path.isdir(ep_dir) and re.match(r'^ep\d+$', name):
                        p = os.path.join(ep_dir, 'occupancy2d.npz')
                        if os.path.isfile(p):
                            cand_paths.append(p)

        # 解析 ep 索引
        def parse_ep_index(path):
            m = re.search(r'ep(\d+)', path)
            return int(m.group(1)) if m else None

        occ_by_ep = [None] * n_ep
        if len(cand_paths) > 0:
            # 尝试按 epXXXX 对齐
            for p in cand_paths:
                ei = parse_ep_index(p)
                if ei is None:
                    continue
                if 0 <= ei < n_ep:
                    try:
                        npz = np.load(p, allow_pickle=True)
                        occ = npz.get('occ') if hasattr(npz, 'get') else npz['occ']
                        meta = npz.get('meta') if hasattr(npz, 'get') else npz['meta']
                        # meta 可能以 object array 存储
                        if isinstance(meta, np.ndarray) and meta.dtype == object and meta.size == 1:
                            meta = meta.item()
                        # 若是 JSON 字符串
                        if isinstance(meta, (bytes, str)):
                            try:
                                meta = json.loads(meta)
                            except Exception:
                                pass
                        occ_by_ep[ei] = (occ, meta)
                    except Exception:
                        occ_by_ep[ei] = None

            # 若全部填充失败，尝试长度匹配（按排序顺序）
            if all(x is None for x in occ_by_ep):
                cand_paths.sort()
                if len(cand_paths) == n_ep:
                    for i, p in enumerate(cand_paths):
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
                            occ_by_ep[i] = (occ, meta)
                        except Exception:
                            occ_by_ep[i] = None

        if any(x is not None for x in occ_by_ep):
            return occ_by_ep

        # 2) 磁盘未找到，再尝试记录 zarr meta 句柄（懒加载）
        try:
            meta_group = self.replay_buffer.meta
            if ('occ2d' in meta_group) and ('occ2d_origin' in meta_group) and ('occ2d_cell_size' in meta_group):
                occ2d = meta_group['occ2d']
                ori = meta_group['occ2d_origin'][:]
                cs = meta_group['occ2d_cell_size'][:]
                valid = meta_group['occ2d_valid'][:] if 'occ2d_valid' in meta_group else None
                if occ2d.shape[0] == n_ep and ori.shape[0] == n_ep and cs.shape[0] == n_ep:
                    self._occ2d_handles = {
                        'occ2d': occ2d,
                        'origin': ori,
                        'cell_size': cs,
                        'valid': valid
                    }
                    return [None] * n_ep
        except Exception:
            pass

        return [None] * n_ep