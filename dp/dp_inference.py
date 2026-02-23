"""Diffusion Policy 推理与动作转换"""
import os
import sys
import math
import cv2
import numpy as np
import torch


# 全局DP缓存
DP_CACHE = {'policy': None, 'cfg': None, 'device': None}


def get_dp_cache():
    """获取DP缓存
    
    Returns:
        dict: {'policy': ..., 'cfg': ..., 'device': ...}
    """
    return DP_CACHE


def _maybe_add_dp_repo_to_syspath(dp_repo_root: str):
    """将DP仓库路径添加到系统路径"""
    try:
        dp_dir = os.path.abspath(str(dp_repo_root))
        if os.path.isdir(dp_dir) and (dp_dir not in sys.path):
            sys.path.append(dp_dir)
    except Exception:
        pass


def _patch_normalizer_device(policy, device_str: str):
    """修补normalizer的设备问题"""
    try:
        import types
        def _wrap(self, x):
            y = _orig(x)
            def _to_dev(t):
                try:
                    return t.to(device_str)
                except Exception:
                    return t
            if isinstance(y, dict):
                return {k: _to_dev(v) for k, v in y.items()}
            return _to_dev(y)
        _orig = policy.normalizer.normalize
        policy.normalizer.normalize = types.MethodType(_wrap, policy.normalizer)
    except Exception:
        pass


def _load_dp_policy(ckpt_path: str, dp_repo_root: str):
    """加载DP策略模型
    
    Args:
        ckpt_path: checkpoint路径
        dp_repo_root: DP仓库根目录
        
    Returns:
        (policy, cfg, device): 策略模型、配置、设备
    """
    _maybe_add_dp_repo_to_syspath(dp_repo_root)
    try:
        import torch
        import dill
        import hydra
    except Exception as e:
        print(f"[DP] 依赖缺失（hydra/dill/torch）: {e}")
        return None, None, None
    try:
        payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill, map_location='cpu')
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        policy = workspace.model
        if getattr(cfg.training, 'use_ema', False):
            policy = workspace.ema_model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        policy.eval().to(device)
        # 推理参数
        try:
            policy.num_inference_steps = 16
            policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
        except Exception:
            pass
        _patch_normalizer_device(policy, str(device))
        try:
            print(f"[DP] Loaded policy: device={device}, horizon={getattr(policy,'horizon',None)}, "
                  f"n_obs_steps={getattr(policy,'n_obs_steps',None)}, "
                  f"n_action_steps={getattr(policy,'n_action_steps',None)}")
        except Exception:
            pass
        return policy, cfg, device
    except Exception as e:
        print(f"[DP] 加载失败: {e}")
        return None, None, None


def preload_dp_if_needed(cfg):
    """启动阶段预加载并暖机DP模型（仅在 dp_enable 为真时触发）
    
    Args:
        cfg: 配置字典
    """
    try:
        if DP_CACHE.get('policy') is not None:
            return
        if not bool(cfg.get('dp_enable', False)):
            return
        ckpt = cfg.get('dp_checkpoint', '')
        if not ckpt:
            return
        pol, pcfg, dev = _load_dp_policy(ckpt, cfg.get('dp_repo_root', ''))
        if pol is None:
            return
        DP_CACHE.update({'policy': pol, 'cfg': pcfg, 'device': dev})

        # 构造一次虚拟观测做"暖机"
        H = int(cfg.get('desired_height', 240))
        W = int(cfg.get('desired_width', 360))
        dummy = np.zeros((H, W, 3), dtype=np.uint8)
        bx, by, bw, bh = W * 0.4, H * 0.4, W * 0.2, H * 0.2
        obs = build_dp_obs_from_frame(dummy, (bx, by, bw, bh), pcfg, dev)
        with torch.no_grad():
            _ = pol.predict_action(obs)
        try:
            print("[DP] Preload OK: model loaded and warmed up.")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[DP] 预加载失败: {e}")
        except Exception:
            pass


def build_dp_obs_from_frame(frame_bgr, bbox_xywh, dp_cfg, device):
    """从帧和bbox构建DP观测
    
    Args:
        frame_bgr: BGR图像
        bbox_xywh: 边界框 (x, y, w, h)
        dp_cfg: DP配置
        device: 设备
        
    Returns:
        dict: {'image': tensor, 'bbox': tensor}
    """
    try:
        from omegaconf import OmegaConf
        sm = OmegaConf.to_container(dp_cfg.task.shape_meta, resolve=True)
        C, Ht, Wt = int(sm['obs']['image']['shape'][0]), int(sm['obs']['image']['shape'][1]), int(sm['obs']['image']['shape'][2])
    except Exception:
        C, Ht, Wt = 3, 240, 360
    img = cv2.resize(frame_bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # C,H,W
    H0, W0 = frame_bgr.shape[:2]
    x, y, w, h = [float(max(0, v)) for v in bbox_xywh]
    w = max(1.0, w)
    h = max(1.0, h)
    bbox = np.array([x / max(1.0, W0), y / max(1.0, H0), w / max(1.0, W0), h / max(1.0, H0)], dtype=np.float32)
    try:
        To = int(dp_cfg.n_obs_steps)
    except Exception:
        To = 1
    image = np.tile(img[None, ...], (To, 1, 1, 1))
    bboxs = np.tile(bbox[None, :], (To, 1))
    obs = {
        'image': torch.from_numpy(image).unsqueeze(0).to(device),  # (1,To,C,H,W)
        'bbox':  torch.from_numpy(bboxs).unsqueeze(0).to(device)   # (1,To,4)
    }
    return obs


def dp_action_to_tello_vel(act3, cfg):
    """将DP动作转换为Tello速度命令
    
    Args:
        act3: 动作向量 [dx, dy, dz]
        cfg: 配置字典
        
    Returns:
        (vy, vz, yaw): 速度命令
    """
    dx, dy, dz = float(act3[0]), float(act3[1]), float(act3[2])
    # 角度法：将 (dx,dy) 转为偏航角变化 φ，再按 dt 转为 yaw_rate；前进速度取合速度
    dt = float(cfg['dp_dt']) if cfg.get('dp_dt') is not None else float(cfg.get('ctrl_dt', 0.1))
    inv_dt = 1.0 / max(1e-6, dt)
    phi = math.atan2(dy, dx)              # dy>0 → 右偏；按题意：yaw_rate>0 为顺时针（右转）
    if dy < 0:
        yaw = phi * inv_dt * 30                    # 偏航速度（按 dt 转换）
    else:
        yaw = phi * inv_dt                   # 偏航速度（按 dt 转换）
    speed = math.hypot(dx, dy) * inv_dt   # 前进速度（合速度）
    vy = speed
    vz = dz * inv_dt
    # 限幅：yaw 建议不超过 ±60，线速度沿用 ctrl_max_speed
    yaw_max = float(cfg.get('dp_yaw_max', 60.0))
    lin_max = float(cfg.get('ctrl_max_speed', 40.0))
    yaw = float(np.clip(yaw, -yaw_max, yaw_max))
    vy = float(np.clip(vy, -lin_max, lin_max))
    vz = float(np.clip(vz, -lin_max, lin_max))
    return vy, vz, yaw




