"""卡尔曼滤波相关工具函数"""
import numpy as np


def init_pkf_bbox(dt=1.0):
    """初始化 PyKalman BBox 滤波器（用于置信度感知平滑）
    
    Args:
        dt: 时间步长
        
    Returns:
        PKF 滤波器实例
    """
    from pykalman import KalmanFilter as PKF
    dim = 4  # x, y, w, h
    F = np.zeros((dim * 2, dim * 2))
    F[:dim, :dim] = np.eye(dim)
    F[:dim, dim:] = np.eye(dim) * float(dt)
    F[dim:, dim:] = np.eye(dim)
    H = np.hstack([np.eye(dim), np.zeros((dim, dim))])
    pkf = PKF(
        transition_matrices=F,
        observation_matrices=H,
        transition_covariance=0.01 * np.eye(dim * 2),
        initial_state_mean=np.zeros(dim * 2),
        initial_state_covariance=0.1 * np.eye(dim * 2)
    )
    return pkf


def conf_to_noise(conf, max_noise=1.0, min_noise=0.0, k=17.0, center=0.4):
    """将置信度转为观测噪声（低置信度→高噪声）
    
    Args:
        conf: 置信度值 [0, 1]
        max_noise: 最大噪声
        min_noise: 最小噪声
        k: sigmoid 斜率参数
        center: sigmoid 中心点
        
    Returns:
        观测噪声值
    """
    c = float(np.clip(conf, 0.0, 1.0))
    return float(max_noise / (1.0 + np.exp(k * (c - center))) + min_noise)


def estimate_velocity_robustly(center_history):
    """从历史中心点估计速度（线性拟合）
    
    Args:
        center_history: 历史中心点列表 [(cx, cy, a, h), ...]
        
    Returns:
        (vx, vy, va, vh): 速度估计
    """
    if len(center_history) < 2:
        return 0.0, 0.0, 0.0, 0.0
    x0, y0, a0, h0 = center_history[0]
    x1, y1, a1, h1 = center_history[-1]
    vx = (x1 - x0) / max(1, len(center_history) - 1)
    vy = (y1 - y0) / max(1, len(center_history) - 1)
    va = 0.0
    vh = 0.0
    return vx, vy, va, vh


