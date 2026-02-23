#!/usr//bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import time
import json
import argparse
import random
import numpy as np
import copy
import matplotlib
# Force non-interactive backend to avoid GUI blocking/crashes under headless/VSCode
try:
    if not os.environ.get('MPLBACKEND'):
        matplotlib.use('Agg', force=True)
except Exception:
    pass
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scipy import ndimage

from gym_unrealcv.envs.utils import env_unreal, misc
from gym_unrealcv.envs.utils.unrealcv_basic import UnrealCv
from gym_unrealcv.envs.tracking.interaction import Tracking

try:
    import cv2
except Exception:
    cv2 = None

# ------------------------ Utils ------------------------

def world_to_grid(x: float, y: float, meta: dict) -> Tuple[int, int]:
    i = int((y - meta['origin'][1]) // meta['cell_size'])
    j = int((x - meta['origin'][0]) // meta['cell_size'])
    return i, j

def grid_to_world(i: int, j: int, meta: dict) -> Tuple[float, float]:
    x = meta['origin'][0] + (j + 0.5) * meta['cell_size']
    y = meta['origin'][1] + (i + 0.5) * meta['cell_size']
    return x, y

def _rotz_deg(yaw_deg: float) -> np.ndarray:
    """Rotation matrix Rz(yaw) with yaw in degrees."""
    yaw = np.deg2rad(float(yaw_deg))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

def world_abs_to_body0(traj_xyz: np.ndarray, yaw0_deg: float) -> np.ndarray:
    """
    Rotate a (T,3) trajectory that is already relative-to-start in world frame
    into the starting body frame (Body0) using the initial facing yaw yaw0_deg.
    """
    if traj_xyz is None:
        return traj_xyz
    arr = np.asarray(traj_xyz, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return arr
    R = _rotz_deg(-float(yaw0_deg))  # world -> body0
    return (R @ arr.T).T

def _yaw_away_sign_by_fan(occ: np.ndarray, meta: dict, uav_xy: Tuple[float,float], yaw_deg: float) -> int:
    """
    扇形前向扫描，返回 +1=右转, -1=左转, 0=无法判定。
    内部常量（保持简洁）：
      - max_range = 800.0（世界单位）
      - sweep_deg = 35.0（左右各扫角）
      - num_rays = 9（每侧 4 条 + 4 条，忽略正中）
      - step = cell_size * 0.5
    """
    import math, random
    H, W = occ.shape
    cs = float(meta['cell_size'])
    step = max(1.0, cs * 0.5)
    max_range = 800.0
    sweep_deg = 35.0
    num_rays = 9

    ux, uy = float(uav_xy[0]), float(uav_xy[1])
    k = max(1, int(num_rays // 2))
    offs = [-(i+1) * (sweep_deg / k) for i in range(k)] + [(i+1) * (sweep_deg / k) for i in range(k)]

    def first_hit_distance(angle_deg: float) -> float:
        ang = math.radians(angle_deg)
        dx, dy = math.cos(ang), math.sin(ang)
        d = 0.0
        while d <= max_range:
            x = ux + dx * d
            y = uy + dy * d
            i, j = world_to_grid(x, y, meta)
            if not (0 <= i < H and 0 <= j < W):
                return d
            if occ[i, j] == 1:
                return d
            d += step
        return max_range

    left_clear, right_clear = 0.0, 0.0
    for off in offs:
        dist = first_hit_distance(yaw_deg + off)
        if off < 0:
            left_clear = max(left_clear, dist)
        else:
            right_clear = max(right_clear, dist)

    if abs(right_clear - left_clear) < 1e-6:
        return 1 if random.random() < 0.5 else -1
    return 1 if right_clear > left_clear else -1

def astar(occ: np.ndarray, start_xy: Tuple[float, float], goal_xy: Tuple[float, float], meta: dict) -> List[Tuple[float, float]]:
    from heapq import heappush, heappop
    H, W = occ.shape
    si, sj = world_to_grid(start_xy[0], start_xy[1], meta)
    gi, gj = world_to_grid(goal_xy[0], goal_xy[1], meta)
    if not (0 <= si < H and 0 <= sj < W and 0 <= gi < H and 0 <= gj < W):
        return []
    if occ[gi, gj] == 1:
        return []
    ds = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    g = {(si,sj):0.0}
    parent = {(si,sj):None}
    pq = [(0.0, (si,sj))]
    while pq:
        _, cur = heappop(pq)
        if cur == (gi,gj):
            path = []
            t = cur
            while t is not None:
                path.append(grid_to_world(t[0], t[1], meta))
                t = parent[t]
            return path[::-1]
        i,j = cur
        for di,dj in ds:
            ni, nj = i+di, j+dj
            if not (0 <= ni < H and 0 <= nj < W) or occ[ni, nj] == 1:
                continue
            w = math.hypot(di, dj)
            cand = g[cur] + w
            if (ni,nj) not in g or cand < g[(ni,nj)]:
                g[(ni,nj)] = cand
                parent[(ni,nj)] = cur
                h = math.hypot(ni-gi, nj-gj)
                heappush(pq, (cand+h, (ni,nj)))
    return []

def shortcut_path(path: List[Tuple[float,float]], occ: np.ndarray, meta: dict, max_iter: int = 200) -> List[Tuple[float,float]]:
    if len(path) < 3:
        return path
    def free_segment(p0, p1) -> bool:
        # 采样段上的若干点检查是否命中障碍
        steps = max(2, int(math.hypot(p1[0]-p0[0], p1[1]-p0[1]) / meta['cell_size']))
        for k in range(steps+1):
            x = p0[0] + (p1[0]-p0[0]) * k/steps
            y = p0[1] + (p1[1]-p0[1]) * k/steps
            i,j = world_to_grid(x,y,meta)
            if not (0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]) or occ[i,j] == 1:
                return False
        return True
    pts = path[:]
    it = 0
    while it < max_iter and len(pts) > 2:
        i = random.randint(0, len(pts)-3)
        j = random.randint(i+2, len(pts)-1)
        if free_segment(pts[i], pts[j]):
            pts = pts[:i+1] + pts[j:]
        it += 1
    return pts

def los_blocked_2d(p0: Tuple[float,float], p1: Tuple[float,float], occ: np.ndarray, meta: dict) -> bool:
    """
    用占用图近似判断 p0->p1 之间是否被障碍阻断（有任意采样点落在 occ==1）。
    """
    steps = max(2, int(math.hypot(p1[0]-p0[0], p1[1]-p0[1]) / max(1, int(meta['cell_size']))))
    for k in range(steps+1):
        x = p0[0] + (p1[0]-p0[0]) * (k/float(steps))
        y = p0[1] + (p1[1]-p0[1]) * (k/float(steps))
        i, j = world_to_grid(x, y, meta)
        if not (0 <= i < occ.shape[0] and 0 <= j < occ.shape[1]):
            return True  # 视作不通
        if occ[i, j] == 1:
            return True
    return False

from collections import deque

def reachable_2d(occ: np.ndarray, start_ij: Tuple[int,int], goal_ij: Tuple[int,int]) -> bool:
    H, W = occ.shape
    si, sj = int(start_ij[0]), int(start_ij[1])
    gi, gj = int(goal_ij[0]), int(goal_ij[1])
    if not (0 <= si < H and 0 <= sj < W and 0 <= gi < H and 0 <= gj < W):
        return False
    if occ[si, sj] == 1 or occ[gi, gj] == 1:
        return False
    if (si, sj) == (gi, gj):
        return True
    q = deque([(si, sj)])
    vis = set([(si, sj)])
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while q:
        i, j = q.popleft()
        for di, dj in dirs:
            ni, nj = i+di, j+dj
            if not (0 <= ni < H and 0 <= nj < W):
                continue
            if occ[ni, nj] == 1:
                continue
            if (ni, nj) in vis:
                continue
            if (ni, nj) == (gi, gj):
                return True
            vis.add((ni, nj))
            q.append((ni, nj))
    return False

def snap_to_free_xy(occ: np.ndarray, meta: dict, x: float, y: float, max_radius_cells: int = 3) -> Tuple[float,float]:
    """If (x,y) lands in obstacle, snap to nearest free cell within a small radius."""
    H, W = occ.shape
    i0, j0 = world_to_grid(x, y, meta)
    if 0 <= i0 < H and 0 <= j0 < W and occ[i0, j0] == 0:
        return float(x), float(y)
    # ring search
    for r in range(1, max(1, int(max_radius_cells)) + 1):
        for di in range(-r, r+1):
            for dj in range(-r, r+1):
                if abs(di) != r and abs(dj) != r:
                    continue  # only check ring
                ii, jj = i0 + di, j0 + dj
                if 0 <= ii < H and 0 <= jj < W and occ[ii, jj] == 0:
                    xw, yw = grid_to_world(int(ii), int(jj), meta)
                    return float(xw), float(yw)
    # fallback original
    return float(x), float(y)

################################################################## 
    # 在occ map中通过多次随机采点的方式，找到2个能否符合约束的点
##################################################################
def find_occluded_initial_pair_random(occ: np.ndarray, meta: dict, follow_dist: float, max_trials: int = 500) -> Tuple[Optional[Tuple[float,float]], Optional[Tuple[float,float]]]:
    """
    在可通行区域内随机采样一对 (human_xy, uav_xy)，满足：
      - 距离约束：dist ∈ [follow_dist/2, follow_dist*2]
      - 遮挡约束：los_blocked_2d(uav_xy, human_xy) == True
    找到则返回 ((hx, hy), (ux, uy))，否则返回 (None, None)。
    """
    xmin, xmax, ymin, ymax = meta['reset_area'][:4]
    H, W = occ.shape
    # 预先收集自由格坐标索引，提升采样效率
    free_cells = np.argwhere(occ == 0)
    if free_cells.size == 0:
        return (None, None)
    for _ in range(max_trials):
        idx_h = np.random.randint(0, len(free_cells))
        ih, jh = free_cells[idx_h]
        hx, hy = grid_to_world(int(ih), int(jh), meta)
        # UAV 候选
        idx_u = np.random.randint(0, len(free_cells))
        iu, ju = free_cells[idx_u]
        ux, uy = grid_to_world(int(iu), int(ju), meta)
        d = math.hypot(hx - ux, hy - uy)
        if d < max(1.0, float(follow_dist) / 2.0) or d > float(follow_dist) * 2.0:
            continue
        if los_blocked_2d((ux, uy), (hx, hy), occ, meta):
            return (float(hx), float(hy)), (float(ux), float(uy))
    return (None, None)

################################################################## 
    # 在occ map中启发式找UAV和Camera Pair函数
##################################################################
def find_occluded_initial_pair_heuristic(
    occ: np.ndarray,
    meta: dict,
    follow_dist: float = 500,
    random_range: int = 3,
    vis_dir: Optional[str] = None,
    vis_prefix: Optional[str] = None,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    
    print("[DEBUG] 开始获取uav和human pair")
    
    # 获取occ的宽高
    H, W = occ.shape
    
    # 对occ进行连通图分解
    labels, num_labels, stats, centroids = find_connected_components(occ)
    
    # 从全部的障碍物中选择1个
    selected_object_idx = random.randint(1, num_labels - 1) # 需要0是free 空间
    # 获取连通图属性
    occ_x, occ_y, occ_w, occ_h, area = stats[selected_object_idx]
    # 获取occ的上右下左边
    occ_edge = [
        [occ_y, [occ_x,occ_x+occ_w-1]], # 上
         [[occ_y,occ_y+occ_h-1], occ_x+occ_w-1], # 右
        [occ_y+occ_h-1, [occ_x,occ_x+occ_w-1]], # 下
        [[occ_y,occ_y+occ_h-1], occ_x], # 左
    ]
    random_edge = [
        [-random_range,-1], # 上边，需要对x进行随机化
        [1, random_range], # 右边，需要对y进行随机化
        [1, random_range], # 下边，需要对x进行随机化
        [-random_range, -1], # 左边，需要对y进行随机化
        
    ]

    max_trials = 10
    for _ in range(max_trials):
        
        # 判定：如果选择了一条完全贴边的边，需要重新选择
        rechoose_human_edge_idx_flag = True
        while rechoose_human_edge_idx_flag:
            # 随机化连通图的上(idx=0)右(idx=3)下(idx=1)左(idx=2)边
            select_occ_edge_idx = random.randint(0, 3) # 上0下1左2右3
            select_occ_edge = occ_edge[select_occ_edge_idx] # 选择的边表示
            select_random_edge = random_edge[select_occ_edge_idx]
            
            for idx in range(len(select_occ_edge)):
                if not isinstance(select_occ_edge[idx],list):
                    if select_occ_edge[idx] == 0: # 选择到了单侧贴墙的edge
                        rechoose_human_edge_idx_flag = True
                    else:
                        rechoose_human_edge_idx_flag = False

        # 获取人的位置（加循环上限防卡死）
        human_pos = []
        regain_human_pos = True
        guard_human = 0
        # 判断条件：
        # 1) len(human_pos)==0: 还没有获取
        # 2) regain_human_pos：获取的点超出了范围
        # 3) labels[human_pos[0],human_pos[1]]!=0：获取的点不是free的
        while len(human_pos)==0 or regain_human_pos or labels[human_pos[0],human_pos[1]]!=0:
            for idx in range(len(select_occ_edge)):
                if isinstance(select_occ_edge[idx],list):
                    human_pos.append(random.randint(select_occ_edge[idx][0], select_occ_edge[idx][1]))
                else:
                    human_pos.append(select_occ_edge[idx]+random.randint(select_random_edge[0],select_random_edge[1]))
            if human_pos[0] >= H or human_pos[1] >= W:
                regain_human_pos = True
            else:
                regain_human_pos = False
            guard_human += 1
            if guard_human > 100:
                print("[WARN] human_pos sampling exceeded 100 iters, retry obstacle edge selection")
                break
        if guard_human > 100:
            continue
        
        human_pos_occ = copy.deepcopy(human_pos)
        human_pos[0], human_pos[1] = grid_to_world(int(human_pos[0]), int(human_pos[1]), meta)
        print("[Success] 完成human_pos的获取")

        # 判定：如果选择了一条完全贴边的边，需要重新选择
        rechoose_cam_edge_idx_flag = True
        while rechoose_cam_edge_idx_flag:
            # 随机选择相机是左转还是右转(例如：行人在上边界，相机可以在左边界，则右转，在右边界，则左转)
            cam_direction = random.randint(0, 1)
            # 获取相机所在边的编号
            if cam_direction == 0: # 上一条边
                cam_edge_idx = select_occ_edge_idx-1
                if cam_edge_idx < 0:
                    cam_edge_idx += 4
            elif cam_direction == 1: # 下一条边
                cam_edge_idx = (select_occ_edge_idx+1) % 4

            # 获取相机所在边的信息
            cam_edge = occ_edge[cam_edge_idx]
            cam_random_edge = random_edge[cam_edge_idx]
            
            for idx in range(len(cam_edge)):
                if not isinstance(cam_edge[idx],list):
                    if cam_edge[idx] == 0: # 选择到了单侧贴墙的edge
                        rechoose_cam_edge_idx_flag = True
                    else:
                        rechoose_cam_edge_idx_flag = False

        cam_pos = []
        regain_cam_pos = True
        guard_cam = 0
        # 判断条件：
        # 1) len(cam_pos)==0: 还没有获取
        # 2) regain_cam_pos：获取的点超出了范围
        # 3) labels[cam_pos[0],cam_pos[1]]!=0：获取的点不是free的
        while len(cam_pos)==0 or regain_cam_pos or labels[cam_pos[0],cam_pos[1]]!=0:
            for idx in range(len(cam_edge)):
                if isinstance(cam_edge[idx],list):
                    cam_pos.append(random.randint(cam_edge[idx][0], cam_edge[idx][1]))
                else:
                    cam_pos.append(cam_edge[idx]+random.randint(cam_random_edge[0],cam_random_edge[1]))
            if cam_pos[0] >= H or cam_pos[1] >= W:
                regain_cam_pos = True
            else:
                regain_cam_pos = False
            guard_cam += 1
            if guard_cam > 100:
                print("[WARN] cam_pos sampling exceeded 100 iters, retry obstacle edge selection")
                break
        if guard_cam > 100:
            continue

        cam_pos_occ = copy.deepcopy(cam_pos)
        cam_pos[0], cam_pos[1] = grid_to_world(int(cam_pos[0]), int(cam_pos[1]), meta)
        print("[Success] 完成cam_pos的获取")

        d = math.hypot(human_pos[0] - cam_pos[0], human_pos[1] - cam_pos[1])
        if d > float(follow_dist) * 2.0:
            print("[Fail] 距离过大，需要重新获取")
            continue

        if los_blocked_2d((cam_pos[0], cam_pos[1]), (human_pos[0], human_pos[1]), occ, meta):
            print("[Success] 获取成功！！")
            # 连接性检查：两点必须在同一自由连通域
            try:
                if not reachable_2d(occ, (human_pos_occ[0], human_pos_occ[1]), (cam_pos_occ[0], cam_pos_occ[1])):
                    print("[DEBUG] pair not reachable on occ, resample")
                    continue
            except Exception:
                pass
            cornor_points = [
                [occ_y, occ_x], # 左上
                [occ_y, occ_x+occ_w-1], # 右上
                [occ_y+occ_h-1, occ_x+occ_w-1], # 右下
                [occ_y+occ_h-1, occ_x], # 左下
            ]
            # Save visualization image without GUI (Agg backend). Allow custom dir/prefix.
            try:
                out_dir = vis_dir or os.getcwd()
                os.makedirs(out_dir, exist_ok=True)
                name = vis_prefix or f'occluded_pair_{int(time.time()*1000)}'
                save_path = os.path.join(out_dir, f'{name}.png')
                visualize_occluded_pair_with_obstacle(
                    occ, human_pos_occ, cam_pos_occ, selected_object_idx, cornor_points, save_path=save_path
                )
            except Exception:
                pass
            return (
                (float(human_pos[0]), float(human_pos[1])),
                (float(cam_pos[0]), float(cam_pos[1])),
                (float(human_pos_occ[0]), float(human_pos_occ[1])),
                (float(cam_pos_occ[0]), float(cam_pos_occ[1]))
            )
    return (None, None, None, None)

###### 1. 获取占用地图中的全部连通图 ######
def find_connected_components(occ_map):
    """
    使用OpenCV找到占用地图中所有连通的障碍物组件
    """
    # 确保输入是uint8类型
    occ_uint8 = (occ_map * 255).astype(np.uint8)
    
    # 使用cv2.connectedComponentsWithStats进行连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(occ_uint8, connectivity=8)
    
    # 参数1：labels，将每一个连通图都转化成了不同的标签，然后保存成array(0号元素是free的连通图)
    # 参数2：num_labels连通图的数量
    # 参数3：stats代表每一个连通图的属性[x,y,w,h,a]，分别表示左上角点的x,y，整个连通图的高h和宽w，以及面积a
    # 参数4：centroids代表每一个连通图的中点
    return labels, num_labels, stats, centroids

###### 2. 可视化目标和无人机初始化位置 ######
def visualize_positions_on_occ(occ, human_pos_occ, uav_pos_occ, save_path="positions_on_occ.png"):
    """
    在占用地图上可视化human_pos_occ和uav_pos_occ的位置
    """
    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 显示占用地图
    ax.imshow(occ, cmap='gray', origin='lower')
    
    # 标记human位置（红色圆圈）
    if human_pos_occ is not None:
        ax.plot(human_pos_occ[0], human_pos_occ[1], 'ro', markersize=10, label='Human')
        ax.annotate('Human', (human_pos_occ[0], human_pos_occ[1]), 
                   xytext=(5, 5), textcoords='offset points', 
                   color='red', fontsize=12, fontweight='bold')
    
    # 标记UAV位置（蓝色方块）
    if uav_pos_occ is not None:
        ax.plot(uav_pos_occ[0], uav_pos_occ[1], 'bs', markersize=10, label='UAV')
        ax.annotate('UAV', (uav_pos_occ[0], uav_pos_occ[1]), 
                   xytext=(5, 5), textcoords='offset points', 
                   color='blue', fontsize=12, fontweight='bold')
    
    # 设置标题和标签
    ax.set_title('Human and UAV Positions on Occupancy Map', fontsize=14)
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"位置可视化图像已保存到: {save_path}")

###### 2. 可视化目标和无人机初始化位置Plus版本 ######
def visualize_occluded_pair_with_obstacle(
    occ, human_pos_occ, cam_pos_occ, 
    selected_object_idx, corner_points, 
    save_path="occluded_pair_visualization.png"):
    """
    可视化遮挡对和选择的障碍物
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # 创建图像
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 显示占用地图
    ax.imshow(occ, cmap='gray', origin='lower')
    
    # 标记human位置（红色圆圈）
    if human_pos_occ is not None:
        ax.plot(human_pos_occ[1], human_pos_occ[0], 'ro', markersize=12, label='Human')
        ax.annotate(f'Human\n({human_pos_occ[0]}, {human_pos_occ[1]})', 
                   (human_pos_occ[1], human_pos_occ[0]), 
                   xytext=(10, 10), textcoords='offset points', 
                   color='red', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 标记UAV位置（蓝色方块）
    if cam_pos_occ is not None:
        ax.plot(cam_pos_occ[1], cam_pos_occ[0], 'bs', markersize=12, label='UAV')
        ax.annotate(f'UAV\n({cam_pos_occ[0]}, {cam_pos_occ[1]})', 
                   (cam_pos_occ[1], cam_pos_occ[0]), 
                   xytext=(10, -20), textcoords='offset points', 
                   color='blue', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 标记选择的障碍物边界框（绿色矩形）
    if corner_points:
        # 计算边界框
        min_row = min(point[0] for point in corner_points)
        max_row = max(point[0] for point in corner_points)
        min_col = min(point[1] for point in corner_points)
        max_col = max(point[1] for point in corner_points)
        
        # 绘制边界框
        rect = patches.Rectangle((min_col, min_row), max_col - min_col + 1, max_row - min_row + 1,
                               linewidth=3, edgecolor='green', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # 标注障碍物ID
        center_row = (min_row + max_row) / 2
        center_col = (min_col + max_col) / 2
        ax.annotate(f'Obstacle {selected_object_idx}', 
                   (center_col, center_row), 
                   xytext=(0, 0), textcoords='offset points', 
                   color='green', fontsize=12, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # 标记4个角点（绿色小圆点）
    for i, (row, col) in enumerate(corner_points):
        ax.plot(col, row, 'go', markersize=6)
        ax.annotate(f'C{i+1}\n({row}, {col})', 
                   (col, row), 
                   xytext=(5, 5), textcoords='offset points', 
                   color='green', fontsize=8, fontweight='bold')
    
    # 设置标题和标签
    ax.set_title(f'Occluded Pair Visualization - Obstacle {selected_object_idx}', fontsize=14)
    ax.set_xlabel('X (grid)')
    ax.set_ylabel('Y (grid)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"遮挡对可视化图像已保存到: {save_path}")

def get_projected_bbox(ucv: UnrealCv, target_name: str, cam_id: int = 0) -> Optional[List[int]]:
    """
    通过视图矩阵与投影矩阵将目标3D包围盒的8个角点投影至屏幕坐标，忽略遮挡，返回 [x,y,w,h]。
    若无法获取或没有点投影至前方视锥，则返回 None。
    """
    try:
        # 读取目标世界包围盒
        bounds = ucv.get_obj_bounds(target_name)
        minx, miny, minz, maxx, maxy, maxz = map(float, bounds)
        # 生成8个角点（齐次坐标）
        corners = [
            [minx, miny, minz, 1.0], [minx, miny, maxz, 1.0], [minx, maxy, minz, 1.0], [minx, maxy, maxz, 1.0],
            [maxx, miny, minz, 1.0], [maxx, miny, maxz, 1.0], [maxx, maxy, minz, 1.0], [maxx, maxy, maxz, 1.0],
        ]
        P_world = np.array(corners, dtype=np.float32)  # (8,4)
        # 读取矩阵
        resp_view = ucv.client.request(f'vget /camera/{cam_id}/view_matrix', -1)
        resp_proj = ucv.client.request(f'vget /camera/{cam_id}/proj_matrix', -1)
        def parse_mat(resp) -> np.ndarray:
            s = resp.decode('utf-8') if isinstance(resp, (bytes, bytearray)) else str(resp)
            vals = [float(x) for x in s.strip().split() if x.strip()]
            if len(vals) != 16:
                raise ValueError('bad matrix length')
            return np.array(vals, dtype=np.float32).reshape((4,4))
        V = parse_mat(resp_view)
        P = parse_mat(resp_proj)
        # 分辨率
        W, H = None, None
        if hasattr(ucv, 'resolution') and isinstance(ucv.resolution, (list, tuple)) and len(ucv.resolution) >= 2:
            W, H = int(ucv.resolution[0]), int(ucv.resolution[1])
        if W is None or H is None or W <= 0 or H <= 0:
            try:
                img0 = ucv.read_image(cam_id, 'lit', 'fast')
                if img0 is not None and hasattr(img0, 'shape') and len(img0.shape) >= 2:
                    H, W = int(img0.shape[0]), int(img0.shape[1])
            except Exception:
                pass
        if W is None or H is None or W <= 0 or H <= 0:
            W, H = 320, 240
        # 两种约定都尝试：
        def project_with_matrix(M: np.ndarray) -> Optional[List[int]]:
            clip = (M @ P_world.T)  # (4,8)
            w = clip[3, :]
            # 过滤 w 近零，避免除零
            mask = np.abs(w) > 1e-6
            if not np.any(mask):
                return None
            ndc = clip[:, mask] / np.abs(w[mask])
            x_ndc = ndc[0, :]
            y_ndc = ndc[1, :]
            # 允许轻微越界，保留 [-1.2, 1.2]
            in_frustum = (x_ndc >= -1.2) & (x_ndc <= 1.2) & (y_ndc >= -1.2) & (y_ndc <= 1.2)
            if not np.any(in_frustum):
                return None
            x_ndc = x_ndc[in_frustum]
            y_ndc = y_ndc[in_frustum]
            x_scr = (x_ndc + 1.0) * 0.5 * float(W)
            y_scr = (1.0 - y_ndc) * 0.5 * float(H)
            xmin = max(0, int(np.floor(np.min(x_scr))))
            xmax = min(W - 1, int(np.ceil(np.max(x_scr))))
            ymin = max(0, int(np.floor(np.min(y_scr))))
            ymax = min(H - 1, int(np.ceil(np.max(y_scr))))
            if xmax <= xmin or ymax <= ymin:
                return None
            return [xmin, ymin, max(1, xmax - xmin), max(1, ymax - ymin)]
        # 尝试列向量约定
        bbox = project_with_matrix(P @ V)
        if bbox is None:
            # 尝试行向量约定（等价于对列约定转置）
            bbox = project_with_matrix((P.T @ V.T))
        return bbox
    except Exception:
        return None

def compute_vis_ratio_by_mask(ucv: UnrealCv, target_name: str, obstacles: Optional[List[str]] = None, cam_id: int = 0) -> float:
    """
    以 mask 当前可见像素 / 在临时隐藏障碍时的满可见像素 作为可见度比例，返回 [0,1]。
    失败时返回 1.0（偏保守）。
    """
    try:
        mask_now = ucv.read_image(cam_id, 'object_mask', 'fast')
        tgt_mask = ucv.get_mask(mask_now, target_name)
        cur = int(cv2.countNonZero(tgt_mask)) if cv2 is not None else int((tgt_mask > 0).sum())
    except Exception:
        cur = 0

    full = 0
    # 过滤掉目标自身，避免误隐藏
    to_hide = [o for o in (obstacles or []) if o != target_name]
    hidden_ok = []
    for obj in to_hide:
        try:
            ucv.hide_obj(obj)
            hidden_ok.append(obj)
        except Exception:
            pass

    # 强制渲染刷新 + 适当等待，避免读到同一帧
    try:
        _ = ucv.read_image(cam_id, 'lit', 'fast')
    except Exception:
        pass
    time.sleep(0.05)
    try:
        # 连续读取两次，确保拿到最新 mask
        mask_full = ucv.read_image(cam_id, 'object_mask', 'fast')
        time.sleep(0.01)
        mask_full = ucv.read_image(cam_id, 'object_mask', 'fast')
        tgt_mask2 = ucv.get_mask(mask_full, target_name)
        full = int(cv2.countNonZero(tgt_mask2)) if cv2 is not None else int((tgt_mask2 > 0).sum())
    except Exception:
        full = 0

    # 恢复显示
    for obj in hidden_ok:
        try:
            ucv.show_obj(obj)
        except Exception:
            pass
    if hidden_ok:
        time.sleep(0.02)

    if full <= 0:
        try:
            print(f"[DEBUG] vis: full<=0 for target={target_name} cam={cam_id}, fail-open as 0.0")
        except Exception:
            pass
        return 0.0
    return float(cur) / float(full)

# 随机化场景中的物体
def randomize_layout_simple(ucv: UnrealCv, setting: dict, tall_obstacles: bool = False,
                            xy_scale_range: Tuple[float, float] = (0.6, 2.0),
                            z_scale_range: Tuple[float, float] = (0.6, 2.0),
                            verbose: bool = False,
                            uav_xy_hint: Optional[Tuple[float,float]] = None,
                            target_xy_hint: Optional[Tuple[float,float]] = None,
                            min_gap: float = 150.0) -> List[str]: # Changed return type
    env_cfg = setting.get('env') or {}
    spawn_class_overrides = setting.get('spawn_class_overrides') or []
    reset_area = setting.get('reset_area')
    objects_list = list(env_cfg.get('objects') or [])
    if len(objects_list) == 0 or not (isinstance(reset_area, list) and len(reset_area) >= 6):
        return [] # Return empty list
    xmin, xmax, ymin, ymax, zmin, zmax = reset_area
    safe_pts = setting.get('safe_start', []) or []
    if isinstance(safe_pts, list) and len(safe_pts) > 0 and isinstance(safe_pts[0], list):
        xs = [p[0] for p in safe_pts]; ys = [p[1] for p in safe_pts]
        avoid_rect = [min(xs), max(xs), min(ys), max(ys)]
    else:
        avoid_rect = None

    placed_names: List[str] = [] # Changed from placed_ok to a list of names
    placed_rects: List[Tuple[float,float,float,float]] = []  # minx,miny,maxx,maxy
    # 相当于将object_list中的全部物体都在场景内随机初始化1次
    for obj in objects_list:
        # 首先随机化物体的大小
        sx = random.uniform(xy_scale_range[0], xy_scale_range[1]) # x方向的缩放
        sy = random.uniform(xy_scale_range[0], xy_scale_range[1]) # y方向的缩放
        sz = random.uniform(z_scale_range[0], z_scale_range[1]) # z方向的缩放
        if tall_obstacles: # 如果是高障碍，则取最大的z方向缩放
            sz = max(sz, z_scale_range[0])
        try:
            ucv.client.request(f'vset /object/{obj}/scale {sx} {sy} {sz}', -1) # 设置物体的大小
        except Exception:
            pass
        tries = 0
        while True:
            tries += 1
            # 带LoS偏好采样：沿 UAV->Target 线段附近摆放（若提供了hint则强制使用）
            if uav_xy_hint is not None and target_xy_hint is not None:
                t = random.uniform(0.2, 0.8)
                x = uav_xy_hint[0] * (1-t) + target_xy_hint[0] * t + random.uniform(-120, 120)
                y = uav_xy_hint[1] * (1-t) + target_xy_hint[1] * t + random.uniform(-120, 120)
            else:
                x = random.uniform(xmin + 100, xmax - 100) # 随机化障碍物的x坐标
                y = random.uniform(ymin + 100, ymax - 100) # 随机化障碍物的y坐标
            z = random.uniform(zmin, zmax) # 随机化障碍物的z坐标
            if avoid_rect is None:
                break
            if not (avoid_rect[0] <= x <= avoid_rect[1] and avoid_rect[2] <= y <= avoid_rect[3]):
                break
            if tries > 50:
                break
        
        # 尝试避免与既有障碍重叠：若重叠则重试（无碰撞放置物体）
        placed = False
        for _ in range(40):
            try:
                ucv.set_obj_location(obj, [x, y, z]) # 放置物体
                bounds = ucv.get_obj_bounds(obj) # 获取物体的边界
                minx, miny, _, maxx, maxy, _ = bounds
                # 扩展矩形以形成间隙
                minx -= min_gap; maxx += min_gap; miny -= min_gap; maxy += min_gap
                overlap = False
                for (ax0, ay0, ax1, ay1) in placed_rects:
                    if not (maxx < ax0 or ax1 < minx or maxy < ay0 or ay1 < miny):
                        overlap = True
                        break
                if overlap:
                    # 重采样位置（若有hint则继续沿LoS采样）
                    if uav_xy_hint is not None and target_xy_hint is not None:
                        t = random.uniform(0.2, 0.8)
                        x = uav_xy_hint[0] * (1-t) + target_xy_hint[0] * t + random.uniform(-120, 120)
                        y = uav_xy_hint[1] * (1-t) + target_xy_hint[1] * t + random.uniform(-120, 120)
                    else:
                        x = random.uniform(xmin + 100, xmax - 100)
                        y = random.uniform(ymin + 100, ymax - 100)
                    continue
                placed_rects.append((minx, miny, maxx, maxy))
                placed_names.append(obj) # Add the object name to the list
                placed = True
                break
            except Exception:
                break
        if not placed:
            continue
    if verbose:
        try:
            print(f"[DEBUG] layout(all) objects={len(objects_list)}, placed_ok={len(placed_names)}")
        except Exception:
            pass
    return placed_names # Return the list of placed object names

def build_occupancy2d(ucv: UnrealCv, setting: dict, cell: int = 50, inflate: int = 80, use_whitelist: bool = True) -> Tuple[np.ndarray, dict]:
    reset_area = setting['reset_area']
    xmin, xmax, ymin, ymax = reset_area[:4]
    width = max(1, int((xmax - xmin) // cell) + 1)
    height = max(1, int((ymax - ymin) // cell) + 1)
    occ = np.zeros((height, width), dtype=np.uint8) # 初始化occ_map，如果像素被占用，则为1，如果未被占用，则为0
    env_cfg = setting.get('env') or {}
    # 如果使用use_whitelist，则直接获取使用的资产(应该得要打开)
    objects = env_cfg.get('objects') if use_whitelist else None
    # 直接调用unrealcv获取场景内所有的物体
    objs = list(objects) if objects else ucv.get_objects()
    ignore = ('floor','FLOOR','ground','Ground','Landscape','LandscapeProxy','wall','Wall','WALL','Room','room','Sky','sky','SkySphere','Ceiling','ceiling')
    for obj in objs:
        name = str(obj)
        # 即使这里过滤掉了，依然会存在很多其他没用的资产
        if any(k in name for k in ignore):
            continue
        try:
            minx, miny, minz, maxx, maxy, maxz = ucv.get_obj_bounds(obj)
        except Exception:
            continue
        minx -= inflate; maxx += inflate; miny -= inflate; maxy += inflate # 给occ增加一定的安全距离
        ix0 = max(0, int((minx - xmin) // cell)); ix1 = min(width - 1, int((maxx - xmin) // cell))
        iy0 = max(0, int((miny - ymin) // cell)); iy1 = min(height - 1, int((maxy - ymin) // cell))
        if ix0 <= ix1 and iy0 <= iy1:
            occ[iy0:iy1+1, ix0:ix1+1] = 1
    meta = dict(origin=[float(xmin), float(ymin)], cell_size=int(cell), shape=tuple(occ.shape), reset_area=list(reset_area))
    return occ, meta


    

def world_to_grid_3d(x: float, y: float, z: float, meta3d: dict) -> Tuple[int,int,int]:
    cell_xy, cell_z = float(meta3d['cell_size'][0]), float(meta3d['cell_size'][2])
    j = int((x - meta3d['origin'][0]) // cell_xy)
    i = int((y - meta3d['origin'][1]) // cell_xy)
    k = int((z - meta3d['origin'][2]) // cell_z)
    return k, i, j  # (k,z), (i,y), (j,x)

def grid_to_world_3d(k: int, i: int, j: int, meta3d: dict) -> Tuple[float,float,float]:
    cell_xy, cell_z = float(meta3d['cell_size'][0]), float(meta3d['cell_size'][2])
    x = meta3d['origin'][0] + (j + 0.5) * cell_xy
    y = meta3d['origin'][1] + (i + 0.5) * cell_xy
    z = meta3d['origin'][2] + (k + 0.5) * cell_z
    return x, y, z

    

def astar3d(occ3d: np.ndarray, start_xyz: Tuple[float,float,float], goal_xyz: Tuple[float,float,float], meta3d: dict) -> List[Tuple[float,float,float]]:
    from heapq import heappush, heappop
    nz, ny, nx = occ3d.shape
    ks, is_, js = world_to_grid_3d(*start_xyz, meta3d)
    kg, ig, jg = world_to_grid_3d(*goal_xyz, meta3d)
    if not (0<=ks<nz and 0<=is_<ny and 0<=js<nx and 0<=kg<nz and 0<=ig<ny and 0<=jg<nx):
        return []
    if occ3d[kg, ig, jg] == 1:
        return []
    # 26邻域
    neighbors = [(dz,dy,dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)]
    def h(k,i,j):
        x,y,z = grid_to_world_3d(k,i,j,meta3d)
        return math.sqrt((x-goal_xyz[0])**2 + (y-goal_xyz[1])**2 + (z-goal_xyz[2])**2)
    g = {(ks,is_,js): 0.0}
    parent = {(ks,is_,js): None}
    pq = [(h(ks,is_,js), (ks,is_,js))]
    visited = set()
    while pq:
        _, (k,i,j) = heappop(pq)
        if (k,i,j) in visited:
            continue
        visited.add((k,i,j))
        if (k,i,j) == (kg,ig,jg):
            path = []
            cur = (k,i,j)
            while cur is not None:
                path.append(grid_to_world_3d(*cur, meta3d))
                cur = parent[cur]
            return path[::-1]
        for dz,dy,dx in neighbors:
            nk, ni, nj = k+dz, i+dy, j+dx
            if not (0<=nk<nz and 0<=ni<ny and 0<=nj<nx):
                continue
            if occ3d[nk,ni,nj] == 1:
                continue
            # 代价：各向异性步长
            step_xy = math.hypot(dx, dy) * float(meta3d['cell_size'][0])
            step_z  = abs(dz) * float(meta3d['cell_size'][2])
            w = math.hypot(step_xy, step_z)
            cand = g[(k,i,j)] + w
            if (nk,ni,nj) not in g or cand < g[(nk,ni,nj)]:
                g[(nk,ni,nj)] = cand
                parent[(nk,ni,nj)] = (k,i,j)
                heappush(pq, (cand + h(nk,ni,nj), (nk,ni,nj)))
    return []

def segment_free_3d(p0: Tuple[float,float,float], p1: Tuple[float,float,float], occ3d: np.ndarray, meta3d: dict) -> bool:
    """
    采样检查3D线段p0->p1是否无碰撞（在安全占用/占用图中为0）。
    采样步数与路径长度和体素尺寸相关。
    """
    cell_xy = float(meta3d['cell_size'][0])
    cell_z  = float(meta3d['cell_size'][2])
    step_unit = min(cell_xy, cell_z)
    dist = math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2 + (p1[2]-p0[2])**2)
    steps = max(2, int(math.ceil(dist / max(1e-6, step_unit))))
    for k in range(steps+1):
        t = k / float(steps)
        x = p0[0] + (p1[0]-p0[0]) * t
        y = p0[1] + (p1[1]-p0[1]) * t
        z = p0[2] + (p1[2]-p0[2]) * t
        kz, iy, jx = world_to_grid_3d(x, y, z, meta3d)
        if not (0 <= kz < occ3d.shape[0] and 0 <= iy < occ3d.shape[1] and 0 <= jx < occ3d.shape[2]):
            return False
        if occ3d[kz, iy, jx] == 1:
            return False
    return True

def shortcut_path3d(path: List[Tuple[float,float,float]], occ3d: np.ndarray, meta3d: dict, max_iter: int = 200) -> List[Tuple[float,float,float]]:
    """
    随机-贪心的3D路径平滑：随机挑选(i,j)尝试直连，无碰撞则删去中间点。
    """
    if len(path) < 3:
        return path
    pts = path[:]
    it = 0
    while it < max_iter and len(pts) > 2:
        i = random.randint(0, max(0, len(pts)-3))
        j = random.randint(i+2, len(pts)-1)
        p0, p1 = pts[i], pts[j]
        if segment_free_3d(p0, p1, occ3d, meta3d):
            pts = pts[:i+1] + pts[j:]
        it += 1
    return pts

def get_cam_pose_safe(ucv: UnrealCv, cam_id: int) -> List[float]:
    """优先硬读取，失败则软读取，避免 UnrealCV 返回 'error' 导致转换异常。"""
    try:
        loc = ucv.get_location(cam_id, 'hard')
        rot = ucv.get_rotation(cam_id, 'hard')
        return loc + rot
    except Exception:
        try:
            loc = ucv.get_location(cam_id, 'soft')
            rot = ucv.get_rotation(cam_id, 'soft')
            return loc + rot
        except Exception:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# 可视化astar规划出来的轨迹
def visualize_occ_with_trajectory(occ: np.ndarray, meta: dict, 
                                 uav_xy: Tuple[float, float], 
                                 target_xy: Tuple[float, float],
                                 traj_pts_xyz: List[Tuple[float, float, float]],
                                 output_path: str = "occ_trajectory_vis.png"):
    """
    在占用网格上可视化无人机位置、目标位置和规划轨迹
    
    Args:
        occ: 占用网格 (0=自由空间, 1=障碍物)
        meta: 网格元数据
        uav_xy: 无人机当前位置 (x, y)
        target_xy: 目标位置 (hx, hy)
        traj_pts_xyz: 规划轨迹点列表 [(x, y, z), ...]
        output_path: 输出图像路径
    """
    import cv2
    import numpy as np
    
    H, W = occ.shape
    scale = 6  # 放大倍数，便于查看
    
    # 创建可视化图像
    # 将占用网格转换为图像 (0=黑色障碍物, 255=白色自由空间)
    img_occ = (255 * (1 - occ.astype(np.uint8))).astype(np.uint8)
    
    # 放大图像并转换为彩色
    vis = cv2.cvtColor(cv2.resize(img_occ, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    
    # 定义颜色
    colors = {
        'target': (255, 0, 255),      # 洋红色 - 目标
        'uav': (0, 0, 255),           # 红色 - 无人机
        'trajectory': (0, 255, 0),    # 绿色 - 轨迹
        'start': (255, 0, 0),         # 蓝色 - 起点
        'end': (0, 255, 255),         # 黄色 - 终点
    }
    
    # 辅助函数：世界坐标转图像坐标
    def world_to_image_coords(wx: float, wy: float) -> Tuple[int, int]:
        gi, gj = world_to_grid(wx, wy, meta)
        if 0 <= gi < H and 0 <= gj < W:
            ix = int((gj + 0.5) * scale)
            iy = int((gi + 0.5) * scale)
            return ix, iy
        return None, None
    
    # 1. 标记目标位置 (hx, hy)
    hx, hy = target_xy
    tx, ty = world_to_image_coords(hx, hy)
    if tx is not None and ty is not None:
        # 画十字标记
        cv2.line(vis, (tx-10, ty), (tx+10, ty), colors['target'], 3)
        cv2.line(vis, (tx, ty-10), (tx, ty+10), colors['target'], 3)
        cv2.circle(vis, (tx, ty), 8, colors['target'], 2)
        cv2.putText(vis, 'TARGET', (tx+12, ty-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['target'], 2)
    
    # 2. 标记无人机位置 (uav_xy)
    ux, uy = world_to_image_coords(uav_xy[0], uav_xy[1])
    if ux is not None and uy is not None:
        cv2.circle(vis, (ux, uy), 10, colors['uav'], -1)
        cv2.putText(vis, 'UAV', (ux+12, uy+12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors['uav'], 2)
    
    # 3. 绘制规划轨迹
    if traj_pts_xyz and len(traj_pts_xyz) > 0:
        # 转换轨迹点到图像坐标
        traj_points = []
        for i, (px, py, pz) in enumerate(traj_pts_xyz):
            ix, iy = world_to_image_coords(px, py)
            if ix is not None and iy is not None:
                traj_points.append((ix, iy, i))
        
        # 绘制轨迹连线
        for i in range(len(traj_points) - 1):
            pt1 = (traj_points[i][0], traj_points[i][1])
            pt2 = (traj_points[i+1][0], traj_points[i+1][1])
            cv2.line(vis, pt1, pt2, colors['trajectory'], 3)
        
        # 绘制轨迹点
        for i, (px, py, idx) in enumerate(traj_points):
            if i == 0:
                # 起点
                cv2.circle(vis, (px, py), 6, colors['start'], -1)
                cv2.putText(vis, f'S{idx}', (px+8, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['start'], 1)
            elif i == len(traj_points) - 1:
                # 终点
                cv2.circle(vis, (px, py), 6, colors['end'], -1)
                cv2.putText(vis, f'E{idx}', (px+8, py-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['end'], 1)
            else:
                # 中间点
                cv2.circle(vis, (px, py), 4, colors['trajectory'], -1)
                cv2.putText(vis, str(idx), (px+6, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # 4. 添加图例
    legend_y = 30
    cv2.putText(vis, 'Legend:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    legend_y += 25
    
    # 目标图例
    cv2.circle(vis, (20, legend_y), 6, colors['target'], -1)
    cv2.putText(vis, 'Target', (35, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 20
    
    # 无人机图例
    cv2.circle(vis, (20, legend_y), 6, colors['uav'], -1)
    cv2.putText(vis, 'UAV', (35, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 20
    
    # 轨迹图例
    cv2.circle(vis, (20, legend_y), 6, colors['trajectory'], -1)
    cv2.putText(vis, 'Trajectory', (35, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 20
    
    # 起点图例
    cv2.circle(vis, (20, legend_y), 6, colors['start'], -1)
    cv2.putText(vis, 'Start', (35, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    legend_y += 20
    
    # 终点图例
    cv2.circle(vis, (20, legend_y), 6, colors['end'], -1)
    cv2.putText(vis, 'End', (35, legend_y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 5. 添加信息文本
    info_text = f"Grid: {W}x{H}, Scale: {scale}x"
    cv2.putText(vis, info_text, (10, vis.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if traj_pts_xyz:
        traj_info = f"Trajectory: {len(traj_pts_xyz)} points"
        cv2.putText(vis, traj_info, (10, vis.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存图像
    cv2.imwrite(output_path, vis)
    print(f"已保存轨迹可视化图像: {output_path}")
    
    return vis

# 可视化远近点函数(**远近点目前已经被废除了**)
def visualize_occ_with_astar(occ: np.ndarray, meta: dict, gx_far: float, gy_far: float, 
                         gx: float, gy: float, hx: float, hy: float, 
                         uav_xy: Tuple[float, float], output_path: str = "occ_with_targets.png"):
    """
    在占用网格上标记各种目标点并保存为图像
    
    Args:
        occ: 占用网格 (0=自由空间, 1=障碍物)
        meta: 网格元数据
        gx_far, gy_far: 远距离跟随点
        gx, gy: 近距离跟随点  
        hx, hy: 目标行人位置
        uav_xy: 无人机当前位置
        output_path: 输出图像路径
    """
    import cv2
    import matplotlib.pyplot as plt
    
    H, W = occ.shape
    scale = 4  # 放大倍数，便于查看
    
    # 创建可视化图像
    # 将占用网格转换为图像 (0=黑色障碍物, 255=白色自由空间)
    img_occ = (255 * (1 - occ.astype(np.uint8))).astype(np.uint8)
    
    # 放大图像并转换为彩色
    vis = cv2.cvtColor(cv2.resize(img_occ, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
    
    # 定义颜色
    colors = {
        'target': (255, 0, 255),      # 洋红色 - 目标行人
        'uav': (255, 0, 0),           # 红色 - 无人机
        'near_goal': (0, 255, 0),     # 绿色 - 近距离跟随点
        'far_goal': (0, 0, 255),      # 蓝色 - 远距离跟随点
    }
    
    # 标记目标行人位置 (hx, hy)
    ti, tj = world_to_grid(hx, hy, meta)
    if 0 <= ti < H and 0 <= tj < W:
        cx = int((tj + 0.5) * scale)
        cy = int((ti + 0.5) * scale)
        # 画十字标记
        cv2.line(vis, (cx-8, cy), (cx+8, cy), colors['target'], 3)
        cv2.line(vis, (cx, cy-8), (cx, cy+8), colors['target'], 3)
        cv2.circle(vis, (cx, cy), 6, colors['target'], 2)
        cv2.putText(vis, 'TARGET', (cx+10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['target'], 1)
    
    # 标记无人机位置 (uav_xy)
    ui, uj = world_to_grid(uav_xy[0], uav_xy[1], meta)
    if 0 <= ui < H and 0 <= uj < W:
        ux = int((uj + 0.5) * scale)
        uy = int((ui + 0.5) * scale)
        cv2.circle(vis, (ux, uy), 8, colors['uav'], -1)
        cv2.putText(vis, 'UAV', (ux+10, uy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors['uav'], 1)
    
    # 标记近距离跟随点 (gx, gy)
    ni, nj = world_to_grid(gx, gy, meta)
    if 0 <= ni < H and 0 <= nj < W:
        nx = int((nj + 0.5) * scale)
        ny = int((ni + 0.5) * scale)
        cv2.circle(vis, (nx, ny), 6, colors['near_goal'], 2)
        cv2.putText(vis, 'NEAR', (nx+10, ny+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['near_goal'], 1)
    
    # 标记远距离跟随点 (gx_far, gy_far)
    fi, fj = world_to_grid(gx_far, gy_far, meta)
    if 0 <= fi < H and 0 <= fj < W:
        fx = int((fj + 0.5) * scale)
        fy = int((fi + 0.5) * scale)
        cv2.circle(vis, (fx, fy), 6, colors['far_goal'], 2)
        cv2.putText(vis, 'FAR', (fx+10, fy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['far_goal'], 1)
    
    # 画连接线
    # 目标到近距离跟随点
    if 0 <= ti < H and 0 <= tj < W and 0 <= ni < H and 0 <= nj < W:
        cv2.line(vis, (cx, cy), (nx, ny), colors['near_goal'], 2)
    
    # 目标到远距离跟随点  
    if 0 <= ti < H and 0 <= tj < W and 0 <= fi < H and 0 <= fj < W:
        cv2.line(vis, (cx, cy), (fx, fy), colors['far_goal'], 2)
    
    # 无人机到近距离跟随点
    if 0 <= ui < H and 0 <= uj < W and 0 <= ni < H and 0 <= nj < W:
        cv2.line(vis, (ux, uy), (nx, ny), (0, 255, 255), 1)  # 黄色虚线
    
    # 无人机到远距离跟随点
    if 0 <= ui < H and 0 <= uj < W and 0 <= fi < H and 0 <= fj < W:
        cv2.line(vis, (ux, uy), (fx, fy), (255, 255, 0), 1)  # 青色虚线
    
    # 保存图像
    cv2.imwrite(output_path, vis)
    print(f"已保存占用网格可视化图像: {output_path}")
    
    return vis

def collect_single_sample(
    ucv: UnrealCv,
    occ: np.ndarray,
    meta: dict,
    occ3d: Optional[np.ndarray],
    meta3d: Optional[dict],
    target_name: str,
    args,
    human_xy: Tuple[float,float],
    sub_ep_idx: int,
    cam_id: int,
    vis: bool = False
) -> Optional[Tuple[dict, np.ndarray, Optional[np.ndarray]]]:
    """
    执行一次静态的 观测-规划-记录：
      - 观测：对准 human_xy 拍摄一帧，并计算 bbox（针对指定相机 cam_id）
      - 规划：从当前 UAV 位姿到“近/远锚点”进行三维优先的路径规划（或2D回退）
      - 输出：验证成功后，返回 (rec, img_pre, vis_data)
    """
    hx, hy = float(human_xy[0]), float(human_xy[1]) # 获取行人当前在XoY平面的位置
    # 获取行人当前的高度
    base_h = float(args.uav_height) if args.uav_height is not None else float((meta.get('reset_area',[0,0,0,0,0,0])[4] + meta.get('reset_area',[0,0,0,0,0,0])[5]) * 0.5)
    # 获取 UAV 位置（按相机）
    try:
        uav_loc = ucv.get_location(cam_id)
    except Exception:
        uav_loc = get_cam_pose_safe(ucv, cam_id)[:3]
    # 获取当前相机(无人机)的位置
    uav_xy = (float(uav_loc[0]), float(uav_loc[1]))
    uav_z  = float(uav_loc[2]) if len(uav_loc) >= 3 else float(base_h)

    # 进入函数的关键信息
    if getattr(args, 'debug', False):
        try:
            print(f"[DEBUG] enter collect: vis_thr={getattr(args,'vis_threshold',None)} "
                  f"yaw_jitter_deg={getattr(args,'yaw_jitter_deg',None)} "
                  f"hx={hx:.1f},hy={hy:.1f}, cam=({uav_xy[0]:.1f},{uav_xy[1]:.1f},{uav_z:.1f})")
        except Exception:
            pass

    # 朝向并拍摄：在“正对目标”的基础上，依据前向扇形扫描远离障碍小幅偏转
    yaw_face_base = math.degrees(math.atan2(hy - uav_xy[1], hx - uav_xy[0]))
    horiz = max(1e-3, math.hypot(hx - uav_xy[0], hy - uav_xy[1]))
    pitch_face = -math.degrees(math.atan2((uav_z - base_h), horiz))
    try:
        # 仅暴露一个参数：最大偏转角
        jitter = float(getattr(args, 'yaw_jitter_deg', 8.0))
        yaw_face = yaw_face_base
        sign = _yaw_away_sign_by_fan(occ, meta, uav_xy, yaw_face_base)
        if jitter > 0.0 and sign != 0:
            yaw_face = (yaw_face_base + sign * random.uniform(0.5 * jitter, jitter) + 360.0) % 360.0
        ucv.set_rotation(cam_id, [pitch_face, yaw_face, 0])
        time.sleep(0.02)
    except Exception:
        pass

    img_pre: Optional[np.ndarray] = None
    try:
        img_pre = ucv.read_image(cam_id, 'lit', 'fast') # 获取相机拍摄的图像
    except Exception:
        img_pre = None
    if img_pre is None:
        # 回正视角重试一次
        try:
            if getattr(args, 'debug', False):
                try:
                    print('[DEBUG] drop: img_pre is None after yaw; retry center')
                except Exception:
                    pass
            ucv.set_rotation(cam_id, [pitch_face, yaw_face_base, 0])
            time.sleep(0.02)
            img_pre = ucv.read_image(cam_id, 'lit', 'fast')
        except Exception:
            img_pre = None
    if img_pre is None:
        if getattr(args, 'debug', False):
            try:
                print('[DEBUG] drop: img_pre still None after center')
            except Exception:
                pass
        return None

    # 可见度门控（仅针对本对 target）
    try:
        obstacles_list = getattr(args, 'obstacles_whitelist', []) or [] # 获取当前全部的障碍物
        # 核心函数：计算可见度比率
        vis_ratio_old = compute_vis_ratio_by_mask(ucv, target_name, obstacles_list, cam_id=cam_id)
        if getattr(args, 'debug', False):
            try:
                print(f"[DEBUG] gate: vis_ratio={vis_ratio_old:.3f} thr={float(getattr(args,'vis_threshold',0.35))}")
            except Exception:
                pass
        if vis_ratio_old > float(getattr(args, 'vis_threshold', 0.35)):
            if getattr(args, 'debug', False):
                try:
                    print(f"[DEBUG] drop: vis_ratio_old too high {vis_ratio_old:.2f} target={target_name} cam={cam_id}")
                except Exception:
                    pass
            return None
    except Exception:
        pass

    # 计算 bbox（优先投影，失败回退无遮挡mask）
    bbox_py = get_projected_bbox(ucv, target_name, cam_id=cam_id)
    if bbox_py is not None and getattr(args, 'debug', False):
        try:
            print(f"[DEBUG] proj bbox ok: {bbox_py}")
        except Exception:
            pass
    if bbox_py is None and getattr(args, 'debug', False):
        try:
            print('[DEBUG] proj bbox failed at current yaw; retry center yaw')
        except Exception:
            pass
    if bbox_py is None:
        # 回正视角再试一次投影 bbox
        try:
            ucv.set_rotation(cam_id, [pitch_face, yaw_face_base, 0])
            time.sleep(0.02)
            bbox_py = get_projected_bbox(ucv, target_name, cam_id=cam_id)
        except Exception:
            pass
    if bbox_py is not None and getattr(args, 'debug', False):
        try:
            print(f"[DEBUG] proj bbox ok after center: {bbox_py}")
        except Exception:
            pass
    if bbox_py is None and getattr(args, 'debug', False):
        try:
            print('[DEBUG] proj bbox failed after center yaw; try mask fallback')
        except Exception:
            pass
    if bbox_py is None:
        obstacles_list = getattr(args, 'obstacles_whitelist', []) or []
        hidden_ok = []
        try:
            for obj in obstacles_list:
                try:
                    ucv.hide_obj(obj)
                    hidden_ok.append(obj)
                except Exception:
                    pass
            time.sleep(0.02)
            mask_full = ucv.read_image(cam_id, 'object_mask', 'fast')
            if mask_full is None and getattr(args, 'debug', False):
                try:
                    print('[DEBUG] mask fallback read None')
                except Exception:
                    pass
            if mask_full is not None:
                _, bbox_gt = ucv.get_bbox(mask_full, target_name, normalize=False)
                bbox_py = [int(bbox_gt[0]), int(bbox_gt[1]), int(bbox_gt[2]), int(bbox_gt[3])]
                if getattr(args, 'debug', False):
                    try:
                        print(f"[DEBUG] mask fallback bbox ok: {bbox_py}")
                    except Exception:
                        pass
        except Exception:
            bbox_py = None
        finally:
            for obj in hidden_ok:
                try:
                    ucv.show_obj(obj)
                except Exception:
                    pass
    if bbox_py is None or bbox_py[2] <= 1 or bbox_py[3] <= 1:
        if getattr(args, 'debug', False):
            try:
                if bbox_py is None:
                    print('[DEBUG] mask fallback failed (bbox=None)')
                else:
                    print(f"[DEBUG] mask fallback failed (bbox={bbox_py})")
            except Exception:
                pass
        return None

    # 规划：近/远锚点（使用正对 yaw 更稳妥）
    yaw_dir = yaw_face_base
    gx = hx - args.follow_dist * math.cos(math.radians(yaw_dir)) # args.follow_dist默认是250
    gy = hy - args.follow_dist * math.sin(math.radians(yaw_dir))
    # args.standoff_mult默认是1.3
    gx_far = hx - (args.follow_dist * args.standoff_mult) * math.cos(math.radians(yaw_dir))
    gy_far = hy - (args.follow_dist * args.standoff_mult) * math.sin(math.radians(yaw_dir))

    traj_pts_xyz: List[Tuple[float,float,float]] = []
    # 使用occ 3d+ A star 3D进行规划，默认不使用
    if occ3d is not None and meta3d is not None:
        desired = args.follow_dist * args.standoff_mult
        p_far = astar3d(occ3d, (uav_xy[0], uav_xy[1], uav_z), (gx_far, gy_far, uav_z), meta3d)
        p_near = astar3d(occ3d, (uav_xy[0], uav_xy[1], uav_z), (gx, gy, uav_z), meta3d)
        cands = []
        for pth in (p_far or [], p_near or []):
            if pth and len(pth) >= 2:
                pth2 = shortcut_path3d(pth, occ3d, meta3d, max_iter=200)
                nx, ny, nz = pth2[min(1, len(pth2)-1)]
                err = abs(math.hypot(hx - nx, hy - ny) - desired)
                cands.append((err, len(pth2), pth2))
        if not cands:
            p_ret = astar3d(occ3d, (uav_xy[0], uav_xy[1], uav_z), (hx, hy, uav_z), meta3d)
            if p_ret and len(p_ret) >= 2:
                traj_pts_xyz = [(float(x), float(y), float(z)) for (x,y,z) in shortcut_path3d(p_ret, occ3d, meta3d, max_iter=200)]
                if getattr(args, 'debug', False):
                    try:
                        print(f"[DEBUG] astar3d traj_len={len(traj_pts_xyz)}")
                    except Exception:
                        pass
        else:
            cands.sort(key=lambda t: (t[0], t[1]))
            traj_pts_xyz = [(float(x), float(y), float(z)) for (x,y,z) in cands[0][2]]
            if getattr(args, 'debug', False):
                try:
                    print(f"[DEBUG] astar3d traj_len={len(traj_pts_xyz)} (cand)")
                except Exception:
                    pass
        # 若3D未能找到路径，回退到2D规划
        if not traj_pts_xyz:
            desired2 = args.follow_dist * args.standoff_mult
            p2_far = astar(occ, uav_xy, (gx_far, gy_far), meta)
            p2_near = astar(occ, uav_xy, (gx, gy), meta)
            cands2 = []
            for pth in (p2_far or [], p2_near or []):
                if pth and len(pth) >= 2:
                    nx, ny = pth[min(1, len(pth)-1)]
                    err = abs(math.hypot(hx - nx, hy - ny) - desired2)
                    cands2.append((err, len(pth), pth))
            if not cands2:
                p_ret2d = astar(occ, uav_xy, (hx, hy), meta)
                if p_ret2d and len(p_ret2d) >= 2:
                    traj_pts_xyz = [(float(x), float(y), float(uav_z)) for (x,y) in p_ret2d]
                    if getattr(args, 'debug', False):
                        try:
                            print(f"[DEBUG] astar2d traj_len={len(p_ret2d)} (fallback from 3d)")
                        except Exception:
                            pass
            else:
                cands2.sort(key=lambda t: (t[0], t[1]))
                psel2 = cands2[0][2]
                traj_pts_xyz = [(float(x), float(y), float(uav_z)) for (x,y) in psel2]
            if getattr(args, 'debug', False):
                try:
                    print(f"[DEBUG] astar2d traj_len={len(psel2)} (fallback cand)")
                except Exception:
                    pass
    else: # 默认使用astar 2D进行路径的获取
        
        ### 可视化
        if vis:
            visualize_occ_with_astar(occ,meta,gx_far,gy_far,gx, gy,hx,hy,uav_xy)

        # 2D 回退：固定高度
        p_ret2d = astar(occ, uav_xy, (hx, hy), meta)
        if not p_ret2d or len(p_ret2d) < 2:
            if getattr(args, 'debug', False):
                try:
                    print(f"[DEBUG] astar failed on strict occ from ({uav_xy[0]:.1f},{uav_xy[1]:.1f}) to ({hx:.1f},{hy:.1f})")
                except Exception:
                    pass
            if cv2 is not None:
                try:
                    k = np.ones((3,3), np.uint8)
                    occ_loose = cv2.erode(occ.astype(np.uint8), k)
                    p_relax = astar(occ_loose, uav_xy, (hx, hy), meta)
                    if p_relax and len(p_relax) >= 2:
                        p_ret2d = p_relax
                        if getattr(args, 'debug', False):
                            try:
                                print("[DEBUG] astar succeeded after obstacle shrinking")
                            except Exception:
                                pass
                except Exception:
                    pass
        if p_ret2d and len(p_ret2d) >= 2:
            traj_pts_xyz = [(float(x), float(y), float(uav_z)) for (x,y) in p_ret2d]
            if getattr(args, 'debug', False):
                try:
                    print(f"[DEBUG] astar2d traj_len={len(p_ret2d)} (strict or relaxed)")
                except Exception:
                    pass
    
    # 可视化astar规划出来的轨迹
    if vis:
        visualize_occ_with_trajectory(occ,meta,uav_xy,(hx,hy),traj_pts_xyz)

    if not traj_pts_xyz:
        if getattr(args, 'debug', False):
            try:
                print(f"[DEBUG] drop: empty trajectory after planning (target={target_name} cam={cam_id})")
            except Exception:
                pass
        return None

    horizon = max(1, int(args.traj_horizon))
    uav0_xyz = (float(uav_xy[0]), float(uav_xy[1]), float(uav_z))
    
    # 从traj_pts_xyz中获取horizon帧数的点(均匀采样)
    n = len(traj_pts_xyz)
    if n >= horizon:
        idxs = np.linspace(0, n - 1, num=horizon, dtype=int)
        traj_save_world = [traj_pts_xyz[i] for i in idxs]
    else:
        traj_save_world = traj_pts_xyz[:]
    
    # 如果规划出来的点不够，则用最后一个点补齐
    if len(traj_save_world) < horizon:
        last_pt = traj_save_world[-1]
        traj_save_world = traj_save_world + [last_pt] * (horizon - len(traj_save_world))
    traj_save_rel = [(float(x) - uav0_xyz[0], float(y) - uav0_xyz[1], float(z) - uav0_xyz[2]) for (x,y,z) in traj_save_world]
    # Rotate to Body0 (absolute-style in starting body frame) using the initial facing yaw
    try:
        yaw0_deg = float(math.degrees(math.atan2(hy - uav_xy[1], hx - uav_xy[0])))
    except Exception:
        yaw0_deg = 0.0
    traj_body0 = world_abs_to_body0(np.asarray(traj_save_rel, dtype=np.float32), yaw0_deg)
    rec = {
        'frame': 'placeholder.jpg',
        'bbox_xywh': [int(bbox_py[0]), int(bbox_py[1]), int(bbox_py[2]), int(bbox_py[3])],
        'traj_xyz': [[float(p[0]), float(p[1]), float(p[2])] for p in traj_body0.tolist()]
    }

    # 生成可视化图（可选）
    vis_img: Optional[np.ndarray] = None
    if getattr(args, 'save_traj_vis', False) and cv2 is not None:
        try:
            H, W = occ.shape
            scale = 4
            img_occ = (255 * (1 - occ.astype(np.uint8))).astype(np.uint8)
            vis = cv2.cvtColor(cv2.resize(img_occ, (W*scale, H*scale), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            # 目标位置
            ti, tj = world_to_grid(hx, hy, meta)
            if 0 <= ti < H and 0 <= tj < W:
                cx = int((tj + 0.5) * scale)
                cy = int((ti + 0.5) * scale)
                cv2.line(vis, (cx-4, cy), (cx+4, cy), (255, 0, 255), 2)
                cv2.line(vis, (cx, cy-4), (cx, cy+4), (255, 0, 255), 2)
            # UAV 规划轨迹
            pts = []
            for (uxx, uyy, uzz) in traj_save_world:
                ii, jj = world_to_grid(uxx, uyy, meta)
                if 0 <= ii < H and 0 <= jj < W:
                    px = int((jj + 0.5) * scale)
                    py = int((ii + 0.5) * scale)
                    pts.append((px, py))
            for a, b in zip(pts, pts[1:]):
                cv2.line(vis, a, b, (0,255,0), 2)
            for idx_vis, p in enumerate(pts):
                color = (0,255,255)
                if idx_vis == 0:
                    color = (255,0,0)
                elif idx_vis == len(pts)-1:
                    color = (0,0,255)
                cv2.circle(vis, p, 3, color, -1)
                cv2.putText(vis, str(idx_vis), (p[0]+3, p[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)
            vis_img = vis
        except Exception:
            vis_img = None

    print("[Success] 成功完成采样！")
    return rec, img_pre, vis_img

# ------------------------ Texture Utils ------------------------
# 新增：收集图片文件（root 与一层子目录）
def _gather_texture_files(root_dir: str) -> List[str]:
    files: List[str] = []
    try:
        if not os.path.isdir(root_dir):
            return []
        for name in os.listdir(root_dir):
            p = os.path.join(root_dir, name)
            if os.path.isfile(p) and name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                files.append(os.path.abspath(p))
        for name in os.listdir(root_dir):
            pdir = os.path.join(root_dir, name)
            if not os.path.isdir(pdir):
                continue
            for sub in os.listdir(pdir):
                f = os.path.join(pdir, sub)
                if os.path.isfile(f) and sub.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    files.append(os.path.abspath(f))
    except Exception:
        files = []
    return files

# 环境增强脚本(根据增强的等级，对环境内的分布，player纹理，光照，障碍物纹理，背景纹理，都进行随机化)
def environment_augmentation(ucv: Tracking, setting: dict, level: int, args: argparse.Namespace):
    """
    Apply procedural environment randomization based on a specified level.
    Uses methods from the Tracking class.
    """
    if level == 0:
        return

    # --- Prepare resources ---
    texture_files = []
    # 获取全部的texture文件
    if args.texture_root:
        # Get absolute paths for texture files
        texture_files = [os.path.abspath(f) for f in _gather_texture_files(args.texture_root)]
        texture_files = list(dict.fromkeys(texture_files))

    player_list = setting.get('players', []) # 被跟踪目标列表
    env_configs = setting.get("env", {}) # 环境列表
    objects_list = env_configs.get("objects", []) # 物体列表
    background_list = env_configs.get("backgrounds", []) # 背景列表
    light_list = env_configs.get("lights", []) # 光照列表
    
    # 【修改开始】
    # --- Apply randomization based on level ---
    placed_objects = []

    # Level 1: Layout and Player Mesh
    if level >= 1:
        if args.debug: print("[DEBUG] Randomizing layout (Level 1+)")
        try:
            # 清理上一轮的障碍物
            ucv.clean_obstacles()

            # 放置新的障碍物
            placed_objects = randomize_layout_simple(
                ucv, setting,
                tall_obstacles=args.tall_obstacles,
                xy_scale_range=tuple(args.obj_xy_scale),
                z_scale_range=tuple(args.obj_z_scale),
                verbose=args.debug
            )
            
            # 更新ucv对象中的障碍物列表，以便下一轮清理
            ucv.obstacles = placed_objects
        except Exception as e:
            if args.debug: print(f"[DEBUG] Failed to randomize layout: {e}")

        if args.debug: print("[DEBUG] Randomizing player mesh (Level 1+)")
        for obj in player_list:
            try:
                env_name = setting.get('env_name', '')
                if env_name == 'MPRoom':
                    map_id, spline = np.random.choice([2, 3, 6, 7, 9]), False
                else:
                    map_id, spline = np.random.choice([1, 2, 3, 4]), True
                ucv.set_appearance(obj, map_id, spline)
            except Exception as e:
                if args.debug: print(f"[DEBUG] Failed to set player mesh for {obj}: {e}")

    # Level 2: Player Texture
    if level >= 2:
        if args.debug: print("[DEBUG] Randomizing player texture (Level 2+)")
        if texture_files and player_list:
            try:
                for player in player_list:
                    # player需要被随机化的player，texture_files贴图文件路径，num=3选择3个贴图（对于一个player）
                    ucv.random_player_texture(player, texture_files, num=3) # 随机化玩家的外观材质
            except Exception as e:
                if args.debug: print(f"[DEBUG] Failed to randomize player texture: {e}")

    # Level 3: Light
    if level >= 3:
        if args.debug: print("[DEBUG] Randomizing lighting (Level 3+)")
        if light_list:
            try:
                ucv.random_lit(light_list) # 随机初始化场景光照
            except Exception as e:
                if args.debug: print(f"[DEBUG] Failed to randomize lighting: {e}")

    # Level 4: Obstacle Texture ONLY
    if level >= 4:
        if args.debug: print("[DEBUG] Randomizing obstacle textures (Level 4+)")
        # 只对当前已经放置的障碍物进行贴图
        if texture_files and placed_objects:
            try:
                # 随机化放置的物体
                 ucv.random_texture(placed_objects, texture_files, num=-1) # num=-1 表示全部
            except Exception as e:
                if args.debug: print(f"[DEBUG] Failed to randomize obstacle texture: {e}")

    # Level 5: Background Texture
    if level >= 5:
        if args.debug: print("[DEBUG] Randomizing background texture (Level 5+)")
        if texture_files and background_list:
            try:
                # 随机化背景的贴图
                ucv.random_texture(background_list, texture_files, num=-1) # num=-1 表示全部
            except Exception as e:
                if args.debug: print(f"[DEBUG] Failed to randomize background texture: {e}")
    # 【修改结束】

# ------------------------ Main ------------------------

def main():
    ap = argparse.ArgumentParser(description='Collect tracking data with human path planning and A* follower')
    ap.add_argument('--setting', required=True, help='env setting json (relative to envs/setting/)')
    ap.add_argument('--out', required=True, help='output dir')
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--samples-per-layout', type=int, default=50, help='在重新生成障碍物布局前，每个布局下每对采集的样本数量上限')
    ap.add_argument('--fps', type=int, default=10)
    # --- New Augmentation Control ---
    ap.add_argument('--randomization-level', type=int, default=0, help='环境随机化等级 (0-5)。0:无, 1:随机布局+玩家模型, 2:+玩家贴图, 3:+光照, 4:+障碍物贴图, 5:+背景贴图')
    ap.add_argument('--randomize-layout', action='store_true', help='[DEPRECATED] Use --randomization-level 4 or higher.')

    ap.add_argument('--cell', type=int, default=50)
    ap.add_argument('--inflate', type=int, default=80)
    ap.add_argument('--use-setting-objects', action='store_true')
    ap.add_argument('--tall-obstacles', action='store_true', help='随机生成更高的障碍（增加遮挡概率）')
    ap.add_argument('--obj-xy-scale', type=float, nargs=2, default=[0.6, 2.0], metavar=('MIN','MAX'), help='障碍物XY缩放范围')
    ap.add_argument('--obj-z-scale', type=float, nargs=2, default=[1.5, 3.5], metavar=('MIN','MAX'), help='障碍物Z缩放范围(高度)')
    ap.add_argument('--follow-dist', type=float, default=300.0)
    ap.add_argument('--standoff-mult', type=float, default=1.3, help='相对跟随距离的安全放大倍数，用于保持更远距离')
    ap.add_argument('--traj-horizon', type=int, default=6, help='保存的规划轨迹的步数（不执行，仅记录）')
    ap.add_argument('--yaw-jitter-deg', type=float, default=8.0, help='相机绕Z轴最大随机偏转(度)，朝远离障碍的一侧')
    ap.add_argument('--save-traj-vis', action='store_true', help='保存每条样本的2D占用图叠加短轨迹可视化')
    ap.add_argument('--pair-vis-dir', type=str, default=None, help='保存遮挡选位可视化的目录（默认保存在每个episode目录）')
    ap.add_argument('--texture-root', type=str, default='gym_unrealcv/envs/UnrealEnv/textures', help='贴图根目录（相对ROOT_DIR或绝对路径）')
    ap.add_argument('--texture-element', type=int, default=-1, help='材质元素索引（-1 表示随机）')
    # 可见度门控阈值
    ap.add_argument('--vis-threshold', type=float, default=0.35, help='可见度筛选阈值，超过此比值的样本将被丢弃')
    # UAV constraints
    # 移除安全距离膨胀相关参数（仅保留构图时膨胀）
    ap.add_argument('--uav-height', type=float, default=None, help='无人机飞行高度（不填则用setting.height）')
    # 3D体素占用（可选）
    ap.add_argument('--use-occ3d', action='store_true', help='启用三维体素占用与三维A*')
    ap.add_argument('--cell-z', type=float, default=100.0, help='体素Z尺寸（世界单位）')
    ap.add_argument('--inflate-z', type=float, default=120.0, help='Z向膨胀（世界单位）')
    ap.add_argument('--z-min', type=float, default=None)
    ap.add_argument('--z-max', type=float, default=None)
    # 多对并行
    ap.add_argument('--num-pairs', type=int, default=1, help='并行采集的人-机对数量')
    # 调试开关
    ap.add_argument('--debug', action='store_true', help='enable verbose debug prints')
    ap.add_argument('--log-interval', type=int, default=20, help='print debug info every N steps')
    ap.add_argument('--save-debug', action='store_true', help='save debug.jsonl with per-step info')
    
    ap.add_argument('--seed', type=int, default=1, help='随机种子')
    args = ap.parse_args()

    # 设置随机种子，保证实验可复现
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    # 导入环境的settings(在gym_unrealcv/envs/setting/tracking下）
    setting = misc.load_env_setting(args.setting)
    # 若未指定高度，从setting中读取默认高度（例如 FlexibleRoom.json 的 230）
    try:
        if args.uav_height is None:
            args.uav_height = float(setting.get('height', 230)) # 默认高度230
    except Exception:
        pass

    # 为可见度门控与无遮挡bbox回退准备障碍白名单
    try:
        env_cfg = (setting.get('env') or {})
        obj_list = list(env_cfg.get('objects') or []) # 获取可放置在场景中的物体
        bkg_list = list(env_cfg.get('backgrounds') or []) # 获取场景中的背景
        args.obstacles_whitelist = obj_list + bkg_list
    except Exception:
        args.obstacles_whitelist = []
    
    # 运行Unreal Engine的后端
    runner = env_unreal.RunUnreal(ENV_BIN=setting['env_bin'])
    ip, port = runner.start(docker=False, resolution=(320,240)) # 图像分辨率为320*240
    # Replace UnrealCv with Tracking 创建一个Tracking任务类(用于管理整个任务)---client端
    ucv = Tracking(port=port, ip=ip, env=runner.path2env, cam_id=0, resolution=(320,240))
    
    # 添加此部分代码来初始化障碍物字典，解决AttributeError
    try:
        objects_list_for_init = (setting.get('env') or {}).get('objects') or []
        if objects_list_for_init:
            ucv.init_objects(objects_list_for_init) # 初始化障碍物
    except Exception as e:
        if args.debug:
            print(f"[DEBUG] 警告: 初始化障碍物字典失败: {e}")
    

    if args.debug:
        print(f"[DEBUG] env started pid on port={port}")
        print(f"[DEBUG] setting: env_bin={setting.get('env_bin')}, reset_area={setting.get('reset_area')}")

    # 目标名称：默认获取第一个元素的人作为目标
    target_name = (setting.get('players') or ['targetbp'])[0] # 默认setting.get('players')为['targetbp', 'targetbp2']
    if args.debug:
        print(f"[DEBUG] players={setting.get('players')}, target={target_name}")

    # 若仅单对，保留一个人；否则为多对模式，保留所有人
    if int(getattr(args, 'num_pairs', 1)) <= 1:  # 单人模式
        try:
            players = setting.get('players') or []
            others = [p for p in players if p != target_name]
            for o in others:
                try:
                    ucv.hide_obj(o)  # 将除了target之外的另外的player直接隐藏
                    if args.debug:
                        print(f"[DEBUG] hide {o}")
                except Exception:
                    try:
                        ra = setting.get('reset_area', [-1e3,1e3,-1e3,1e3,0,500])
                        far_loc = [ra[1] + 10000, ra[3] + 10000, ra[5] + 1000]
                        ucv.set_obj_location(o, far_loc)
                        if args.debug:
                            print(f"[DEBUG] move {o} to far {far_loc}")
                    except Exception:
                        try:
                            ucv.client.request(f'vset /object/{o}/scale 0.001 0.001 0.001', -1)
                            if args.debug:
                                print(f"[DEBUG] scale {o} to tiny")
                        except Exception:
                            pass
        except Exception:
            pass

    # 多对初始化：补足玩家与相机，建立 pairs 列表
    players_all = list((setting.get('players') or []))
    if args.debug:
        print(f"[DEBUG] players(all)={players_all}")

    try:
        ra = setting.get('reset_area', [-1000,1000,-1000,1000,0,500])
        def rand_loc(): # 在reset_area中获取随机位置
            return [random.uniform(ra[0]+100, ra[1]-100),
                    random.uniform(ra[2]+100, ra[3]-100),
                    float(args.uav_height or (ra[4]+ra[5])*0.5)]
        while len(players_all) < int(getattr(args, 'num_pairs', 1)):
            idx = len(players_all)
            name = f'target_auto_{idx}'
            ucv.new_obj('target_C', name, rand_loc()) # 将target随机放置在场景中
            players_all.append(name) # 将随机放置的target添加到list里面
            if args.debug:
                print(f"[DEBUG] spawn player {name}")
    except Exception as e:
        if args.debug:
            print(f"[DEBUG] warn: spawn players failed: {e}")

    try:
        # 如果相机数量不够，则新增相机
        while ucv.get_camera_num() < int(getattr(args, 'num_pairs', 1)):
            ucv.new_camera()
        if args.debug:
            print(f"[DEBUG] cameras={ucv.get_camera_num()}")
    except Exception as e:
        if args.debug:
            print(f"[DEBUG] warn: spawn cameras failed: {e}")
    # 构建target名称和camera_id的pair
    pairs = []
    for i in range(int(getattr(args, 'num_pairs', 1))):
        pairs.append((players_all[i], i))  # (target_name, cam_id)

    # 构建 object mask 的颜色映射
    try:
        ucv.build_color_dic([p for (p, _) in pairs])
    except Exception:
        pass

    if args.debug:
        print(f"[DEBUG] pairs={pairs}")

    # 准备3D占用函数
    build3d = None
    meta3d = None
    occ3d = None
    if args.use_occ3d: # 默认不使用3D的占用地图
        try:
            from tools.gen_occupancy import build_occupancy3d_from_bounds
            build3d = build_occupancy3d_from_bounds
        except Exception:
            build3d = None
            if args.debug:
                print("[WARN] build_occupancy3d_from_bounds not available, fallback to 2D")

    try:
        for ep in range(args.episodes): # 采集args.episodes条数据
            ep_dir = os.path.join(args.out, f'ep{ep:03d}')
            os.makedirs(ep_dir, exist_ok=True)
            if args.debug:
                print(f"[DEBUG] === Episode {ep} ===")

            # --- Centralized Environment Randomization ---
            placed_objects = []
            try:
                environment_augmentation(ucv, setting, args.randomization_level, args)
                # try to fetch placed objects from ucv
                placed_objects = getattr(ucv, 'obstacles', []) or []
            except Exception:
                placed_objects = []
            time.sleep(0.3)  # Allow changes to take effect


            # 更新可见度白名单：合并（已放置 ∪ setting.objects ∪ setting.backgrounds）
            try:
                env_cfg = setting.get('env') or {}
                env_objs = list(env_cfg.get('objects') or [])
                env_bkgs = list(env_cfg.get('backgrounds') or [])
                merged = list(dict.fromkeys((placed_objects or []) + env_objs + env_bkgs))
                if merged:
                    args.obstacles_whitelist = merged
                    if args.debug:
                        print(f"[DEBUG] obstacles_whitelist merged: placed={len(placed_objects or [])} setting={len(env_objs)+len(env_bkgs)} total={len(merged)}")
            except Exception:
                pass

            # 2D 占用 & 安全占用
            # TODO: 这里会得到全1的occ map
            occ, meta = build_occupancy2d(ucv, setting, cell=args.cell, inflate=args.inflate, use_whitelist=args.use_setting_objects)
            try:
                np.savez(os.path.join(ep_dir, 'occupancy2d.npz'), occ=occ, meta=meta)
            except Exception:
                pass
            if args.debug:
                try:
                    print(f"[DEBUG] occ2d shape={occ.shape} mean={occ.mean():.3f} ones={int(occ.sum())}")
                except Exception:
                    pass
            # 仅使用构图时膨胀，不再计算安全距离占用

            ####### 使用 3D的 占用（可选）: 默认直接使用occ 2d
            occ3d = None
            if build3d is not None:
                z_override = None
                if args.z_min is not None and args.z_max is not None:
                    z_override = (args.z_min, args.z_max)
                inflate_z_eff = int(args.inflate_z * (1.5 if args.tall_obstacles else 1.0))
                occ3d, meta3d = build3d(
                    ucv,
                    reset_area=setting['reset_area'],
                    cell_xy=int(args.cell),
                    cell_z=int(args.cell_z),
                    inflate_xy=int(args.inflate),
                    inflate_z=inflate_z_eff,
                    ignore_names=("floor","FLOOR","ground","Ground","Landscape","LandscapeProxy","wall","Wall","WALL","Room","room","Sky","sky","SkySphere","Ceiling","ceiling"),
                    verbose=False,
                    objects_whitelist=(setting.get('env') or {}).get('objects'),
                    max_vol_ratio=0.98,
                    debug_topk=0,
                    z_override=z_override,
                )
                # 仅使用构图时膨胀，不再执行安全距离膨胀
                try:
                    np.savez(os.path.join(ep_dir, 'occupancy3d.npz'), occ3d=occ3d, meta3d=meta3d)
                except Exception:
                    pass
                if args.debug:
                    try:
                        print(f"[DEBUG] occ3d shape={occ3d.shape} mean={occ3d.mean():.3f}")
                    except Exception:
                        pass

            # debug 文件
            fdbg = None
            if args.save_debug:
                fdbg = open(os.path.join(ep_dir, 'debug.jsonl'), 'w')

            labels_path = os.path.join(ep_dir, 'labels.jsonl')
            with open(labels_path, 'w') as f:
                # 对每个布局，每对采集 samples_per_layout 条，共 num_pairs * samples_per_layout
                for sub_ep in range(int(args.samples_per_layout)):
                    for p_idx, (target_name, cam_id) in enumerate(pairs):
                        # 根据当前的 occ_map，获取行人和无人机的位置
                        (human_pos, uav_pos, human_pos_occ, uav_pos_occ) = find_occluded_initial_pair_heuristic(
                            occ, meta, follow_dist=float(args.follow_dist),
                            vis_dir=(args.pair_vis_dir or ep_dir),
                            vis_prefix=f'sample_{ep:03d}_{sub_ep:04d}_p{p_idx}_pair'
                        )
                        if human_pos is None or uav_pos is None:
                            if args.debug:
                                try:
                                    print(f"[DEBUG] skip pair={p_idx}: no occluded pair at sub_ep={sub_ep}")
                                except Exception:
                                    pass
                            continue

                        hx, hy = human_pos
                        ux, uy = uav_pos
                        base_h = float(args.uav_height) if args.uav_height is not None else float((setting.get('reset_area',[0,0,0,0,130,130])[4] + setting.get('reset_area',[0,0,0,0,130,130])[5]) * 0.5)

                        # 设置此对的人与相机位置(仅包含位置，还未设置相机角度)
                        try:
                            ucv.set_obj_location(target_name, [hx, hy, base_h])
                        except Exception:
                            pass
                        try:
                            ucv.set_location(cam_id, [ux, uy, base_h])
                        except Exception:
                            pass
                        try:
                            time.sleep(0.03)
                        except Exception:
                            pass

                        # 采集单条样本
                        result_tuple = collect_single_sample(ucv, occ, meta, occ3d, meta3d, target_name, args, (hx, hy), sub_ep, cam_id=cam_id)
                        if result_tuple is None:
                            if args.debug:
                                print(f"[DEBUG] ✗ collect_single_sample返回None，样本被丢弃 (sub_ep={sub_ep}, pair={p_idx})")
                            continue
                        if args.debug:
                            print(f"[DEBUG] ✓ collect_single_sample成功 (sub_ep={sub_ep}, pair={p_idx})")
                        rec, img_data, vis_data = result_tuple

                        # 唯一命名，带上 pair 后缀
                        base_filename = f'sample_{ep:03d}_{sub_ep:04d}_p{p_idx}'
                        img_name = f'{base_filename}.jpg'
                        vis_name = f'{base_filename}_traj.png'
                        rec['frame'] = img_name

                        try:
                            # 1) 写 JSON
                            f.write(json.dumps(rec) + '\n')
                            f.flush()  # 立即刷新到磁盘
                            if args.debug:
                                print(f"[DEBUG] ✓ 成功保存样本 {img_name} 到 labels.jsonl")
                            # 2) 保存图像
                            if cv2 is not None and img_data is not None:
                                cv2.imwrite(os.path.join(ep_dir, img_name), img_data)
                                # 2b) 保存叠加bbox的检查图
                                try:
                                    img_dbg = img_data.copy()
                                    x, y, w, h = map(int, rec['bbox_xywh'])
                                    H_img, W_img = img_dbg.shape[0], img_dbg.shape[1]
                                    x = max(0, min(x, W_img - 1))
                                    y = max(0, min(y, H_img - 1))
                                    x2 = max(x, min(x + w - 1, W_img - 1))
                                    y2 = max(y, min(y + h - 1, H_img - 1))
                                    cv2.rectangle(img_dbg, (x, y), (x2, y2), (0, 255, 0), 2)
                                    cv2.imwrite(os.path.join(ep_dir, f'{base_filename}_bbox.jpg'), img_dbg)
                                except Exception:
                                    if getattr(args, 'debug', False):
                                        try:
                                            print(f"[DEBUG] warn: failed to save bbox vis for {base_filename}")
                                        except Exception:
                                            pass

                            # 3) 保存可视化（可选）
                            if getattr(args, 'save_traj_vis', False) and cv2 is not None and vis_data is not None:
                                cv2.imwrite(os.path.join(ep_dir, vis_name), vis_data)
                        except Exception as e:
                            if args.debug:
                                print(f"[DEBUG] ✗ 保存样本失败: {e}")
                            pass

            if args.save_debug and fdbg is not None:
                fdbg.close()

            # 已移除俯视图拼接

    finally:
        runner.close()


if __name__ == '__main__':
    main()