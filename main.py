#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

GYM_UNREALCV_PATH = os.path.expanduser("/home/ws0/gh/follow/gym-unrealcv/gym-unrealcv")
if os.path.isdir(GYM_UNREALCV_PATH) and GYM_UNREALCV_PATH not in sys.path:
    sys.path.insert(0, GYM_UNREALCV_PATH)

import time
import cv2
import argparse
import threading
import numpy as np
import torch

from pathlib import Path

# 视频/无人机/Unreal
from VIDEO.video import ThreadedCamera
from DRONE.drone_controller import TelloController
from VIDEO.unrealcv_stream import UnrealCVStream
from DRONE.unreal_sim_controller import UnrealSimController

# 检测与特征提取
from detectors import (
    initialize_yolo_model,
    initialize_ortrack_tracker,
    perform_yolo_detection_for_candidates,
)
from dinov3_feature_extractor import DINOv3FeatureExtractor

# 跟踪相关（首帧检测、可视化、参考特征、增强线程）
from tracking.first_frame import detect_object_yolo as detect_object_yolo_first_frame
from tracking.reference_store import ReferenceFeatureStore
from tracking.visualization import (
    plot_and_save_if_neded,
    save_follow_stream_frame,
    save_detection_debug_image,
)
from tracking.workers import OnlineEnhancerWorker, YOLO_LOCK as YOLO_LOCK_SHARED
from tracking.ortrack_core import track_object_with_ortrack

# DP
from dp import preload_dp_if_needed

# 工具
from utils import read_and_log
from config import (
    DEFAULT_YOLO_MODEL,
    DEFAULT_ORTRACK_CHECKPOINT,
    DEFAULT_DINOV3_CHECKPOINT,
    YOLO_WEIGHTS_DIR,
    ORTRACK_WEIGHTS_DIR,
    DINOV3_WEIGHTS_DIR,
    ensure_directories,
)


# ========== 全局变量 ==========
mission_counter = 0

# ========== 全局锁（YOLO并发保护） ==========
YOLO_LOCK = YOLO_LOCK_SHARED


# ========== 参数 ==========
parser = argparse.ArgumentParser(description='YOLO + DINOv3 + ORTrack pipeline (modular entry)')

# 视频/可视化
parser.add_argument('--path_to_video', default='video/whales.mp4', help='输入视频/流/目录')
parser.add_argument('--desired_height', default=240, type=int, help='输入高度')
parser.add_argument('--desired_width', default=320, type=int, help='输入宽度')
parser.add_argument('--fps', default=0, type=float, help='读取帧率，>0 则用线程读取')
parser.add_argument('--wait_key', default=30, type=int, help='可视化 waitKey')
parser.add_argument('--plot_visualizations', action='store_true', default=False, help='是否显示窗口')
parser.add_argument('--save_images_to', default=False, help='保存可视化图像的目录')
parser.add_argument('--save_video_to', type=str, default='', help='保存带检测/跟踪框的视频路径(.mp4或.avi)')
parser.add_argument('--video_writer_fps', type=float, default=None, help='强制设置VideoWriter FPS（默认自动探测或25）')

# UnrealCV 可复现随机
parser.add_argument('--unreal_seed', type=int, default=None, help='UnrealCV 环境随机种子（固定人物起点与路线）')

# 评测
parser.add_argument('--eval_table1', action='store_true', default=False, help='按 Table 1 统计 AR/EL/SR')
parser.add_argument('--episodes', type=int, default=100, help='评测每个环境的 episode 数')
parser.add_argument('--distractors', type=int, default=None, help='干扰者数量 D（若环境支持动态配置）')

# 外形参数（固定目标与干扰项的外形 ID）
parser.add_argument('--target_appearance_id', type=int, default=2, help='目标外形ID（set_appearance id）')
parser.add_argument('--distractor_appearance_id', type=int, default=4, help='干扰项外形ID（set_appearance id）')

# YOLO
parser.add_argument('--text_query', default='', help='YOLOE 文本提示（如："person car"）')
parser.add_argument('--yolo_model_path', type=str, default=DEFAULT_YOLO_MODEL, help='YOLO 模型权重路径（默认 weights/yolo 下）')
parser.add_argument('--yolo_confidence_thresh', type=float, default=0.25, help='YOLO 置信度阈值')
parser.add_argument('--max_detection_attempts', type=int, default=300, help='首帧检测最大尝试帧数')

# ORTrack
parser.add_argument('--ortrack_prj_path', type=str, default='ORTrack', help='ORTrack 项目路径')
parser.add_argument('--ortrack_model_name', type=str, default='deit_tiny_patch16_224', help='ORTrack 模型名')
parser.add_argument('--ortrack_checkpoint_path', type=str, default=None, help='ORTrack 权重路径，None 用默认')
parser.add_argument('--ortrack_lost_thresh', type=float, default=0.6, help='ORTrack 置信度丢失阈值')
parser.add_argument('--ortrack_max_lost_frames', type=int, default=3, help='ORTrack 连续丢失最大帧数')
parser.add_argument('--ortrack_template_size', type=int, default=128, help='ORTrack 模板尺寸')
parser.add_argument('--ortrack_search_size', type=int, default=256, help='ORTrack 搜索尺寸')

# 匹配重检测（DINOv3）
parser.add_argument('--reference_feature_path', type=str, help='预先提取的参考特征 (.pt)')
parser.add_argument('--match_similarity_thresh', type=float, default=0.75, help='匹配相似度阈值')
parser.add_argument('--dinov3_model_name', type=str, default='dinov3_vith16plus', help='DINOv3 模型名称（如 dinov3_vith16plus、dinov3_vitb16 等）')
parser.add_argument('--dinov3_checkpoint_path', type=str, default=None, help='DINOv3 权重路径，默认为配置中的默认值')

# 在线参考特征增强
parser.add_argument('--online_enhance', action='store_true', default=False, help='启用在线参考特征增强（EMA）')
parser.add_argument('--enhance_interval', type=int, default=20, help='增强间隔（帧）')
parser.add_argument('--enhance_alpha', type=float, default=0.8, help='EMA 融合系数 alpha')
parser.add_argument('--enhance_iou_thresh', type=float, default=0.3, help='增强时YOLO匹配与ORTrack框的IoU阈值')
parser.add_argument('--enhance_crop_expand', type=float, default=0.2, help='局部YOLO检测时对ORTrack bbox的扩张比例（0使用整帧）')

# 丢失阶段并行找回（保留与原版本一致的配置入口）
parser.add_argument('--reacq_parallel', action='store_true', default=True, help='丢失阶段启用并行找回线程')
parser.add_argument('--reacq_interval', type=int, default=2, help='丢失阶段每隔N帧入队一次用于找回')
parser.add_argument('--reacq_crop_expand', type=float, default=0.4, help='找回时对预测框的扩张比例（0表示全帧）')
parser.add_argument('--reacq_iou_with_pred', type=float, default=0.2, help='找回时候选与预测框的最小IoU')
parser.add_argument('--reacq_match_thresh', type=float, default=0.70, help='丢失阶段相似度阈值')
parser.add_argument('--reacq_fullframe_after_n', type=int, default=3, help='ROI失败N次后尝试一次全帧找回')

# 无人机
parser.add_argument('--fly_drone', default=False, action='store_true', help='是否连接无人机')
parser.add_argument('--auto_takeoff', action='store_true', default=False, help='连接成功后自动起飞')

# 控制策略与参数（保留原参数项以兼容）
parser.add_argument('--target_area_ratio', type=float, default=0.12, help='目标框面积占全帧比例的设定')
parser.add_argument('--area_tolerance_ratio', type=float, default=0.20, help='对 target_area 的相对容差比例')
parser.add_argument('--debug_control', action='store_true', default=False, help='打印控制调试信息')

parser.add_argument('--ctrl_yaw_kp', type=float, default=0.45)
parser.add_argument('--ctrl_yaw_ki', type=float, default=0.0)
parser.add_argument('--ctrl_yaw_kd', type=float, default=0.1)
parser.add_argument('--ctrl_vz_kp', type=float, default=0.5)
parser.add_argument('--ctrl_vz_ki', type=float, default=0.0)
parser.add_argument('--ctrl_vz_kd', type=float, default=0.05)
parser.add_argument('--ctrl_vy_kp', type=float, default=0.35)
parser.add_argument('--ctrl_vy_ki', type=float, default=0.0)
parser.add_argument('--ctrl_vy_kd', type=float, default=0.15)
parser.add_argument('--ctrl_dt', type=float, default=0.1, help='PID 控制周期 dt 秒')
parser.add_argument('--ctrl_max_speed', type=int, default=40, help='速度上限 |v|,|yaw|')
parser.add_argument('--ctrl_low_battery', type=int, default=15, help='低电阈值(%)')
parser.add_argument('--ctrl_vy_min_output', type=float, default=0.8, help='前后速度反死区的最小输出幅值')

# 卡尔曼/预测
parser.add_argument('--kf_history_len', type=int, default=8, help='历史长度，用于鲁棒速度估计')
parser.add_argument('--kf_max_predict_frames', type=int, default=15, help='预测阶段的最大帧数')
parser.add_argument('--kf_mode', type=str, default='confaware', choices=['none', 'standard', 'confaware'])

# DP（保持一致）
parser.add_argument('--dp_enable', action='store_true', default=False, help='丢失后启用Diffusion Policy规划补救')
parser.add_argument('--dp_repo_root', type=str, default='/home/ws0/gh/follow/gym-unrealcv/gym-unrealcv/diffusion_policy-main')
parser.add_argument('--dp_checkpoint', type=str, default='/home/ws0/gh/follow/gym-unrealcv/gym-unrealcv/diffusion_policy-main/run_20k/checkpoints/epoch=0060-val_loss=0.0204.ckpt')
parser.add_argument('--dp_action_exec_steps', type=int, default=None, help='DP：每次规划仅执行前N步(Ta)，None/0执行完整段')
parser.add_argument('--dp_max_plans', type=int, default=0, help='DP补救：单次最多规划次数(>0生效)，0表示不限')
parser.add_argument('--dp_dt', type=float, default=1, help='DP阶段控制步长 dt（秒），留空则使用 --ctrl_dt')

args = parser.parse_args()


def create_video_from_images(cfg):
    img_array = []
    fileidx = 0
    while True:
        filename = '{}/{}.jpg'.format(cfg['path_to_video'], fileidx)
        if not os.path.exists(filename):
            break
        img = cv2.imread(filename)
        if img is None:
            break
        img_array.append(img)
        fileidx += 1

    if len(img_array) == 0:
        raise FileNotFoundError('No images found in directory to create video.')

    out_path = '{}/video_from_images.avi'.format(cfg['path_to_video'])
    h, w = img_array[0].shape[:2]
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (w, h))
    for im in img_array:
        out.write(im)
    out.release()
    cfg['path_to_video'] = out_path
    return cv2.VideoCapture(cfg['path_to_video'])


def init_system():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    cfg = vars(args)

    # 确保权重目录存在
    try:
        ensure_directories()
    except Exception:
        pass

    # ORTrack
    print("Initializing ORTrack tracker...")
    tracker = initialize_ortrack_tracker(
        ortrack_prj_path=cfg['ortrack_prj_path'],
        model_name=cfg['ortrack_model_name'],
        checkpoint_path=cfg['ortrack_checkpoint_path'],
        lost_threshold=cfg['ortrack_lost_thresh'],
        max_lost_frames=cfg['ortrack_max_lost_frames'],
        template_size=cfg['ortrack_template_size'],
        search_size=cfg['ortrack_search_size'],
        keep_last_position=True
    )

    # YOLO
    print("Initializing YOLO detector...")
    target_classes_list = None
    if cfg.get('text_query'):
        target_classes_list = cfg['text_query'].strip().split()
    # 若未找到指定 YOLO 权重，则回退到 weights/yolo 下的默认文件（并允许自动下载到该目录）
    yolo_weight_path = cfg.get('yolo_model_path') or DEFAULT_YOLO_MODEL
    if not os.path.exists(yolo_weight_path):
        # 确保以 weights/yolo 下的文件名作为下载位置
        try:
            os.makedirs(YOLO_WEIGHTS_DIR, exist_ok=True)
        except Exception:
            pass
        # 若用户传了文件名或空，则使用默认文件名
        if os.path.isdir(yolo_weight_path) or (not yolo_weight_path.endswith('.pt')):
            yolo_weight_path = DEFAULT_YOLO_MODEL
        print(f"[YOLO] weight not found. Will download/use at: {yolo_weight_path}")
    detector = initialize_yolo_model(model_path=yolo_weight_path, target_classes=target_classes_list)

    # DINOv3
    print("Initializing DINOv3 feature extractor (for re-detection matching)...")
    dinov3_model_name = cfg.get('dinov3_model_name') or 'dinov3_vith16plus'
    dinov3_checkpoint_path = cfg.get('dinov3_checkpoint_path')
    if dinov3_checkpoint_path:
        dinov3_checkpoint_path = os.path.expanduser(dinov3_checkpoint_path)
    feature_extractor = DINOv3FeatureExtractor(
        model_name=dinov3_model_name,
        checkpoint_path=dinov3_checkpoint_path,
        device=device,
    )

    # 预加载 Diffusion Policy（若启用），避免首次 DP 触发时重复加载或 policy=None
    try:
        preload_dp_if_needed(cfg)
    except Exception as e:
        print(f"[DP] preload failed: {e}")

    # 视频源
    print("Init video...")
    if isinstance(cfg["path_to_video"], str) and cfg["path_to_video"].startswith("unrealcv:"):
        env_id = cfg["path_to_video"].split("unrealcv:", 1)[1]
        video = UnrealCVStream(env_id, width=cfg['desired_width'], height=cfg['desired_height'])
        # 应用人数与种子
        try:
            base = getattr(video.env, 'unwrapped', video.env)
            if cfg.get('distractors') is not None and hasattr(base, 'player_num'):
                d = max(0, int(cfg['distractors']))
                base.player_num = int(2 + d)
            if cfg.get('unreal_seed') is not None and hasattr(base, 'seed'):
                base.seed(int(cfg['unreal_seed']))
            if hasattr(base, 'reset'):
                base.reset()
        except Exception:
            pass
    elif os.path.isdir(cfg["path_to_video"]):
        video = create_video_from_images(cfg)
    elif os.path.exists(cfg["path_to_video"]) and cfg['fps'] < 1:
        video = cv2.VideoCapture(cfg["path_to_video"])
    else:
        video = ThreadedCamera(cfg["path_to_video"], fps=cfg['fps'])

    # 无人机
    if cfg["fly_drone"]:
        vehicle = TelloController(cfg['desired_width'], cfg['desired_height'])
        print("Init Tello Drone...")
        for _ in range(3):
            if vehicle.connect():
                break
        else:
            print("连接Tello失败")
            sys.exit(1)
        vehicle.start_video()
        if cfg.get('auto_takeoff', False):
            try:
                vehicle.takeoff()
                time.sleep(0.5)
                while(vehicle.get_height() < 100):
                    vehicle.send_velocity(0, 0, 20, 0)
                    time.sleep(0.5)
                vehicle.send_velocity(0, 0, 0, 0)
            except Exception as e:
                print(f"Tello takeoff failed: {e}")

        class TelloVideoStream:
            def __init__(self, controller, maxq=8):
                import threading, queue
                self.controller = controller
                self.q = queue.Queue(maxsize=maxq)
                self._stop = False
                self.t = threading.Thread(target=self._loop, name="TelloVideoStream", daemon=True)
                self.t.start()
            def _loop(self):
                import time
                while not self._stop:
                    frame = self.controller.get_frame()
                    try:
                        if self.q.full():
                            self.q.get_nowait()
                        self.q.put_nowait(frame)
                    except Exception:
                        pass
                    time.sleep(0)
            def read(self):
                import queue
                try:
                    frame = self.q.get(timeout=0.02)
                    return (frame is not None), frame
                except queue.Empty:
                    return False, None
            def close(self):
                self._stop = True
        video = TelloVideoStream(vehicle)
    else:
        vehicle = None
        # Unreal 模拟控制器（当视频源为 UnrealCV 时）
        try:
            if isinstance(video, UnrealCVStream):
                tracker_index = getattr(video.env, 'tracker_id', 0)
                vehicle = UnrealSimController(video.env, tracker_index=tracker_index, max_speed=99)
        except Exception:
            pass

    # 创建保存目录（与 follow2.py 保持一致）
    if cfg['save_images_to']:
        # 仅创建根目录；各可视化模块在需要时自行创建子目录（避免产生无用文件夹）
        os.makedirs(cfg['save_images_to'], exist_ok=True)

    # 视频写入器（懒加载为主；此处先尝试初始化）
    cfg['video_writer'] = None
    cfg['video_writer_size'] = None
    cfg['video_writer_fps'] = None
    if cfg.get('save_video_to'):
        tmp_cap = None
        src_is_thread = isinstance(video, ThreadedCamera)
        if not src_is_thread:
            tmp_cap = video
        if tmp_cap is not None and hasattr(tmp_cap, 'get'):
            width = int(tmp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(tmp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = tmp_cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 1:
                fps = 25.0
            size = (max(1, width), max(1, height))
        else:
            size = (cfg['desired_width'], cfg['desired_height'])
            fps = cfg['fps'] if cfg['fps'] > 0 else 25.0
        fourcc = cv2.VideoWriter_fourcc(*('mp4v' if cfg['save_video_to'].lower().endswith('.mp4') else 'XVID'))
        try:
            vw = cv2.VideoWriter(cfg['save_video_to'], fourcc, fps, size)
            if vw is not None and vw.isOpened():
                cfg['video_writer'] = vw
                cfg['video_writer_size'] = size
                cfg['video_writer_fps'] = fps
                print(f"VideoWriter initialized: {cfg['save_video_to']} at {size} @ {fps}fps")
            else:
                print(f"Failed to open VideoWriter at {cfg['save_video_to']}")
        except Exception as e:
            print(f"Create VideoWriter error: {e}")

    # 固定可控体数量与外形
    try:
        if isinstance(video, UnrealCVStream):
            base = getattr(video.env, 'unwrapped', video.env)
            if hasattr(base, 'controable_agent'):
                base.controable_agent = 1
            if hasattr(base, 'unrealcv') and hasattr(base, 'player_list'):
                ucv = base.unrealcv
                players = list(getattr(base, 'player_list', []))
                # 目标（索引1）固定外形 id=2
                if len(players) >= 2:
                    try:
                        ucv.set_appearance(players[1], id=int(cfg.get('target_appearance_id', 2)), spline=True)
                    except Exception:
                        try:
                            ucv.set_appearance(players[1], id=int(cfg.get('target_appearance_id', 2)), spline=False)
                        except Exception:
                            pass
                # 干扰项（索引>=2）固定外形 id=4
                for name in players[2:]:
                    try:
                        ucv.set_appearance(name, id=int(cfg.get('distractor_appearance_id', 4)), spline=True)
                    except Exception:
                        try:
                            ucv.set_appearance(name, id=int(cfg.get('distractor_appearance_id', 4)), spline=False)
                        except Exception:
                            pass
    except Exception:
        pass

    # 预加载DP（若启用）
    try:
        if cfg.get('dp_enable', False):
            preload_dp_if_needed(cfg)
    except Exception:
        pass

    return device, cfg, tracker, detector, video, vehicle, feature_extractor


def start_mission(device, cfg, tracker, detector, video, vehicle, feature_extractor, step_cb=None, eval_state=None):
    global mission_counter
    mission_counter += 1
    # 将 mission_counter 放入 cfg 供可视化函数使用
    cfg['mission_counter'] = mission_counter

    ref_store = ReferenceFeatureStore()

    # 首帧检测（使用重构模块的首帧检测函数）
    bounding_boxes, masks, saved_frame, reference_feature = detect_object_yolo_first_frame(
        cfg, detector, video, feature_extractor,
        read_and_log, perform_yolo_detection_for_candidates,
        save_follow_stream_frame, save_detection_debug_image, plot_and_save_if_neded,
        YOLO_LOCK
    )
    if not bounding_boxes:
        print("Initial detection failed. No bounding boxes found.")
        return

    if cfg.get('reference_feature_path') and reference_feature is None:
        print("Error: reference_feature_path provided but feature could not be loaded or no match found.")
        return

    if cfg.get('reference_feature_path'):
        ref_store.initialize_from_list(reference_feature)

    enhancer = None
    if cfg.get('online_enhance', False):
        enhancer = OnlineEnhancerWorker(detector, feature_extractor, ref_store, cfg)
        enhancer.start()

    initial_bbox_xywh = bounding_boxes[0]
    initial_frame_rgb = cv2.cvtColor(saved_frame, cv2.COLOR_BGR2RGB)

    # 跟踪主循环（使用模块化实现）
    status = track_object_with_ortrack(
        tracker,
        initial_bbox_xywh,
        initial_frame_rgb,
        video,
        cfg,
        detector,
        feature_extractor,
        reference_feature,
        vehicle,
        ref_store,
        enhancer,
        step_cb=step_cb
    )

    if enhancer is not None:
        try:
            enhancer.stop()
        except Exception:
            pass

    if status == 'REDETECT':
        print("Tracker requested full re-detection cycle. Restarting mission...")
        return start_mission(device, cfg, tracker, detector, video, vehicle, feature_extractor, step_cb=step_cb, eval_state=eval_state)

    if cfg.get('video_writer') is not None:
        try:
            cfg['video_writer'].release()
            print(f"Saved video to: {cfg['save_video_to']}")
        except Exception:
            pass

    return {'status': status}


if __name__ == '__main__':
    vehicle = None
    try:
        device, cfg, tracker, detector, video, vehicle, feature_extractor = init_system()
        if cfg.get('eval_table1', False) and isinstance(video, UnrealCVStream):
            print("[INFO] Table1 评测逻辑已迁移到独立脚本 eval_table1.py。请使用：python eval_table1.py <同样的参数>")
        else:
            start_mission(device, cfg, tracker, detector, video, vehicle, feature_extractor)
    except KeyboardInterrupt:
        print("用户中断")
        try:
            if vehicle is not None:
                try:
                    vehicle.land()
                except Exception:
                    try:
                        vehicle.emergency_stop()
                    except Exception:
                        pass
        except Exception:
            pass
    finally:
        if vehicle is not None:
            try:
                vehicle.land()
            except Exception:
                pass
            try:
                vehicle.close()
            except Exception:
                pass




