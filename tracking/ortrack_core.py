"""ORTrack 跟踪主循环

此模块包含使用 ORTrack 进行目标跟踪的主要逻辑。
由于主函数非常大且复杂，暂时保留在 follow2.py 中，
后续可以逐步拆分重构。
"""

# 说明：
# track_object_with_ortrack 函数约520行，包含：
# - ORTrack 初始化和跟踪循环
# - 卡尔曼滤波预测
# - 丢失检测和找回逻辑
# - DP 补救机制
# - 在线特征增强
# - 控制命令计算
#
# 这个函数仍在 follow2.py (1269-1787行)
# 未来可以拆分为更小的子模块

import time
import copy
import cv2
import numpy as np
import torch

from kalman_filter import KalmanFilter
from control import PIDController, compute_drone_action_while_tracking as compute_action
from utils import read_and_log
from tracking.kalman_utils import init_pkf_bbox, conf_to_noise, estimate_velocity_robustly
from tracking.workers import OnlineEnhancerWorker, LostReacquireWorker
from tracking.detection import find_instance_by_features
from tracking.visualization import plot_and_save_if_neded, save_follow_stream_frame
from dp import (
    dp_reposition_and_try_reacquire as dp_rescue,
    get_dp_cache,
    build_dp_obs_from_frame,
    dp_action_to_tello_vel
)
from dp.dp_inference import _load_dp_policy as _load_dp_policy_fn


def _clip_box_xywh_to_frame(x, y, w, h, W, H):
    x = int(x); y = int(y); w = int(w); h = int(h)
    if x < 0:
        w += x; x = 0
    if y < 0:
        h += y; y = 0
    if x + w > W:
        w = max(1, W - x)
    if y + h > H:
        h = max(1, H - y)
    return x, y, max(1, w), max(1, h)


def track_object_with_ortrack(ortrack_tracker,
                              initial_bbox_xywh,
                              initial_frame_rgb,
                              video,
                              cfg,
                              detector=None,
                              feature_extractor=None,
                              reference_feature=None,
                              vehicle=None,
                              ref_store=None,
                              enhancer: OnlineEnhancerWorker | None = None,
                              step_cb=None):
    """使用 ORTrack 进行跟踪的主循环（模块化实现）"""
    print("Tracking with ORTrack...")
    try:
        ortrack_tracker.initialize(initial_frame_rgb, {'init_bbox': list(initial_bbox_xywh)})
    except Exception as e:
        print(f"Error initializing ORTrack with bbox {initial_bbox_xywh}: {e}")
        return "FAILED"

    # PID 控制器（本地实例）
    pid_controllers = {
        'yaw': PIDController(Kp=cfg.get('ctrl_yaw_kp', 0.45),
                             Ki=cfg.get('ctrl_yaw_ki', 0.0),
                             Kd=cfg.get('ctrl_yaw_kd', 0.1)),
        'vz': PIDController(Kp=cfg.get('ctrl_vz_kp', 0.5),
                            Ki=cfg.get('ctrl_vz_ki', 0.0),
                            Kd=cfg.get('ctrl_vz_kd', 0.05)),
        'vy': PIDController(Kp=cfg.get('ctrl_vy_kp', 0.35),
                            Ki=cfg.get('ctrl_vy_ki', 0.0),
                            Kd=cfg.get('ctrl_vy_kd', 0.15)),
    }

    frame_idx = 0
    lost_frames_count = 0
    max_lost_frames = int(cfg.get('ortrack_max_lost_frames', 3))
    lost_threshold = float(cfg.get('ortrack_lost_thresh', 0.6))

    kf = KalmanFilter()
    x0 = initial_bbox_xywh[0] + initial_bbox_xywh[2] / 2
    y0 = initial_bbox_xywh[1] + initial_bbox_xywh[3] / 2
    a0 = initial_bbox_xywh[2] / max(1, initial_bbox_xywh[3])
    h0 = initial_bbox_xywh[3]
    kf_mean, kf_cov = kf.initiate(np.array([x0, y0, a0, h0]))

    max_predict_frames = int(cfg.get('kf_max_predict_frames', 15))
    predict_frames_count = 0
    in_predict_mode = False

    # 并行找回线程控制
    reacq_worker: LostReacquireWorker | None = None
    reacq_interval = int(cfg.get('reacq_interval', 2))

    # PKF（丢失阶段平滑）
    pkf = None
    _pkf_mean = None
    _pkf_cov = None

    predict_frame_idx = 0
    history_len = int(cfg.get('kf_history_len', 8))
    center_history = []
    session_idx = 0
    predict_hold_sent = False
    pred_x = pred_y = pred_w = pred_h = None
    last_good_bbox = None

    # FPS 统计
    last_frame_time = time.time()
    fps_ema = None
    fps_alpha = float(cfg.get('fps_alpha', 0.2))
    fps_log_interval = int(cfg.get('fps_log_interval', 10))

    with torch.cuda.amp.autocast():
        while True:
            read_one_frame, frame_bgr_orig = read_and_log(video, cfg)
            if not read_one_frame:
                print("Video stream ended during ORTrack tracking.")
                break

            current_frame_bgr = frame_bgr_orig
            current_frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb_for_track = current_frame_rgb

            tracking_start_time = time.time()
            try:
                output = ortrack_tracker.track(frame_rgb_for_track)
            except Exception as e:
                print(f"ORTrack 跟踪器异常: {e}")
                if frame_idx == 0:
                    print("第一帧跟踪失败，尝试重新检测...")
                    if reacq_worker is not None:
                        try:
                            reacq_worker.stop()
                        except Exception:
                            pass
                    return "REDETECT"
                else:
                    if reacq_worker is not None:
                        try:
                            reacq_worker.stop()
                        except Exception:
                            pass
                    return "FAILED"
            print(f"ORTrack track time: {time.time() - tracking_start_time:.4f}s")

            pred_bbox_on_search_size = output['target_bbox']
            confidence = float(output.get('confidence', 1.0))
            print(f"ORTrack bbox: {pred_bbox_on_search_size}, conf: {confidence:.4f}")
            current_bbox_xywh = tuple(int(v) for v in pred_bbox_on_search_size)
            is_lost = confidence < lost_threshold

            if not in_predict_mode:
                if is_lost:
                    lost_frames_count += 1
                    print(f"[跟踪丢失尝试] is_lost=True, lost_frames_count={lost_frames_count}/{max_lost_frames}")
                else:
                    lost_frames_count = 0

                cx = current_bbox_xywh[0] + current_bbox_xywh[2] / 2
                cy = current_bbox_xywh[1] + current_bbox_xywh[3] / 2
                a = current_bbox_xywh[2] / max(1, current_bbox_xywh[3])
                h = current_bbox_xywh[3]
                measurement = np.array([cx, cy, a, h])
                mode = str(cfg.get('kf_mode', 'confaware')).lower()

                if mode == 'confaware':
                    if is_lost:
                        try:
                            t0 = time.time()
                            # 初始化（仅首次）
                            if pkf is None:
                                pkf = init_pkf_bbox(dt=1.0)
                                x0b, y0b, w0b, h0b = float(current_bbox_xywh[0]), float(current_bbox_xywh[1]), float(current_bbox_xywh[2]), float(current_bbox_xywh[3])
                                _pkf_mean = np.hstack([np.array([x0b, y0b, w0b, h0b], dtype=float), np.zeros(4, dtype=float)])
                                _pkf_cov = pkf.initial_state_covariance

                            # 构造 R（随置信度缩放）
                            R = np.eye(4) * conf_to_noise(float(confidence), max_noise=1.0, min_noise=0.0, k=20.0, center=0.5)
                            # PKF 更新
                            _pkf_mean, _pkf_cov = pkf.filter_update(
                                _pkf_mean, _pkf_cov,
                                observation=np.array([
                                    float(current_bbox_xywh[0]),
                                    float(current_bbox_xywh[1]),
                                    float(current_bbox_xywh[2]),
                                    float(current_bbox_xywh[3])
                                ], dtype=float),
                                observation_covariance=R
                            )
                            # 回写平滑后的 bbox（clamp）
                            sx, sy, sw, sh = [float(v) for v in _pkf_mean[:4]]
                            x_hat = int(round(sx))
                            y_hat = int(round(sy))
                            w_hat = int(round(max(1.0, sw)))
                            h_hat = int(round(max(1.0, sh)))
                            Hc, Wc = current_frame_bgr.shape[:2]
                            x_hat, y_hat, w_hat, h_hat = _clip_box_xywh_to_frame(x_hat, y_hat, w_hat, h_hat, Wc, Hc)
                            if w_hat >= 1 and h_hat >= 1:
                                current_bbox_xywh = (int(x_hat), int(y_hat), int(w_hat), int(h_hat))

                            # 同步内部 KF（xyah）用于预测阶段
                            try:
                                cx_sync = x_hat + w_hat / 2.0
                                cy_sync = y_hat + h_hat / 2.0
                                a_sync = (w_hat / max(1, h_hat))
                                h_sync = float(h_hat)
                                kf_mean, kf_cov = kf.update(kf_mean, kf_cov, np.array([cx_sync, cy_sync, a_sync, h_sync]))
                            except Exception:
                                pass

                            try:
                                print(f"[PKF] total={(time.time() - t0)*1000.0:.2f}ms conf={float(confidence):.3f}")
                            except Exception:
                                pass
                        except Exception:
                            pass
                    else:
                        # 正常跟踪：仅用观测更新内部 KF（保证预测状态连续）
                        kf_mean, kf_cov = kf.update(kf_mean, kf_cov, measurement)
                elif mode == 'standard':
                    kf_mean, kf_cov = kf.update(kf_mean, kf_cov, measurement)
                else:
                    # none：不更新 KF（纯用观测）
                    pass

                # 记录最近有效框（用于 kf_mode=none 的预测期回退）
                last_good_bbox = current_bbox_xywh

                center_history.append((cx, cy, a, h))
                if len(center_history) > history_len:
                    center_history.pop(0)

                # 在线增强：仅在未丢失时，按间隔入队
                if cfg.get('online_enhance', False) and enhancer is not None and (frame_idx % int(cfg.get('enhance_interval', 20)) == 0):
                    if ref_store is not None and ref_store.is_valid():
                        enhancer.enqueue(current_frame_bgr.copy(), current_bbox_xywh, frame_idx)

                # 进入预测模式
                if lost_frames_count >= max_lost_frames:
                    print(f"进入预测阶段，最大预测帧数: {max_predict_frames}")
                    in_predict_mode = True
                    predict_frames_count = 0
                    predict_frame_idx = 0
                    session_idx += 1

                    if str(cfg.get('kf_mode', 'confaware')).lower() != 'none':
                        vx, vy, va, vh = estimate_velocity_robustly(center_history)
                        print(f"[Kalman Init] 速度估计 (vx, vy)=({vx:.2f}, {vy:.2f})；将 va/vh 置零。")
                        va, vh = 0.0, 0.0
                        x0c, y0c, a0c, h0c = center_history[-1]
                        kf_mean, kf_cov = kf.initiate(np.array([x0c, y0c, a0c, h0c]), init_velocity=[vx, vy, va, vh])

                    # 启动并行找回
                    if cfg.get('reacq_parallel', True) and detector is not None and feature_extractor is not None:
                        if reacq_worker is None:
                            static_ref = reference_feature if isinstance(reference_feature, list) and len(reference_feature) > 0 else None
                            reacq_worker = LostReacquireWorker(detector, feature_extractor, ref_store, static_ref, cfg)
                            reacq_worker.start()

                # 基于当前检测到的框进行无人机控制
                area = current_bbox_xywh[2] * current_bbox_xywh[3]
                mean_point_for_drone = (int(cy), int(cx))
                compute_action(mean_point_for_drone, area, cfg, vehicle, pid_controllers, is_predict=False)

            elif in_predict_mode:
                mode = str(cfg.get('kf_mode', 'confaware')).lower()
                if mode != 'none':
                    kf_mean, kf_cov = kf.predict(kf_mean, kf_cov)
                    pred_cx, pred_cy, pred_a, pred_h_val = kf_mean[:4]
                    pred_w_val = max(1, abs(pred_a * pred_h_val))
                    pred_h_val = max(1, abs(pred_h_val))
                    pred_x = int(pred_cx - pred_w_val / 2)
                    pred_y = int(pred_cy - pred_h_val / 2)
                    pred_w = int(pred_w_val)
                    pred_h = int(pred_h_val)
                else:
                    # 无 KF：回退到最近有效框或中心小框
                    H, W = current_frame_bgr.shape[:2]
                    if last_good_bbox is not None:
                        bx, by, bw, bh = last_good_bbox
                        pred_x, pred_y, pred_w, pred_h = _clip_box_xywh_to_frame(bx, by, bw, bh, W, H)
                    else:
                        cw, ch = int(W * 0.2), int(H * 0.2)
                        pred_x = int(W // 2 - cw // 2)
                        pred_y = int(H // 2 - ch // 2)
                        pred_w, pred_h = int(cw), int(ch)

                # 预测阶段仅预测，不产生控制动作；首次进入时清零一次以刹停
                if not predict_hold_sent and vehicle is not None:
                    try:
                        vehicle.send_velocity(0, 0, 0, 0)
                    except Exception:
                        pass
                    predict_hold_sent = True

                # 并行找回：周期性入队预测框ROI
                if cfg.get('reacq_parallel', True) and reacq_worker is not None:
                    if (frame_idx % reacq_interval) == 0:
                        H, W = current_frame_bgr.shape[:2]
                        px, py, pw, ph = int(pred_x), int(pred_y), int(pred_w), int(pred_h)
                        px, py, pw, ph = _clip_box_xywh_to_frame(px, py, pw, ph, W, H)
                        if pw > 1 and ph > 1:
                            reacq_worker.enqueue(current_frame_bgr.copy(), (px, py, int(pw), int(ph)), frame_idx)

                    # 轮询找回结果
                    if reacq_worker.found_event.is_set():
                        with reacq_worker.result_lock:
                            result = copy.deepcopy(reacq_worker.best_result)
                        if result is not None and result.get('bbox') is not None:
                            detected_bbox = result['bbox']
                            print(f"[Reacquire] Found target with sim={result.get('similarity', 0):.3f} at {detected_bbox}")
                            try:
                                ortrack_tracker.initialize(current_frame_rgb, {'init_bbox': list(detected_bbox)})
                            except Exception as e:
                                print(f"Error re-initializing ORTrack with bbox {detected_bbox}: {e}")
                                if reacq_worker is not None:
                                    try:
                                        reacq_worker.stop()
                                    except Exception:
                                        pass
                                return "FAILED"
                            # 重新初始化KF
                            x0r = detected_bbox[0] + detected_bbox[2] / 2
                            y0r = detected_bbox[1] + detected_bbox[3] / 2
                            a0r = detected_bbox[2] / max(1, detected_bbox[3])
                            h0r = detected_bbox[3]
                            if str(cfg.get('kf_mode', 'confaware')).lower() != 'none':
                                kf_mean, kf_cov = kf.initiate(np.array([x0r, y0r, a0r, h0r]), init_velocity=[0, 0, 0, 0])
                            lost_frames_count = 0
                            in_predict_mode = False
                            predict_frames_count = 0
                            predict_frame_idx = 0
                            center_history.clear()
                            # 停止并清理找回线程
                            if reacq_worker is not None:
                                try:
                                    reacq_worker.stop()
                                except Exception:
                                    pass
                                reacq_worker = None
                            # 切回正常跟踪
                            frame_idx += 1
                            continue

                # 如果未启用并行找回，串行找回（使用统一 detection 模块）
                if not cfg.get('reacq_parallel', True):
                    ref_list = None
                    if ref_store is not None and ref_store.is_valid():
                        ref_list = ref_store.get_features()
                    elif isinstance(reference_feature, list) and len(reference_feature) > 0:
                        ref_list = reference_feature
                    if ref_list is None or len(ref_list) == 0:
                        predict_frames_count += 1
                        if predict_frames_count >= max_predict_frames:
                            print("预测阶段超时，未找回目标，任务重启")
                            return "REDETECT"
                    else:
                        detected, detected_bbox = find_instance_by_features(
                            current_frame_bgr, detector, feature_extractor, ref_list, cfg, stage=f"lost_track_frame_{frame_idx}"
                        )
                        if detected:
                            print("Instance re-acquired. Resuming tracking.")
                            try:
                                ortrack_tracker.initialize(current_frame_rgb, {'init_bbox': list(detected_bbox)})
                            except Exception as e:
                                print(f"Error re-initializing ORTrack with bbox {detected_bbox}: {e}")
                                return "FAILED"

                            x0i = detected_bbox[0] + detected_bbox[2] / 2
                            y0i = detected_bbox[1] + detected_bbox[3] / 2
                            a0i = detected_bbox[2] / max(1, detected_bbox[3])
                            h0i = detected_bbox[3]
                            if str(cfg.get('kf_mode', 'confaware')).lower() != 'none':
                                kf_mean, kf_cov = kf.initiate(np.array([x0i, y0i, a0i, h0i]), init_velocity=[0,0,0,0])
                            lost_frames_count = 0
                            in_predict_mode = False
                            predict_frames_count = 0
                            predict_frame_idx = 0
                            center_history.clear()
                            continue

                # 超时处理（并行或串行分支都要递增/检测）
                predict_frames_count += 1
                if predict_frames_count >= max_predict_frames:
                    if cfg.get('dp_enable', False):
                        print("预测阶段超时，尝试DP规划补救以打破遮挡...")
                    # 为防资源竞争，先停并行找回线程
                    if reacq_worker is not None:
                        try:
                            reacq_worker.stop()
                        except Exception:
                            pass
                        reacq_worker = None

                    if cfg.get('dp_enable', False) and vehicle is not None:
                        ok_dp, det_bbox = dp_rescue(
                            video, vehicle, detector, feature_extractor, ref_store, cfg,
                            pred_bbox_xywh=(
                                int(pred_x) if pred_x is not None else 0,
                                int(pred_y) if pred_y is not None else 0,
                                int(pred_w) if pred_w is not None else 1,
                                int(pred_h) if pred_h is not None else 1
                            ),
                            dp_cache=get_dp_cache(),
                            read_and_log_fn=read_and_log,
                            find_instance_by_features_fn=find_instance_by_features,
                            save_follow_stream_frame_fn=save_follow_stream_frame,
                            load_dp_policy_fn=_load_dp_policy_fn,
                            build_dp_obs_fn=build_dp_obs_from_frame,
                            dp_action_to_vel_fn=dp_action_to_tello_vel
                        )
                        if ok_dp and det_bbox is not None:
                            try:
                                ok2, reinit_bgr = read_and_log(video, cfg, stage="Detect")
                                reinit_rgb = cv2.cvtColor(reinit_bgr, cv2.COLOR_BGR2RGB) if ok2 else current_frame_rgb
                                ortrack_tracker.initialize(reinit_rgb, {'init_bbox': list(det_bbox)})
                                # 重新初始化KF
                                x0d = det_bbox[0] + det_bbox[2] / 2
                                y0d = det_bbox[1] + det_bbox[3] / 2
                                a0d = det_bbox[2] / max(1, det_bbox[3])
                                h0d = det_bbox[3]
                                if str(cfg.get('kf_mode', 'confaware')).lower() != 'none':
                                    kf_mean, kf_cov = kf.initiate(np.array([x0d, y0d, a0d, h0d]), init_velocity=[0, 0, 0, 0])
                                lost_frames_count = 0
                                in_predict_mode = False
                                predict_frames_count = 0
                                predict_frame_idx = 0
                                center_history.clear()
                                print("[DP] 找回成功，恢复正常跟踪。")
                                continue
                            except Exception as e:
                                print(f"[DP] 重新初始化跟踪失败: {e}")

                    print("预测阶段超时，未找回目标，任务重启")
                    return "REDETECT"

            # 可视化与保存
            if (cfg.get('plot_visualizations', False) or cfg.get('save_images_to')):
                if not in_predict_mode:
                    vis_frame = current_frame_bgr.copy()
                    color = (0, 0, 255) if is_lost else (0, 255, 0)
                    x_vis, y_vis, w_vis, h_vis = current_bbox_xywh
                    cv2.rectangle(vis_frame, (x_vis, y_vis), (x_vis + w_vis, y_vis + h_vis), color, 2)
                    status_text = f"Conf: {confidence:.2f}"
                    if is_lost: status_text += " (LOST)"
                    cv2.putText(vis_frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    plot_and_save_if_neded(cfg, vis_frame, "Tracker-ORTrack", frame_idx)
                    if cfg.get('plot_visualizations', False):
                        cv2.imshow("ORTrack Tracking", vis_frame)
                        if cv2.waitKey(cfg.get('wait_key', 1)) & 0xFF == 27:
                            if reacq_worker is not None:
                                try:
                                    reacq_worker.stop()
                                except Exception:
                                    pass
                            return "USER_QUIT"
                else:
                    vis_frame = current_frame_bgr.copy()
                    if pred_x is not None and pred_y is not None and pred_w is not None and pred_h is not None:
                        px, py, pw, ph = pred_x, pred_y, pred_w, pred_h
                        cv2.rectangle(vis_frame, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
                        cv2.putText(vis_frame, "KF-PREDICT", (px, max(0, py - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(vis_frame, f"Frame: {frame_idx} [Predict]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    plot_and_save_if_neded(cfg, vis_frame, "Tracker-ORTrack-Predict", frame_idx)
                    if cfg.get('plot_visualizations', False):
                        cv2.imshow("ORTrack Tracking", vis_frame)
                        if cv2.waitKey(cfg.get('wait_key', 1)) & 0xFF == 27:
                            if reacq_worker is not None:
                                try:
                                    reacq_worker.stop()
                                except Exception:
                                    pass
                            return "USER_QUIT"

            # FollowStream：保存跟踪阶段可视化帧
            try:
                if in_predict_mode:
                    pb = (int(pred_x), int(pred_y), int(pred_w), int(pred_h)) if (pred_x is not None and pred_y is not None and pred_w is not None and pred_h is not None) else None
                    save_follow_stream_frame(cfg, current_frame_bgr, stage="Predict", predict_bbox=pb, frame_idx=frame_idx)
                else:
                    if is_lost:
                        save_follow_stream_frame(cfg, current_frame_bgr, stage="Lost", is_lost=True, frame_idx=frame_idx)
                    else:
                        save_follow_stream_frame(cfg, current_frame_bgr, stage="Track", track_bbox=current_bbox_xywh, frame_idx=frame_idx)
            except Exception:
                pass

            # 写入视频（即使未开启图像保存也可写入）
            if cfg.get('video_writer') is not None:
                to_write = current_frame_bgr.copy()
                xw, yw, ww, hw = current_bbox_xywh
                cv2.rectangle(to_write, (xw, yw), (xw + ww, yw + hw), (0, 255, 0) if not is_lost else (0, 0, 255), 2)
                cv2.putText(to_write, f"Conf: {confidence:.2f}{' LOST' if is_lost else ''}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if not is_lost else (0,0,255), 2)
                if in_predict_mode and pred_x is not None and pred_y is not None and pred_w is not None and pred_h is not None:
                    px, py, pw, ph = pred_x, pred_y, pred_w, pred_h
                    cv2.rectangle(to_write, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
                    cv2.putText(to_write, "KF-PREDICT", (px, max(0, py - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cfg['video_writer'].write(to_write)

            # 打印详细帧日志（仅 eval_table1 且开启）
            if cfg.get('eval_table1', False) and cfg.get('eval_log_detail', False):
                try:
                    last = cfg.get('_last_cmd') or {}
                    vy_log  = float(last.get('vy', 0.0))
                    yaw_log = float(last.get('yaw', 0.0))
                    tags = []
                    if last.get('sat_yaw'): tags.append('SAT_YAW')
                    if last.get('sat_vy'):  tags.append('SAT_VY')
                    if last.get('back'):    tags.append('BACKWARD')
                    tag_str = ",".join(tags)
                    xw, yw, ww, hw = current_bbox_xywh
                    target_area = int(float(cfg.get('target_area', cfg.get('target_area_ratio', 0.12))) * (cfg['desired_width']*cfg['desired_height']))
                    print(
                        f"frame={frame_idx} conf={float(confidence):.3f} lost={int(is_lost)} "
                        f"area={int(ww*hw)} target={target_area} yaw={yaw_log:.1f} vy={vy_log:.1f} "
                        f"kf_mode={str(cfg.get('kf_mode','confaware'))} bbox=({xw},{yw},{ww},{hw}) tags={tag_str}"
                    )
                except Exception:
                    pass

            # FPS（即时与指数滑动平均）
            now = time.time()
            dt = max(1e-6, now - last_frame_time)
            last_frame_time = now
            fps_inst = 1.0 / dt
            fps_ema = fps_inst if fps_ema is None else (1.0 - fps_alpha) * fps_ema + fps_alpha * fps_inst
            if fps_log_interval <= 1 or (frame_idx % fps_log_interval == 0):
                try:
                    print(f"FPS: {fps_inst:.1f} (avg {fps_ema:.1f})")
                except Exception:
                    pass

            # 评测回调（控制 episode 结束）
            if step_cb is not None:
                try:
                    cont = step_cb(frame_idx, current_frame_bgr)
                except TypeError:
                    cont = step_cb(frame_idx)
                if cont is False:
                    break

            frame_idx += 1

    # 退出循环时清理找回线程
    if reacq_worker is not None:
        try:
            reacq_worker.stop()
        except Exception:
            pass

    return "COMPLETED"

__all__ = ['track_object_with_ortrack']