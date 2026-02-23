"""可视化相关工具函数"""
import os
import cv2
import numpy as np


# 全局计数器
raw_frame_counter = 0
follow_vis_counter = 0


def plot_and_save_if_neded(cfg, image_to_plot, stage_and_task, count, multiply=1):
    # 不再依赖全局 mission_counter，避免 NameError
    mission_id = int(cfg.get('mission_counter', 0))

    if cfg.get('plot_visualizations', False):
        cv2.imshow(stage_and_task, image_to_plot)
        cv2.waitKey(cfg.get('wait_key', 1))

    save_dir_base = cfg.get('save_images_to')
    if save_dir_base:
        save_dir = os.path.join(save_dir_base, stage_and_task)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        file_name = os.path.join(save_dir, f"{mission_id}_{count}.jpg")
        cv2.imwrite(file_name, image_to_plot * multiply)


def save_raw_frame(frame, save_images_to=None):
    """保存原始视频帧
    
    Args:
        frame: BGR图像
        save_images_to: 保存根目录（可选）
    """
    global raw_frame_counter
    if frame is None or (hasattr(frame, 'size') and frame.size == 0):
        return
    
    if not save_images_to:
        return
    
    raw_stream_dir = os.path.join(save_images_to, "RawStream")
    if not os.path.exists(raw_stream_dir):
        os.makedirs(raw_stream_dir, exist_ok=True)
    raw_img_path = os.path.join(raw_stream_dir, f"raw_{raw_frame_counter:06d}.jpg")
    cv2.imwrite(raw_img_path, frame)
    raw_frame_counter += 1


def save_follow_stream_frame(cfg, frame_bgr, stage,
                             track_bbox=None,
                             predict_bbox=None,
                             det_bboxes=None,
                             det_best_idx=None,
                             is_lost=False,
                             frame_idx=None):
    """保存跟踪流程的可视化帧
    
    Args:
        cfg: 配置字典
        frame_bgr: BGR图像
        stage: 阶段名称
        track_bbox: 跟踪框 (x, y, w, h)
        predict_bbox: 预测框 (x, y, w, h)
        det_bboxes: 检测框列表 [(x, y, w, h), ...]
        det_best_idx: 最佳检测框索引
        is_lost: 是否丢失
        frame_idx: 帧索引（可选）
    """
    base_dir = cfg.get('save_images_to')
    if not base_dir:
        return
    
    try:
        save_dir = os.path.join(base_dir, "FollowStream")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        vis = frame_bgr.copy()
        
        # 绘制不同阶段内容
        if det_bboxes is not None:
            for i, (x, y, w, h) in enumerate(det_bboxes):
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                # 检测阶段候选框：统一使用蓝色；若指定最佳索引，可保持高亮颜色
                color = (0, 255, 255) if (det_best_idx is not None and i == int(det_best_idx)) else (255, 0, 0)
                thickness = 3 if (det_best_idx is not None and i == int(det_best_idx)) else 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        elif predict_bbox is not None:
            px, py, pw, ph = [int(v) for v in predict_bbox]
            cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 255, 255), 2)
        elif (track_bbox is not None) and (not is_lost):
            x, y, w, h = [int(v) for v in track_bbox]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # LOST 文本（不画框）
        if is_lost:
            cv2.putText(vis, "LOST", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
        
        global follow_vis_counter
        out_path = os.path.join(save_dir, f"follow_{follow_vis_counter:06d}_{stage}.jpg")
        cv2.imwrite(out_path, vis)
        follow_vis_counter += 1
    except Exception:
        pass


def save_detection_debug_image(cfg, frame_bgr, candidate_bboxes_xywh, candidate_masks,
                               similarities=None, best_match_idx=None, match_found=False,
                               stage="Detection_YOLO_Debug", count=0):
    """保存检测调试图像（带掩码和相似度可视化）
    
    Args:
        cfg: 配置字典
        frame_bgr: BGR图像
        candidate_bboxes_xywh: 候选框列表 [(x, y, w, h), ...]
        candidate_masks: 候选掩码列表（可选）
        similarities: 相似度列表（可选）
        best_match_idx: 最佳匹配索引（可选）
        match_found: 是否找到匹配
        stage: 阶段名称
        count: 计数器
    """
    vis_image = frame_bgr.copy()
    num_cands = len(candidate_bboxes_xywh) if candidate_bboxes_xywh is not None else 0

    for i, (x, y, w, h) in enumerate(candidate_bboxes_xywh or []):
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        is_best = (match_found and best_match_idx is not None and i == int(best_match_idx))
        color = (0, 0, 255) if is_best else (0, 255, 0)
        thickness = 3 if is_best else 2

        # 掩码半透明叠加
        if candidate_masks is not None and i < len(candidate_masks) and candidate_masks[i] is not None:
            m = candidate_masks[i]
            if m.shape[0] != vis_image.shape[0] or m.shape[1] != vis_image.shape[1]:
                m = cv2.resize(m, (vis_image.shape[1], vis_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            if m.dtype != np.bool_:
                m = m.astype(bool)
            alpha = 0.35
            vis_image[m] = (
                vis_image[m].astype(np.float32) * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)

        # 绘制框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)

        # 候选编号圆圈
        circle_x = max(x1 + 20, 20)
        circle_y = max(y1 + 20, 20)
        circle_radius = 18
        cv2.circle(vis_image, (circle_x, circle_y), circle_radius, color, -1)
        cv2.circle(vis_image, (circle_x, circle_y), circle_radius, (0, 0, 0), 2)
        text_x = circle_x - 8 if i < 10 else circle_x - 12
        text_y = circle_y + 6
        cv2.putText(vis_image, str(i), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 标签（相似度）
        label = None
        if similarities is not None and i < len(similarities):
            try:
                sim_val = float(similarities[i])
            except Exception:
                sim_val = similarities[i].item() if hasattr(similarities[i], 'item') else similarities[i]
            label = f"sim: {sim_val:.3f}"
        if label is not None:
            label_y = min(y2 + 25, vis_image.shape[0] - 5)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(vis_image, (x1, label_y - 20), (x1 + label_size[0] + 10, label_y + 5), color, -1)
            cv2.putText(vis_image, label, (x1 + 5, label_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # 信息栏
    info_lines = [
        f"Candidates: {num_cands}",
        f"Match Found: {bool(match_found)}"
    ]
    if match_found and similarities is not None and best_match_idx is not None:
        try:
            best_sim = float(similarities[int(best_match_idx)])
        except Exception:
            best_sim = similarities[int(best_match_idx)].item() if hasattr(similarities[int(best_match_idx)], 'item') else similarities[int(best_match_idx)]
        info_lines[1] += f" (Similarity: {best_sim:.3f})"

    cv2.rectangle(vis_image, (5, 5), (700, 60), (0, 0, 0), -1)
    for k, line in enumerate(info_lines):
        y_pos = 25 + k * 20
        cv2.putText(vis_image, line, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    plot_and_save_if_neded(cfg, vis_image, stage, count)


