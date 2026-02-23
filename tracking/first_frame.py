"""首帧检测与目标初始化"""
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def detect_object_yolo(cfg, detector, video, feature_extractor, read_and_log_fn, 
                      perform_yolo_detection_for_candidates_fn,
                      save_follow_stream_frame_fn, save_detection_debug_image_fn,
                      plot_and_save_if_neded_fn, yolo_lock):
    """首帧检测：使用YOLO检测目标并可选地通过特征匹配筛选
    
    Args:
        cfg: 配置字典
        detector: YOLO检测器实例
        video: 视频源
        feature_extractor: DINOv3特征提取器
        read_and_log_fn: 读帧函数
        perform_yolo_detection_for_candidates_fn: YOLO批量检测函数
        save_follow_stream_frame_fn: 保存跟踪流帧函数
        save_detection_debug_image_fn: 保存检测调试图函数
        plot_and_save_if_neded_fn: 绘图保存函数
        yolo_lock: YOLO锁（线程安全）
        
    Returns:
        tuple: (bounding_boxes, None, saved_frame, reference_feature)
            - bounding_boxes: 检测到的bbox列表
            - saved_frame: 最后保存的帧
            - reference_feature: 参考特征（如果提供）
    """
    print("Detecting with YOLO...")
    bounding_boxes = []
    saved_frame = None

    detection_attempts = 0
    max_attempts = cfg.get('max_detection_attempts', 50)

    while True:
        read_one_frame = False
        while not read_one_frame:
            read_one_frame, frame = read_and_log_fn(video, cfg, stage="Detect")

        # 使用原始尺寸帧进行检测与匹配
        frame_resized = frame
        saved_frame = frame_resized.copy()

        # 懒初始化视频写入器
        if cfg.get('save_video_to') and cfg.get('video_writer') is None:
            try:
                out_dir = os.path.dirname(cfg['save_video_to'])
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                h, w = saved_frame.shape[:2]
                fps = cfg['video_writer_fps'] if cfg.get('video_writer_fps') else (
                    video.fps if hasattr(video, 'fps') else 25.0)
                fourcc = cv2.VideoWriter_fourcc(*(
                    'mp4v' if cfg['save_video_to'].lower().endswith('.mp4') else 'XVID'))
                vw = cv2.VideoWriter(cfg['save_video_to'], fourcc, fps, (w, h))
                if vw is not None and vw.isOpened():
                    cfg['video_writer'] = vw
                    cfg['video_writer_size'] = (w, h)
                    cfg['video_writer_fps'] = fps
                    print(f"VideoWriter initialized (lazy): {cfg['save_video_to']} at {(w,h)} @ {fps}fps")
                else:
                    print(f"Failed to open VideoWriter at {cfg['save_video_to']}")
            except Exception as e:
                print(f"Lazy VideoWriter init error: {e}")

        detection_loop_start = time.time()
        
        yolo_start_time = time.time()
        with yolo_lock:
            candidate_bboxes, candidate_masks = perform_yolo_detection_for_candidates_fn(
                frame_resized,
                detector,
                confidence_threshold=cfg['yolo_confidence_thresh']
            )
        yolo_detection_time = time.time() - yolo_start_time
        print(f"⏱️ YOLO检测耗时: {yolo_detection_time:.4f}s")
        
        # FollowStream：保存检测阶段帧（带候选框，若无候选则无框）
        try:
            save_follow_stream_frame_fn(cfg, saved_frame, stage="Detect", 
                                       det_bboxes=candidate_bboxes, frame_idx=detection_attempts)
        except Exception:
            pass

        if candidate_bboxes:
            print(f"YOLO found {len(candidate_bboxes)} candidate(s).")

            # 保存候选调试图
            try:
                save_detection_debug_image_fn(cfg, saved_frame, candidate_bboxes, candidate_masks,
                                            similarities=None, best_match_idx=None, match_found=False,
                                            stage="Detection_YOLO_Candidates", count=detection_attempts)
            except Exception:
                pass

            if cfg.get('reference_feature_path') and feature_extractor:
                print("Reference feature provided. Finding the best match among candidates...")

                ref_load_start = time.time()
                try:
                    reference_feature = torch.load(cfg['reference_feature_path'])
                    if not isinstance(reference_feature, list):
                        reference_feature = [reference_feature.squeeze(0)]
                    ref_load_time = time.time() - ref_load_start
                    print(f"⏱️ 参考特征加载耗时: {ref_load_time:.4f}s")

                    # 调用 detection.py 中的统一特征匹配函数（消除重复代码）
                    from .detection import match_candidates_with_reference
                    
                    instance_match_start = time.time()
                    
                    # 使用统一的特征匹配函数（复用 detection.py 的逻辑）
                    best_match_idx, max_similarity, best_pair_similarities, best_bbox = match_candidates_with_reference(
                        candidate_bboxes, candidate_masks, saved_frame,
                        feature_extractor, reference_feature, cfg
                    )
                    
                    instance_match_time = time.time() - instance_match_start
                    
                    # 打印每个候选的相似度
                    for i, score in enumerate(best_pair_similarities):
                        print(f"  - Candidate {i} at {candidate_bboxes[i]}: Similarity = {float(score):.4f}")
                    
                    print(f"⏱️ 实例匹配总耗时: {instance_match_time:.4f}s")

                    # 保存匹配调试图
                    try:
                        save_detection_debug_image_fn(cfg, saved_frame, candidate_bboxes, candidate_masks,
                                                     similarities=best_pair_similarities.detach().cpu().numpy(),
                                                     best_match_idx=best_match_idx,
                                                     match_found=bool(max_similarity > cfg['match_similarity_thresh']),
                                                     stage="Detection_YOLO_MatchDebug", count=detection_attempts)
                    except Exception:
                        pass

                    if max_similarity > cfg['match_similarity_thresh']:
                        print(f"Instance selection successful. Best match at {best_bbox}")
                        bounding_boxes.append(best_bbox)
                        if cfg.get('plot_visualizations') or cfg.get('save_images_to') or cfg.get('video_writer') is not None:
                            vis_frame = saved_frame.copy()
                            x, y, w, h = best_bbox
                            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            plot_and_save_if_neded_fn(cfg, vis_frame, "Detection_YOLO_BestMatch", detection_attempts)
                            if cfg.get('video_writer') is not None:
                                cfg['video_writer'].write(vis_frame)
                        total_detection_time = time.time() - detection_loop_start
                        print(f"⏱️ 完整检测流程耗时: {total_detection_time:.4f}s")
                    else:
                        print("Could not find a matching instance in the first frame with candidates.")
                        if cfg.get('video_writer') is not None:
                            # 画出所有候选以便回看
                            dbg = saved_frame.copy()
                            for (x, y, w, h) in candidate_bboxes:
                                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 1)
                            cfg['video_writer'].write(dbg)
                        detection_attempts += 1
                        # 达到最大尝试次数：首帧匹配失败，退出检测环节
                        if detection_attempts >= max_attempts:
                            print(f"YOLO/feature matching failed after {max_attempts} attempts.")
                            break
                        continue

                except Exception as e:
                    print(f"Error during initial instance selection: {e}")
                    detection_attempts += 1
                    continue
            else:
                print("No reference feature. Selecting the first detected object.")
                bbox = candidate_bboxes[0]
                bounding_boxes.append(list(bbox))
                if cfg.get('plot_visualizations') or cfg.get('save_images_to'):
                    vis_frame = saved_frame.copy()
                    x, y, w, h = bbox
                    cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    plot_and_save_if_neded_fn(cfg, vis_frame, "Detection_YOLO_Default", detection_attempts)

        if bounding_boxes:
            break

        detection_attempts += 1
        loop_time = time.time() - detection_loop_start
        print(f"⏱️ 检测尝试 {detection_attempts} 总耗时: {loop_time:.4f}s")
        
        if detection_attempts >= max_attempts:
            print(f"YOLO failed to detect any target after {max_attempts} attempts.")
            break

    if not bounding_boxes:
        print(f"YOLO failed to detect any valid target.")
        return [], None, saved_frame, None

    # 加载参考特征（用于后续跟踪）
    reference_feature = None
    if cfg.get('reference_feature_path'):
        try:
            reference_feature = torch.load(cfg['reference_feature_path'])
            if not isinstance(reference_feature, list):
                reference_feature = [reference_feature.squeeze(0)]
        except Exception:
            reference_feature = None

    return bounding_boxes, None, saved_frame, reference_feature



