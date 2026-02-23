"""YOLO检测与特征匹配功能"""
import threading
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# 全局YOLO锁（避免并发调用）
YOLO_LOCK = threading.Lock()


def match_candidates_with_reference(candidate_bboxes_xywh, candidate_masks, frame_bgr, 
                                    feature_extractor, reference_feature_set, cfg):
    """核心特征匹配逻辑：给定候选框，提取特征并与参考特征匹配
    
    Args:
        candidate_bboxes_xywh: 候选框列表 [(x, y, w, h), ...]
        candidate_masks: 候选掩码列表（可选）
        frame_bgr: BGR图像
        feature_extractor: DINOv3特征提取器
        reference_feature_set: 参考特征列表
        cfg: 配置字典
        
    Returns:
        tuple: (best_match_idx, max_similarity, best_pair_similarities, best_match_bbox)
            - best_match_idx: 最佳匹配的候选索引
            - max_similarity: 最大相似度
            - best_pair_similarities: 所有候选的最佳配对相似度
            - best_match_bbox: 最佳匹配的边界框
    """
    # 特征提取（掩码紧框 + 求交 + 回退）
    candidate_bboxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in candidate_bboxes_xywh]
    
    if candidate_masks is not None:
        features_list = []
        for xyxy, m in zip(candidate_bboxes_xyxy, candidate_masks):
            x1, y1, x2, y2 = map(int, xyxy)
            if m is None:
                feat = feature_extractor.extract_features(frame_bgr, np.array([[x1, y1, x2, y2]]))
                features_list.append(feat[0])
                continue
            masked_img = frame_bgr.copy()
            if m.shape[0] != masked_img.shape[0] or m.shape[1] != masked_img.shape[1]:
                m = cv2.resize(m, (masked_img.shape[1], masked_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            m_bool = m.astype(bool) if m.dtype != np.bool_ else m
            masked_img[~m_bool] = 0
            ys, xs = np.where(m_bool)
            if len(xs) and len(ys):
                tx1, ty1, tx2, ty2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                tx1, ty1 = max(tx1, x1), max(ty1, y1)
                tx2, ty2 = min(tx2, x2), min(tx2, y2)
                if tx2 > tx1 and ty2 > ty1:
                    x1, y1, x2, y2 = tx1, ty1, tx2, ty2
            feat = feature_extractor.extract_features(masked_img, np.array([[x1, y1, x2, y2]]))
            features_list.append(feat[0])
        candidate_features = torch.stack(features_list, dim=0)
    else:
        candidate_features = feature_extractor.extract_features(frame_bgr, np.array(candidate_bboxes_xyxy))
    
    # 匹配
    ref_features_tensor = torch.stack(reference_feature_set).to(candidate_features.device)
    similarity_matrix = F.cosine_similarity(candidate_features.unsqueeze(1), ref_features_tensor.unsqueeze(0), dim=2)
    best_pair_similarities, _ = similarity_matrix.max(dim=1)
    
    best_match_idx = best_pair_similarities.argmax()
    max_similarity = best_pair_similarities[best_match_idx]
    best_match_bbox = candidate_bboxes_xywh[best_match_idx]
    
    return int(best_match_idx), float(max_similarity), best_pair_similarities, best_match_bbox


def find_instance_by_features(frame_bgr, detector, feature_extractor, reference_feature_set, cfg, stage="detection"):
    """使用YOLO检测候选 + DINOv3特征匹配找到实例
    
    Args:
        frame_bgr: BGR图像
        detector: YOLO检测器
        feature_extractor: DINOv3特征提取器
        reference_feature_set: 参考特征列表
        cfg: 配置字典
        stage: 阶段名称（用于日志）
        
    Returns:
        (found, bbox): 是否找到，找到的bbox (x, y, w, h)
    """
    # 延迟导入避免循环依赖
    from detectors import perform_yolo_detection_for_candidates
    from .visualization import save_detection_debug_image
    
    stage_start_time = time.time()

    if not isinstance(reference_feature_set, list) or not reference_feature_set:
        print(f"[{stage}] Instance finding skipped: Invalid or empty reference feature set provided.")
        return False, None

    print(f"[{stage}] Attempting to find instance with a feature set of size {len(reference_feature_set)}...")

    is_cuda = torch.cuda.is_available()

    # 1. YOLO 候选（含掩码）
    if is_cuda: 
        torch.cuda.synchronize()
    yolo_start_time = time.time()
    with YOLO_LOCK:
        candidate_bboxes_xywh, candidate_masks = perform_yolo_detection_for_candidates(
            frame_bgr,
            detector,
            confidence_threshold=cfg['yolo_confidence_thresh']
        )
    if is_cuda: 
        torch.cuda.synchronize()
    yolo_time = time.time() - yolo_start_time
    print(f"[{stage}] ⏱️ YOLO候选框检测耗时: {yolo_time:.4f}s")

    if not candidate_bboxes_xywh:
        print(f"[{stage}] Instance finding failed: No candidate objects found by YOLO.")
        return False, None

    # 2. 特征提取与匹配（调用统一函数）
    if is_cuda: 
        torch.cuda.synchronize()
    feature_start_time = time.time()
    
    best_match_idx, max_similarity, best_pair_similarities, best_match_bbox = match_candidates_with_reference(
        candidate_bboxes_xywh, candidate_masks, frame_bgr,
        feature_extractor, reference_feature_set, cfg
    )
    
    if is_cuda: 
        torch.cuda.synchronize()
    feature_time = time.time() - feature_start_time
    print(f"[{stage}] ⏱️ 特征提取与匹配耗时: {feature_time:.4f}s ({len(candidate_bboxes_xywh)} 个候选)")

    # 打印每个候选的相似度
    for i, score in enumerate(best_pair_similarities):
        print(f"  - Candidate {i} at {candidate_bboxes_xywh[i]}: Similarity = {score:.4f}")

    print(f"[{stage}] Best match is Candidate {best_match_idx} with Best Pair Similarity: {max_similarity:.4f} (Threshold: {cfg['match_similarity_thresh']})")

    # 可视化保存
    try:
        save_detection_debug_image(cfg, frame_bgr, candidate_bboxes_xywh, candidate_masks,
                                   similarities=best_pair_similarities.detach().cpu().numpy(),
                                   best_match_idx=int(best_match_idx),
                                   match_found=bool(max_similarity > cfg['match_similarity_thresh']),
                                   stage=f"InstanceMatch_{stage}", count=int(time.time()))
    except Exception:
        pass

    if max_similarity > cfg['match_similarity_thresh']:
        print(f"[{stage}] Instance found successfully at {best_match_bbox}")
        return True, best_match_bbox
    else:
        print(f"[{stage}] Instance finding failed: No candidate passed the similarity threshold.")
        return False, None


