"""在线增强与丢失找回工作线程"""
import threading
import queue
import cv2
import numpy as np
import torch
import torch.nn.functional as F


# 全局YOLO锁（避免并发调用）
YOLO_LOCK = threading.Lock()


class OnlineEnhancerWorker:
    """在线参考特征增强工作线程（后台异步执行）"""
    
    def __init__(self, detector, feature_extractor, ref_store, cfg):
        """初始化增强工作线程
        
        Args:
            detector: YOLO检测器
            feature_extractor: DINOv3特征提取器
            ref_store: ReferenceFeatureStore实例
            cfg: 配置字典
        """
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.ref_store = ref_store
        self.cfg = cfg
        self.queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="OnlineEnhancerWorker", daemon=True)

    def start(self):
        """启动工作线程"""
        self._thread.start()

    def stop(self):
        """停止工作线程"""
        self._stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def enqueue(self, frame_bgr, bbox_xywh, frame_idx):
        """入队一帧用于增强（非阻塞）
        
        Args:
            frame_bgr: BGR图像
            bbox_xywh: 跟踪框 (x, y, w, h)
            frame_idx: 帧索引
        """
        if self._stop_event.is_set():
            return
        try:
            if self.queue.full():
                return
            self.queue.put_nowait((frame_bgr, tuple(bbox_xywh), int(frame_idx)))
        except Exception:
            pass

    @staticmethod
    def _clip_box(x1, y1, x2, y2, W, H):
        """裁剪框到图像范围内"""
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _iou_xywh(a, b):
        """计算IoU（xywh格式）"""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _run(self):
        """后台线程主循环"""
        # 延迟导入避免循环依赖
        from detectors import perform_yolo_detection_for_candidates
        
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                continue
            frame_bgr, track_bbox_xywh, frame_idx = item
            try:
                self._process_one(frame_bgr, track_bbox_xywh, frame_idx, perform_yolo_detection_for_candidates)
            except Exception as e:
                print(f"[Enhancer] Error processing frame {frame_idx}: {e}")

    def _process_one(self, frame_bgr, track_bbox_xywh, frame_idx, yolo_func):
        """处理单帧增强
        
        Args:
            frame_bgr: BGR图像
            track_bbox_xywh: 跟踪框
            frame_idx: 帧索引
            yolo_func: YOLO检测函数
        """
        if self.ref_store is None or not self.ref_store.is_valid():
            return
        H, W = frame_bgr.shape[:2]
        x, y, w, h = track_bbox_xywh
        if w <= 0 or h <= 0:
            return
        
        # 生成ROI
        expand = float(self.cfg.get('enhance_crop_expand', 0.2) or 0.0)
        if expand <= 0.0:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, W, H
        else:
            dx = int(w * expand)
            dy = int(h * expand)
            roi_x1 = max(0, x - dx)
            roi_y1 = max(0, y - dy)
            roi_x2 = min(W, x + w + dx)
            roi_y2 = min(H, y + h + dy)
        if roi_x2 - roi_x1 < 2 or roi_y2 - roi_y1 < 2:
            return
        roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        # YOLO on ROI
        with YOLO_LOCK:
            cand_bboxes_roi, cand_masks_roi = yolo_func(
                roi,
                self.detector,
                confidence_threshold=self.cfg['yolo_confidence_thresh']
            )
        
        # Map ROI candidates to full image
        cand_bboxes = []
        if cand_bboxes_roi:
            for (rx, ry, rw, rh) in cand_bboxes_roi:
                cand_bboxes.append((rx + roi_x1, ry + roi_y1, rw, rh))
        cand_masks = None
        if cand_masks_roi is not None:
            cand_masks = []
            for m in cand_masks_roi:
                # place ROI mask back to full-res canvas
                m = m.astype(np.uint8)
                full = np.zeros((H, W), dtype=np.uint8)
                mh, mw = m.shape[:2]
                if mh != (roi_y2 - roi_y1) or mw != (roi_x2 - roi_x1):
                    m = cv2.resize(m, (roi_x2 - roi_x1, roi_y2 - roi_y1), interpolation=cv2.INTER_NEAREST)
                full[roi_y1:roi_y2, roi_x1:roi_x2] = m
                cand_masks.append(full)

        # Choose the candidate that matches the ORTrack bbox by IoU
        best_idx = -1
        best_iou = -1.0
        for i, b in enumerate(cand_bboxes):
            iou = self._iou_xywh(track_bbox_xywh, b)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        use_mask = None
        if best_idx >= 0 and best_iou >= float(self.cfg.get('enhance_iou_thresh', 0.3)):
            bx, by, bw, bh = cand_bboxes[best_idx]
            if cand_masks is not None and best_idx < len(cand_masks):
                use_mask = cand_masks[best_idx]
        else:
            bx, by, bw, bh = track_bbox_xywh  # fallback to tracker bbox

        # Extract masked feature for the chosen bbox
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, W, H)
        masked_img = frame_bgr.copy()
        if use_mask is not None:
            m = use_mask
            if m.shape[0] != H or m.shape[1] != W:
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            m_bool = m.astype(bool) if m.dtype != np.bool_ else m
            masked_img[~m_bool] = 0
            ys, xs = np.where(m_bool)
            if len(xs) and len(ys):
                tx1, ty1, tx2, ty2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                # intersect with original box
                tx1 = max(tx1, x1)
                ty1 = max(ty1, y1)
                tx2 = min(tx2, x2)
                ty2 = min(ty2, y2)
                if tx2 > tx1 and ty2 > ty1:
                    x1, y1, x2, y2 = tx1, ty1, tx2, ty2
        feat = self.feature_extractor.extract_features(masked_img, np.array([[x1, y1, x2, y2]]))
        f_cur = F.normalize(feat[0], dim=0)
        self.ref_store.update_with_feature(f_cur, alpha=float(self.cfg.get('enhance_alpha', 0.2)))


class LostReacquireWorker:
    """丢失阶段并行找回工作线程"""
    
    def __init__(self, detector, feature_extractor, ref_store, static_ref_list, cfg):
        """初始化找回工作线程
        
        Args:
            detector: YOLO检测器
            feature_extractor: DINOv3特征提取器
            ref_store: ReferenceFeatureStore实例
            static_ref_list: 静态参考特征列表（备选）
            cfg: 配置字典
        """
        self.detector = detector
        self.feature_extractor = feature_extractor
        self.ref_store = ref_store
        self.static_ref_list = static_ref_list
        self.cfg = cfg
        self.queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="LostReacquireWorker", daemon=True)
        self.found_event = threading.Event()
        self.result_lock = threading.Lock()
        self.best_result = None  # dict: {bbox, similarity, frame_idx}
        self._roi_fail_count = 0

    def start(self):
        """启动工作线程"""
        self._thread.start()

    def stop(self):
        """停止工作线程"""
        self._stop_event.set()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def reset(self):
        """重置找回状态"""
        with self.result_lock:
            self.best_result = None
            self.found_event.clear()
        self._roi_fail_count = 0

    def enqueue(self, frame_bgr, pred_bbox_xywh, frame_idx):
        """入队一帧用于找回
        
        Args:
            frame_bgr: BGR图像
            pred_bbox_xywh: 预测框
            frame_idx: 帧索引
        """
        if self._stop_event.is_set():
            return
        try:
            # 仅保留最新帧，丢弃旧的
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Exception:
                    break
            self.queue.put_nowait((frame_bgr, tuple(pred_bbox_xywh), int(frame_idx)))
        except Exception:
            pass

    @staticmethod
    def _clip_box(x1, y1, x2, y2, W, H):
        """裁剪框到图像范围内"""
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))
        if x2 <= x1:
            x2 = min(W - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(H - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _iou_xywh(a, b):
        """计算IoU（xywh格式）"""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh
        inter_x1 = max(ax, bx)
        inter_y1 = max(ay, by)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    def _get_reference_list(self):
        """获取参考特征列表（优先使用在线增强的）"""
        if self.ref_store is not None and self.ref_store.is_valid():
            return self.ref_store.get_features()
        if isinstance(self.static_ref_list, list) and len(self.static_ref_list) > 0:
            return self.static_ref_list
        return None

    def _run(self):
        """后台线程主循环"""
        # 延迟导入避免循环依赖
        from detectors import perform_yolo_detection_for_candidates
        
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.3)
            except queue.Empty:
                continue
            if item is None:
                continue
            frame_bgr, pred_bbox_xywh, frame_idx = item
            try:
                self._process_one(frame_bgr, pred_bbox_xywh, frame_idx, perform_yolo_detection_for_candidates)
            except Exception as e:
                print(f"[Reacquire] Error processing frame {frame_idx}: {e}")

    def _process_one(self, frame_bgr, pred_bbox_xywh, frame_idx, yolo_func):
        """处理单帧找回
        
        Args:
            frame_bgr: BGR图像
            pred_bbox_xywh: 预测框
            frame_idx: 帧索引
            yolo_func: YOLO检测函数
        """
        ref_list = self._get_reference_list()
        if ref_list is None or len(ref_list) == 0:
            return
        H, W = frame_bgr.shape[:2]
        x, y, w, h = pred_bbox_xywh
        
        # 生成 ROI
        expand = float(self.cfg.get('reacq_crop_expand', 0.4) or 0.0)
        use_full_frame = False
        if expand <= 0.0:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, W, H
        else:
            dx = int(w * expand)
            dy = int(h * expand)
            roi_x1 = max(0, x - dx)
            roi_y1 = max(0, y - dy)
            roi_x2 = min(W, x + w + dx)
            roi_y2 = min(H, y + h + dy)
            if roi_x2 - roi_x1 < 2 or roi_y2 - roi_y1 < 2:
                use_full_frame = True

        if self._roi_fail_count >= int(self.cfg.get('reacq_fullframe_after_n', 3)):
            use_full_frame = True

        if use_full_frame:
            roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, W, H
        roi = frame_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

        # YOLO on ROI / full frame
        with YOLO_LOCK:
            cand_bboxes_roi, cand_masks_roi = yolo_func(
                roi,
                self.detector,
                confidence_threshold=self.cfg['yolo_confidence_thresh']
            )
        
        # Map ROI candidates to full image coords
        cand_bboxes = []
        if cand_bboxes_roi:
            for (rx, ry, rw, rh) in cand_bboxes_roi:
                cand_bboxes.append((rx + roi_x1, ry + roi_y1, rw, rh))
        cand_masks = None
        if cand_masks_roi is not None:
            cand_masks = []
            for m in cand_masks_roi:
                m = m.astype(np.uint8)
                full = np.zeros((H, W), dtype=np.uint8)
                mh, mw = m.shape[:2]
                if mh != (roi_y2 - roi_y1) or mw != (roi_x2 - roi_x1):
                    m = cv2.resize(m, (roi_x2 - roi_x1, roi_y2 - roi_y1), interpolation=cv2.INTER_NEAREST)
                full[roi_y1:roi_y2, roi_x1:roi_x2] = m
                cand_masks.append(full)

        if not cand_bboxes:
            self._roi_fail_count += 1
            return

        # 按与预测框 IoU 做初筛
        filtered = []
        for i, b in enumerate(cand_bboxes):
            iou = self._iou_xywh(pred_bbox_xywh, b)
            if iou >= float(self.cfg.get('reacq_iou_with_pred', 0.2)):
                filtered.append((i, b))
        if not filtered:
            # 全部不满足 IoU，仍继续用全部候选做匹配
            filtered = list(enumerate(cand_bboxes))

        # 提取候选特征（掩码紧框 + 求交 + 回退）
        features = []
        keep_indices = []
        for i, (xw, yw, ww, hh) in filtered:
            x1, y1, x2, y2 = int(xw), int(yw), int(xw + ww), int(yw + hh)
            x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, W, H)
            masked_img = frame_bgr.copy()
            if cand_masks is not None and i < len(cand_masks) and cand_masks[i] is not None:
                m = cand_masks[i]
                if m.shape[0] != H or m.shape[1] != W:
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                m_bool = m.astype(bool) if m.dtype != np.bool_ else m
                masked_img[~m_bool] = 0
                ys, xs = np.where(m_bool)
                if len(xs) and len(ys):
                    tx1, ty1, tx2, ty2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
                    tx1 = max(tx1, x1)
                    ty1 = max(ty1, y1)
                    tx2 = min(tx2, x2)
                    ty2 = min(ty2, y2)
                    if tx2 > tx1 and ty2 > ty1:
                        x1, y1, x2, y2 = tx1, ty1, tx2, ty2
            feat = self.feature_extractor.extract_features(masked_img, np.array([[x1, y1, x2, y2]]))
            features.append(feat[0])
            keep_indices.append(i)
        if len(features) == 0:
            self._roi_fail_count += 1
            return
        cand_features = torch.stack(features, dim=0)

        # 匹配
        ref_feats_tensor = torch.stack(ref_list).to(cand_features.device)
        sim_mat = F.cosine_similarity(cand_features.unsqueeze(1), ref_feats_tensor.unsqueeze(0), dim=2)
        best_pair_sims, _ = sim_mat.max(dim=1)
        best_idx_local = int(best_pair_sims.argmax().item())
        best_sim = float(best_pair_sims[best_idx_local].item())
        best_cand_global_idx = keep_indices[best_idx_local]
        best_bbox_xywh = cand_bboxes[best_cand_global_idx]

        if best_sim >= float(self.cfg.get('reacq_match_thresh', 0.70)):
            with self.result_lock:
                self.best_result = {
                    'bbox': best_bbox_xywh,
                    'similarity': best_sim,
                    'frame_idx': frame_idx
                }
                self.found_event.set()
        else:
            self._roi_fail_count += 1


