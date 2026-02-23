"""YOLO 检测器封装模块"""
import cv2
import numpy as np
from ultralytics import YOLO


class YOLODetector:
    """YOLO检测器类，封装所有YOLO相关功能"""
    
    def __init__(self, model_path, target_classes=None):
        """
        初始化YOLO检测器
        
        Args:
            model_path: YOLO模型权重路径
            target_classes: 目标类别列表（可选），如 ["person", "car"]
        """
        self.model = YOLO(model_path)
        self.selected_class_ids = None
        self._setup_classes(target_classes)
    
    def _setup_classes(self, target_classes):
        """设置目标类别（先尝试YOLOE文本提示，失败则回退到类别名/ID过滤）"""
        if not target_classes:
            print("===== YOLO: Tracking all classes =====")
            return

        print(f"===== YOLO: Setting target classes/prompts: {target_classes} =====")

        # 先尝试 YOLOE 文本提示（带超时，避免阻塞）
        try:
            import signal

            def _timeout_handler(signum, frame):
                raise TimeoutError("YOLO text encoder download timeout")

            try:
                signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(30)  # 30秒超时
                try:
                    text_pe = self.model.get_text_pe(target_classes)
                    self.model.set_classes(target_classes, text_pe)
                    print("YOLOE textual prompts applied.")
                    # 文本提示设置成功，直接返回（不再使用类索引过滤）
                    return
                finally:
                    signal.alarm(0)  # 关闭闹钟
            except AttributeError:
                # 平台不支持 signal 或模型不支持文本提示接口
                pass
        except (TimeoutError, Exception) as e:
            print(f"Warning: YOLOE text prompt setup failed ({e}); falling back to class index filter.")

        # 回退：按类别名映射到索引
        try:
            name_map = getattr(self.model, 'names', None)
            class_ids = []
            if isinstance(name_map, dict):
                inv = {str(v).lower(): k for k, v in name_map.items()}
                for c in target_classes:
                    cid = inv.get(str(c).lower(), None)
                    if cid is not None:
                        class_ids.append(cid)
            elif isinstance(name_map, (list, tuple)):
                inv = {str(v).lower(): i for i, v in enumerate(name_map)}
                for c in target_classes:
                    cid = inv.get(str(c).lower(), None)
                    if cid is not None:
                        class_ids.append(cid)
            if class_ids:
                self.selected_class_ids = class_ids
                print(f"Using class index filter: {class_ids}")
            else:
                print("Warning: none of target classes found in model names; will detect all classes.")
        except Exception as e:
            print(f"Warning: class setup failed: {e}")
    
    def detect_single(self, frame, confidence_threshold=0.25):
        """
        检测单个最高置信度目标
        
        Args:
            frame: 输入图像(BGR格式)
            confidence_threshold: 置信度阈值
            
        Returns:
            bbox: (x, y, w, h) 格式的边界框，或 None
            confidence: 置信度分数，或 None
        """
        results = self.model(
            frame, 
            conf=confidence_threshold,
            verbose=False,
            classes=self.selected_class_ids
        )
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            
            if len(confidences) > 0:
                max_idx = confidences.argmax()
                max_conf = confidences[max_idx]
                
                # 转换为 xywh 格式
                xyxy = boxes[max_idx].xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                
                return bbox, float(max_conf)
        
        return None, None
    
    def detect_all(self, frame, confidence_threshold=0.25):
        """
        检测所有候选目标
        
        Args:
            frame: 输入图像(BGR格式)
            confidence_threshold: 置信度阈值
            
        Returns:
            all_bboxes: List[(x, y, w, h)] 所有检测框
            masks: List[np.ndarray] 或 None，每个元素为 HxW 的uint8二值掩码
        """
        results = self.model(
            frame,
            conf=confidence_threshold,
            verbose=False,
            classes=self.selected_class_ids
        )
        
        all_bboxes = []
        masks_np = None
        
        if results and results[0].boxes:
            r = results[0]
            
            # 提取所有边界框
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                all_bboxes.append(bbox)
            
            # 处理掩码（如果模型支持分割）
            if getattr(r, 'masks', None) is not None and getattr(r.masks, 'data', None) is not None:
                try:
                    masks_tensor = r.masks.data
                    masks_np_raw = (masks_tensor > 0.5).detach().cpu().numpy().astype('uint8')
                    H, W = frame.shape[:2]
                    masks_np = []
                    for m in masks_np_raw:
                        if m.shape[0] != H or m.shape[1] != W:
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                        masks_np.append(m.astype('uint8'))
                except Exception:
                    masks_np = None
        
        return all_bboxes, masks_np


# ============ 兼容旧接口的函数 ============

def initialize_yolo_model(model_path="weights/yolo/yoloe-11l-seg.pt", target_classes=None):
    """
    初始化YOLO模型（兼容旧接口）
    
    Args:
        model_path: YOLO模型权重路径
        target_classes: 目标类别列表（可选）
        
    Returns:
        YOLODetector 实例
    """
    return YOLODetector(model_path, target_classes)


def perform_yolo_detection(frame, yolo_model, confidence_threshold=0.25, save_dir=None, frame_idx=None):
    """
    使用YOLO模型进行检测，返回置信度最高的单个目标（兼容旧接口）
    
    Args:
        frame: 输入图像
        yolo_model: YOLO模型实例（YOLODetector 或原始 YOLO）
        confidence_threshold: 置信度阈值
        save_dir: 保存目录（未使用，保留兼容性）
        frame_idx: 帧索引（未使用，保留兼容性）
        
    Returns:
        bbox: (x, y, w, h) 格式的边界框
        confidence: 置信度分数
    """
    if isinstance(yolo_model, YOLODetector):
        return yolo_model.detect_single(frame, confidence_threshold)
    
    # 如果传入的是原始 YOLO 模型，兼容处理
    results = yolo_model(frame, conf=confidence_threshold, verbose=False,
                        classes=getattr(yolo_model, "selected_class_ids", None))
    
    if results and results[0].boxes:
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        
        if len(confidences) > 0:
            max_idx = confidences.argmax()
            max_conf = confidences[max_idx]
            xyxy = boxes[max_idx].xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = xyxy
            w = x_max - x_min
            h = y_max - y_min
            bbox = (int(x_min), int(y_min), int(w), int(h))
            return bbox, max_conf
    
    return None, None


def perform_yolo_detection_for_candidates(frame, yolo_model, confidence_threshold=0.25):
    """
    使用YOLO模型进行检测，返回所有候选目标（兼容旧接口）
    
    Args:
        frame: 输入图像
        yolo_model: YOLO模型实例（YOLODetector 或原始 YOLO）
        confidence_threshold: 置信度阈值
        
    Returns:
        all_bboxes: List[(x, y, w, h)] 所有检测框
        masks: List[np.ndarray] 或 None
    """
    if isinstance(yolo_model, YOLODetector):
        return yolo_model.detect_all(frame, confidence_threshold)
    
    # 如果传入的是原始 YOLO 模型，兼容处理
    results = yolo_model(frame, conf=confidence_threshold, verbose=False,
                        classes=getattr(yolo_model, "selected_class_ids", None))
    
    all_bboxes = []
    masks_np = None
    
    if results and results[0].boxes:
        r = results[0]
        boxes = r.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = xyxy
            w = x_max - x_min
            h = y_max - y_min
            bbox = (int(x_min), int(y_min), int(w), int(h))
            all_bboxes.append(bbox)
        
        # 处理掩码
        if getattr(r, 'masks', None) is not None and getattr(r.masks, 'data', None) is not None:
            try:
                masks_tensor = r.masks.data
                masks_np_raw = (masks_tensor > 0.5).detach().cpu().numpy().astype('uint8')
                H, W = frame.shape[:2]
                masks_np = []
                for m in masks_np_raw:
                    if m.shape[0] != H or m.shape[1] != W:
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    masks_np.append(m.astype('uint8'))
            except Exception:
                masks_np = None
    
    return all_bboxes, masks_np



