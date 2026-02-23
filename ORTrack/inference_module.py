# inference_module.py (重命名 inference_script.py 或复制内容到新文件)
import os
import sys
import cv2
import torch
import yaml
# from PIL import Image # PIL Image未使用，可以注释掉
from ultralytics import YOLO  # 确保 ultralytics 已安装

# import time # time 未在暴露的函数中使用

# 添加项目路径 - 这部分对于ORTrack很重要，需要确保路径正确
# 在 follow_anything.py 中调用时，可能需要一种更灵活的方式传递此路径
# 或者期望 ORTrack 库本身能处理好其内部路径
DEFAULT_PRJ_PATH = '/home/gao/ORTrack'  # 保持一个默认值，但最好能外部配置


# 从 lib.test.tracker.ortrack import ORTrack
# ^^^ 这行导入需要 prj_path 先生效。

# 统一输出目录 - 这个可以由调用者（follow_anything.py）管理
# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

class ConfigObject:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)

    def __getattr__(self, name):
        # 适当处理属性不存在的情况，例如返回None或抛出更具体的错误
        # print(f"Warning: ConfigObject attribute '{name}' not found, returning None.")
        return None


def initialize_yolo_model(model_path="weights/yolov8x-worldv2.pt", target_classes=None):
    """初始化YOLO模型并设置目标类别（支持YOLOE文本提示，旧版退化为类索引过滤）"""
    yolo_model = YOLO(model_path)
    # 清空选择类索引
    setattr(yolo_model, "selected_class_ids", None)
    if target_classes:
        print(f"===== YOLO: Setting target classes/prompts: {target_classes} =====")
        # 优先尝试YOLOE文本提示
        try:
            text_pe = yolo_model.get_text_pe(target_classes)
            yolo_model.set_classes(target_classes, text_pe)
            print("YOLOE textual prompts applied.")
        except AttributeError:
            # 映射为类索引以便在predict时传入classes=
            try:
                name_map = getattr(yolo_model, 'names', None)
                if isinstance(name_map, dict):
                    inv = {v: k for k, v in name_map.items()}
                    class_ids = [inv[c] for c in target_classes if c in inv]
                    if class_ids:
                        setattr(yolo_model, "selected_class_ids", class_ids)
                        print(f"Using class index filter: {class_ids}")
                    else:
                        print("Warning: none of target classes found in model names; will detect all classes.")
                else:
                    print("Warning: model.names not available; will detect all classes.")
            except Exception as e:
                print(f"Warning: class index mapping failed: {e}")
                setattr(yolo_model, "selected_class_ids", None)
    else:
        print("===== YOLO: Tracking all classes =====")
    return yolo_model


def perform_yolo_detection(frame, yolo_model, confidence_threshold=0.25, save_dir=None, frame_idx=None):
    """
    使用YOLO模型进行检测，返回置信度最高的单个目标的bbox和置信度。
    bbox格式: (x_min, y_min, width, height)
    """
    results = yolo_model(frame, conf=confidence_threshold, verbose=False,
                         classes=getattr(yolo_model, "selected_class_ids", None))  # verbose=False 减少控制台输出

    if results and results[0].boxes:
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()

        if len(confidences) > 0:
            max_idx = confidences.argmax()
            max_conf = confidences[max_idx]

            # 获取xyxy格式的边界框并转换为xywh
            xyxy_box = boxes[max_idx].xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = xyxy_box
            w = x_max - x_min
            h = y_max - y_min
            bbox = (int(x_min), int(y_min), int(w), int(h))

            if save_dir and frame_idx is not None:
                # 调用一个通用的保存函数，如果需要的话
                # save_annotated_frame(frame, bbox, save_dir, frame_idx, max_conf, "detection_yolo")
                pass  # 保存逻辑可以在 follow_anything.py 中处理
            return bbox, max_conf

    return None, None


def perform_yolo_detection_for_candidates(frame, yolo_model, confidence_threshold=0.25):
    """
    使用YOLO模型进行检测，为后续的特征匹配返回所有候选目标的bbox列表与掩码。
    返回:
      - all_bboxes: List[(x_min, y_min, width, height)]
      - masks: List[np.ndarray] or None  (每个元素为 HxW 的uint8二值掩码)
    """
    results = yolo_model(frame, conf=confidence_threshold, verbose=False,
                         classes=getattr(yolo_model, "selected_class_ids", None))
    
    all_bboxes = []
    masks_np = None
    if results and results[0].boxes:
        r = results[0]
        boxes = r.boxes
        for box in boxes:
            # 获取xyxy格式的边界框并转换为xywh
            xyxy_box = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = xyxy_box
            w = x_max - x_min
            h = y_max - y_min
            bbox = (int(x_min), int(y_min), int(w), int(h))
            all_bboxes.append(bbox)
        # 处理掩码
        if getattr(r, 'masks', None) is not None and getattr(r.masks, 'data', None) is not None:
            try:
                masks_tensor = r.masks.data
                masks_np_raw = (masks_tensor > 0.5).detach().cpu().numpy().astype('uint8')  # [N,H,W]
                H, W = frame.shape[:2]
                masks_np = []
                for m in masks_np_raw:
                    if m.shape[0] != H or m.shape[1] != W:
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                    masks_np.append(m.astype('uint8'))
            except Exception:
                masks_np = None
            
    return all_bboxes, masks_np


def initialize_ortrack_tracker(ortrack_prj_path=DEFAULT_PRJ_PATH, model_name='deit_tiny_patch16_224',
                               checkpoint_path=None,
                               lost_threshold=0.3, max_lost_frames=10, keep_last_position=True,
                               template_size=128, search_size=256):  # keep_last_position默认False
    """初始化ORTrack跟踪器"""
    if ortrack_prj_path not in sys.path:
        sys.path.append(ortrack_prj_path)

    # 动态导入 ORTrack，因为它依赖于 prj_path
    try:
        from lib.test.tracker.ortrack import ORTrack
    except ImportError as e:
        print(
            f"Error importing ORTrack: {e}. Ensure prj_path ('{ortrack_prj_path}') is correct and ORTrack is compiled.")
        raise

    cfg_file = os.path.join(ortrack_prj_path, f"experiments/ortrack/{model_name}.yaml")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"ORTrack config file not found: {cfg_file}")

    with open(cfg_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)

    cfg = ConfigObject(cfg_dict)
    cfg.workspace_dir = ortrack_prj_path  # 确保 workspace_dir 被正确设置

    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            ortrack_prj_path,
            f"output/checkpoints/train/ortrack/{model_name}/ORTrack_ep0300.pth.tar"
        )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ORTrack checkpoint not found: {checkpoint_path}")

    class TrackerParams:
        def __init__(self):
            self.cfg = cfg
            self.checkpoint = checkpoint_path
            self.debug = False  # 可根据需要设为True
            self.save_all_boxes = False

            self.lost_threshold = lost_threshold
            self.max_lost_frames = max_lost_frames
            self.keep_last_position = keep_last_position

            # 确保这些尺寸与传入的cfg中的TEST配置一致或覆盖它们
            # self.template_factor = cfg.TEST.TEMPLATE_FACTOR if hasattr(cfg, 'TEST') and cfg.TEST else 2.0
            # self.template_size = cfg.TEST.TEMPLATE_SIZE if hasattr(cfg, 'TEST') and cfg.TEST else 128
            # self.search_factor = cfg.TEST.SEARCH_FACTOR if hasattr(cfg, 'TEST') and cfg.TEST else 4.0
            # self.search_size = cfg.TEST.SEARCH_SIZE if hasattr(cfg, 'TEST') and cfg.TEST else 256
            self.template_factor = 2.0  # 与inference_script.py中TrackerParams保持一致
            self.template_size = template_size
            self.search_factor = 4.0
            self.search_size = search_size

            # 更新config对象中的测试参数以匹配TrackerParams，如果需要
            if hasattr(cfg, 'TEST'):
                cfg.TEST.TEMPLATE_SIZE = self.template_size
                cfg.TEST.SEARCH_SIZE = self.search_size
                # cfg.TEST.TEMPLATE_FACTOR = self.template_factor # 如果需要
                # cfg.TEST.SEARCH_FACTOR = self.search_factor # 如果需要

    params = TrackerParams()
    print("===== ORTrack Tracker Config =====")
    print(f"  Model: {model_name}")
    print(f"  Checkpoint: {params.checkpoint}")
    print(f"  Lost Threshold: {params.lost_threshold}")
    print(f"  Max Lost Frames: {params.max_lost_frames}")
    print(f"  Template Size: {params.template_size}")
    print(f"  Search Size: {params.search_size}")

    # dataset_name="custom" 似乎是ORTrack内部的一个参数，暂时保留
    return ORTrack(params, dataset_name="custom")

# 不再需要 main 和 argparse 部分，因为这个脚本将作为模块导入
# if __name__ == "__main__":
#     main()
    # "/home/gao/ORTrack/example_videos/car_following.avi"
