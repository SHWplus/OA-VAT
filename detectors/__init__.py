"""
统一的检测器和跟踪器接口模块

提供 YOLO 检测器和 ORTrack 跟踪器的封装，
以及兼容旧代码的函数接口。
"""

from .yolo_detector import (
    YOLODetector,
    initialize_yolo_model,
    perform_yolo_detection,
    perform_yolo_detection_for_candidates,
)

from .ortrack_tracker import (
    initialize_ortrack_tracker,
)

__all__ = [
    # YOLO 相关
    'YOLODetector',
    'initialize_yolo_model',
    'perform_yolo_detection',
    'perform_yolo_detection_for_candidates',
    
    # ORTrack 相关
    'initialize_ortrack_tracker',
]



