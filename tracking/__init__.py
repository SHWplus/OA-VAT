"""
跟踪模块

包含检测、跟踪、参考特征管理、工作线程等功能。
"""

from .reference_store import ReferenceFeatureStore
from .kalman_utils import (
    init_pkf_bbox,
    conf_to_noise,
    estimate_velocity_robustly
)
from .first_frame import detect_object_yolo
from .detection import match_candidates_with_reference

__all__ = [
    'ReferenceFeatureStore',
    'init_pkf_bbox',
    'conf_to_noise',
    'estimate_velocity_robustly',
    'detect_object_yolo',
    'match_candidates_with_reference',
]


