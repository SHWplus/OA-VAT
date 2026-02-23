"""
项目配置文件

统一管理所有路径、默认参数等配置项。
"""
import os

# ============ 路径配置 ============

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 权重文件目录
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, 'weights')
YOLO_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'yolo')
ORTRACK_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'ortrack')
DINOV3_WEIGHTS_DIR = os.path.join(WEIGHTS_DIR, 'dinov3')

# 第三方库路径
THIRD_PARTY_DIR = os.path.join(PROJECT_ROOT, 'third_party')
ORTRACK_PATH = os.path.join(PROJECT_ROOT, 'ORTrack')  # 当前 ORTrack 仍在项目根目录

# DINOv3 路径
DINOV3_PATH = os.path.join(PROJECT_ROOT, 'dinov3-main')

# ============ 默认权重文件 ============

# YOLO 默认权重
DEFAULT_YOLO_MODEL = os.path.join(YOLO_WEIGHTS_DIR, 'yoloe-11l-seg.pt')
# 如果 weights 目录不存在，回退到 ORTrack 原路径
if not os.path.exists(DEFAULT_YOLO_MODEL):
    DEFAULT_YOLO_MODEL = os.path.join(ORTRACK_PATH, 'weights', 'yoloe-11l-seg.pt')

# ORTrack 默认权重
DEFAULT_ORTRACK_CHECKPOINT = os.path.join(
    ORTRACK_WEIGHTS_DIR, 
    'ORTrack_ep0300.pth.tar'
)
# 如果 weights 目录不存在，回退到 ORTrack 原路径
if not os.path.exists(DEFAULT_ORTRACK_CHECKPOINT):
    DEFAULT_ORTRACK_CHECKPOINT = os.path.join(
        ORTRACK_PATH,
        'output/checkpoints/train/ortrack/deit_tiny_patch16_224/ORTrack_ep0300.pth.tar'
    )

# DINOv3 默认权重
DEFAULT_DINOV3_CHECKPOINT = os.path.join(
    DINOV3_WEIGHTS_DIR,
    'dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
)
# 如果 weights 目录不存在，回退到原路径
if not os.path.exists(DEFAULT_DINOV3_CHECKPOINT):
    # 尝试 dinov3-main/dinov3/weights/ 路径
    alt_path = os.path.join(DINOV3_PATH, 'dinov3/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')
    if os.path.exists(alt_path):
        DEFAULT_DINOV3_CHECKPOINT = alt_path
    else:
        # 回退到 checkpoints/ 目录
        DEFAULT_DINOV3_CHECKPOINT = os.path.join(PROJECT_ROOT, 'checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth')

# ============ 默认参数配置 ============

# YOLO 默认参数
YOLO_DEFAULTS = {
    'confidence_threshold': 0.25,
    'text_query': '',  # 空字符串表示检测所有类别
}

# ORTrack 默认参数
ORTRACK_DEFAULTS = {
    'model_name': 'deit_tiny_patch16_224',
    'lost_threshold': 0.6,
    'max_lost_frames': 3,
    'template_size': 128,
    'search_size': 256,
    'keep_last_position': True,
}

# DINOv3 默认参数
DINOV3_DEFAULTS = {
    'model_name': 'dinov3_vith16plus',
}

# ============ 辅助函数 ============

def get_weight_path(weight_type, filename=None):
    """
    获取权重文件路径
    
    Args:
        weight_type: 'yolo', 'ortrack', 或 'dinov3'
        filename: 权重文件名（可选，使用默认值）
        
    Returns:
        权重文件的完整路径
    """
    if weight_type == 'yolo':
        if filename:
            return os.path.join(YOLO_WEIGHTS_DIR, filename)
        return DEFAULT_YOLO_MODEL
    elif weight_type == 'ortrack':
        if filename:
            return os.path.join(ORTRACK_WEIGHTS_DIR, filename)
        return DEFAULT_ORTRACK_CHECKPOINT
    elif weight_type == 'dinov3':
        if filename:
            return os.path.join(DINOV3_WEIGHTS_DIR, filename)
        return DEFAULT_DINOV3_CHECKPOINT
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")


def ensure_directories():
    """确保所有必要的目录存在"""
    dirs_to_create = [
        WEIGHTS_DIR,
        YOLO_WEIGHTS_DIR,
        ORTRACK_WEIGHTS_DIR,
        DINOV3_WEIGHTS_DIR,
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


if __name__ == '__main__':
    # 测试配置
    print("=" * 60)
    print("项目配置信息")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"\n权重目录:")
    print(f"  - YOLO:    {YOLO_WEIGHTS_DIR}")
    print(f"  - ORTrack: {ORTRACK_WEIGHTS_DIR}")
    print(f"  - DINOv3:  {DINOV3_WEIGHTS_DIR}")
    print(f"\n默认权重文件:")
    print(f"  - YOLO:    {DEFAULT_YOLO_MODEL}")
    print(f"    存在: {os.path.exists(DEFAULT_YOLO_MODEL)}")
    print(f"  - ORTrack: {DEFAULT_ORTRACK_CHECKPOINT}")
    print(f"    存在: {os.path.exists(DEFAULT_ORTRACK_CHECKPOINT)}")
    print(f"  - DINOv3:  {DEFAULT_DINOV3_CHECKPOINT}")
    print(f"    存在: {os.path.exists(DEFAULT_DINOV3_CHECKPOINT)}")
    print("=" * 60)


