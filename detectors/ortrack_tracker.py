"""ORTrack 跟踪器封装模块"""
import os
import sys
import yaml


class ConfigObject:
    """配置对象，将字典转换为对象属性访问"""
    
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value))
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name):
        # 属性不存在时返回 None
        return None


def initialize_ortrack_tracker(
    ortrack_prj_path='ORTrack',
    model_name='deit_tiny_patch16_224',
    checkpoint_path=None,
    lost_threshold=0.3,
    max_lost_frames=10,
    keep_last_position=True,
    template_size=128,
    search_size=256
):
    """
    初始化ORTrack跟踪器
    
    Args:
        ortrack_prj_path: ORTrack 项目路径
        model_name: 模型名称
        checkpoint_path: 权重文件路径（None 则使用默认路径）
        lost_threshold: 置信度丢失阈值
        max_lost_frames: 最大丢失帧数
        keep_last_position: 是否保持最后位置
        template_size: 模板尺寸
        search_size: 搜索尺寸
        
    Returns:
        ORTrack 跟踪器实例
    """
    # 添加 ORTrack 项目路径到 sys.path
    if ortrack_prj_path not in sys.path:
        sys.path.append(ortrack_prj_path)
    
    # 动态导入 ORTrack（依赖于 prj_path）
    try:
        from lib.test.tracker.ortrack import ORTrack
    except ImportError as e:
        raise ImportError(
            f"Error importing ORTrack: {e}. "
            f"Ensure prj_path ('{ortrack_prj_path}') is correct and ORTrack is compiled."
        )
    
    # 加载配置文件
    cfg_file = os.path.join(ortrack_prj_path, f"experiments/ortrack/{model_name}.yaml")
    if not os.path.exists(cfg_file):
        raise FileNotFoundError(f"ORTrack config file not found: {cfg_file}")
    
    with open(cfg_file, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    cfg = ConfigObject(cfg_dict)
    cfg.workspace_dir = ortrack_prj_path  # 确保 workspace_dir 被正确设置
    
    # 确定权重文件路径
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            ortrack_prj_path,
            f"output/checkpoints/train/ortrack/{model_name}/ORTrack_ep0300.pth.tar"
        )

    # 若默认路径不存在，回退到项目根下的 weights/ortrack/ 中自动查找
    if not os.path.exists(checkpoint_path):
        try:
            # 计算项目根路径（detectors/ 的上一级）
            detectors_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(detectors_dir)
            candidate_dir = os.path.join(project_root, 'weights', 'ortrack')
            found = None
            if os.path.isdir(candidate_dir):
                # 优先匹配 ORTrack_ep*.pth / .pth.tar，其次任意 .pth* 文件
                prefer = []
                others = []
                for root, _, files in os.walk(candidate_dir):
                    for fn in files:
                        lower = fn.lower()
                        if lower.endswith(('.pth', '.pth.tar', '.pt', '.ckpt')):
                            full = os.path.join(root, fn)
                            if lower.startswith('ortrack_ep'):
                                prefer.append(full)
                            else:
                                others.append(full)
                found = (sorted(prefer) + sorted(others))[0] if (prefer or others) else None
            if found is not None and os.path.exists(found):
                print(f"[ORTrack] Using checkpoint from weights folder: {found}")
                checkpoint_path = found
        except Exception:
            pass

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"ORTrack checkpoint not found: {checkpoint_path}")
    
    # 创建跟踪器参数对象
    class TrackerParams:
        def __init__(self):
            self.cfg = cfg
            self.checkpoint = checkpoint_path
            self.debug = False
            self.save_all_boxes = False
            
            self.lost_threshold = lost_threshold
            self.max_lost_frames = max_lost_frames
            self.keep_last_position = keep_last_position
            
            # 设置模板和搜索尺寸
            self.template_factor = 2.0
            self.template_size = template_size
            self.search_factor = 4.0
            self.search_size = search_size
            
            # 更新 config 对象中的测试参数
            if hasattr(cfg, 'TEST'):
                cfg.TEST.TEMPLATE_SIZE = self.template_size
                cfg.TEST.SEARCH_SIZE = self.search_size
    
    params = TrackerParams()
    
    print("===== ORTrack Tracker Config =====")
    print(f"  Model: {model_name}")
    print(f"  Checkpoint: {params.checkpoint}")
    print(f"  Lost Threshold: {params.lost_threshold}")
    print(f"  Max Lost Frames: {params.max_lost_frames}")
    print(f"  Template Size: {params.template_size}")
    print(f"  Search Size: {params.search_size}")
    
    # 创建并返回跟踪器实例
    return ORTrack(params, dataset_name="custom")



