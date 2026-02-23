"""参考特征存储与管理"""
import threading
import torch
import torch.nn.functional as F


class ReferenceFeatureStore:
    """线程安全的参考特征存储（支持EMA更新）"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._feature = None  # torch.Tensor [C] on CPU

    def initialize_from_list(self, feature_list):
        """从特征列表初始化（取平均并归一化）
        
        Args:
            feature_list: torch.Tensor 列表
        """
        if not isinstance(feature_list, list) or len(feature_list) == 0:
            return
        with self._lock:
            feats = []
            for f in feature_list:
                if isinstance(f, torch.Tensor):
                    t = f.detach().view(-1).float().cpu()
                    t = F.normalize(t, dim=0)
                    feats.append(t)
            if len(feats) == 0:
                return
            mean_feat = torch.stack(feats, dim=0).mean(dim=0)
            self._feature = F.normalize(mean_feat, dim=0)

    def is_valid(self):
        """检查是否已初始化
        
        Returns:
            bool: 是否有效
        """
        with self._lock:
            return self._feature is not None

    def get_features(self):
        """获取特征列表（返回list以复用现有匹配逻辑）
        
        Returns:
            list: 特征tensor列表
        """
        with self._lock:
            if self._feature is None:
                return []
            return [self._feature.clone()]

    def get_feature_vector(self):
        """获取特征向量（单个tensor）
        
        Returns:
            torch.Tensor or None: 特征向量
        """
        with self._lock:
            if self._feature is None:
                return None
            return self._feature.clone()

    def update_with_feature(self, current_feature, alpha=0.2):
        """EMA更新参考特征
        
        Args:
            current_feature: 当前特征tensor
            alpha: EMA系数
        """
        if current_feature is None:
            return
        cur = current_feature.detach().view(-1).float().cpu()
        cur = F.normalize(cur, dim=0)
        with self._lock:
            if self._feature is None:
                self._feature = cur.clone()
            else:
                fused = float(alpha) * self._feature + (1.0 - float(alpha)) * cur
                self._feature = F.normalize(fused, dim=0)


