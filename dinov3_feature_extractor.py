import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.ops import roi_align

# 导入配置
try:
    from config import DEFAULT_DINOV3_CHECKPOINT, DINOV3_PATH
except ImportError:
    # 如果 config.py 不存在，使用默认值
    DEFAULT_DINOV3_CHECKPOINT = 'dinov3-main/dinov3/weights/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
    DINOV3_PATH = 'dinov3-main'

class DINOv3FeatureExtractor(nn.Module):
    def __init__(
        self,
        model_name='dinov3_vith16plus',
        checkpoint_path=None,
        device="cuda",
        verbose: bool = False
    ):
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.verbose = verbose
        
        # 使用配置中的默认路径
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_DINOV3_CHECKPOINT
        
        # 1. Load DINOv3 model
        try:
            # 检查权重文件是否存在
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found at: {checkpoint_path}")
            
            torch.hub.set_dir('.')  # Load from current directory structure
            
            # 加载模型结构（不加载预训练权重）
            self.model = torch.hub.load(
                DINOV3_PATH, 
                self.model_name, 
                source='local', 
                pretrained=False  # 不自动加载权重，避免复制到 checkpoints/
            )
            
            # 手动加载本地权重文件
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            
            logging.info(f"DINOv3 model '{self.model_name}' loaded successfully from checkpoint '{checkpoint_path}'.")
        except Exception as e:
            logging.error(f"Failed to load DINOv3 model: {e}")
            raise

        # 2. 定义图像预处理
        # 标准ImageNet归一化
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(518, antialias=True),  # DINOv3标准分辨率
            transforms.CenterCrop(518),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理图像以适应DINOv3模型
        Args:
            image (np.ndarray): 输入图像，格式为HWC、BGR
        Returns:
            torch.Tensor: 预处理后的图像张量
        """
        # 转换BGR为RGB并应用变换
        image_rgb = image[..., ::-1].copy()
        return self.preprocess(image_rgb).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def extract_features(self, image: np.ndarray, bboxes: np.ndarray):
        """
        从图像中提取给定边界框的特征
        Args:
            image (np.ndarray): 输入图像，格式为HWC、BGR
            bboxes (np.ndarray): 边界框的numpy数组，形状为(N, 4)，格式为[x1, y1, x2, y2]
        Returns:
            torch.Tensor: 特征向量张量，形状为(N, feature_dim)
        """
        import time
        total_start_time = time.time()
        
        is_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")
        
        # === 1. 图像预处理 ===
        if is_cuda: torch.cuda.synchronize()
        preprocess_start = time.time()
        original_h, original_w, _ = image.shape
        preprocessed_image = self.preprocess_image(image)
        if is_cuda: torch.cuda.synchronize()
        preprocess_time = time.time() - preprocess_start
        if self.verbose:
            print(f"    DINOv3图像预处理耗时: {preprocess_time:.4f}s")
        
        # === 2. DINOv3前向推理 ===
        if is_cuda: torch.cuda.synchronize()
        forward_start = time.time()
        # 从模型的最后一层获取特征图
        # get_intermediate_layers返回张量的元组
        features = self.model.get_intermediate_layers(preprocessed_image, n=1, reshape=True)[0]
        if is_cuda: torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        if self.verbose:
            print(f"    DINOv3前向推理耗时: {forward_time:.4f}s")
        
        _, feature_dim, feature_h, feature_w = features.shape
        if self.verbose:
            print(f"    特征图尺寸: {feature_h}x{feature_w}, 特征维度: {feature_dim}")
        
        # === 3. 边界框坐标变换 ===
        bbox_transform_start = time.time()
        # 将边界框从原始图像大小缩放到特征图大小
        scaled_bboxes = bboxes.copy().astype(np.float32)
        scaled_bboxes[:, 0] *= feature_w / original_w
        scaled_bboxes[:, 2] *= feature_w / original_w
        scaled_bboxes[:, 1] *= feature_h / original_h
        scaled_bboxes[:, 3] *= feature_h / original_h

        # 限制坐标在特征图范围内
        scaled_bboxes[:, 0] = np.clip(scaled_bboxes[:, 0], 0, feature_w)
        scaled_bboxes[:, 1] = np.clip(scaled_bboxes[:, 1], 0, feature_h)
        scaled_bboxes[:, 2] = np.clip(scaled_bboxes[:, 2], 0, feature_w)
        scaled_bboxes[:, 3] = np.clip(scaled_bboxes[:, 3], 0, feature_h)
        bbox_transform_time = time.time() - bbox_transform_start
        if self.verbose:
            print(f"    边界框坐标变换耗时: {bbox_transform_time:.4f}s")

        # 过滤裁剪后面积为零或负的框
        valid_mask = (scaled_bboxes[:, 2] > scaled_bboxes[:, 0]) & (scaled_bboxes[:, 3] > scaled_bboxes[:, 1])
        
        if not np.all(valid_mask):
            logging.warning(f"过滤掉 {np.sum(~valid_mask)} 个面积为零/负的框。")
            original_indices = np.where(valid_mask)[0]
            scaled_bboxes = scaled_bboxes[valid_mask]
        else:
            original_indices = np.arange(len(bboxes))

        if scaled_bboxes.shape[0] == 0:
            logging.error("过滤后没有有效的边界框。")
            return torch.empty((0, feature_dim), device=self.device)

        # 准备RoIAlign
        box_indices = torch.zeros(len(scaled_bboxes), 1, device=self.device)
        rois = torch.cat([box_indices, torch.from_numpy(scaled_bboxes).to(self.device)], dim=1)

        # === 4. RoI特征提取 ===
        if is_cuda: torch.cuda.synchronize()
        roi_start = time.time()
        # 使用RoIAlign为每个框裁剪特征
        pooled_features = roi_align(features, rois, output_size=(1, 1), spatial_scale=1.0, aligned=True)
        
        # 重塑为(N, C)
        pooled_features = pooled_features.view(len(scaled_bboxes), feature_dim)
        if is_cuda: torch.cuda.synchronize()
        roi_time = time.time() - roi_start
        if self.verbose:
            print(f"    RoI特征池化耗时: {roi_time:.4f}s")

        # === 5. 特征张量组装 ===
        if is_cuda: torch.cuda.synchronize()
        assembly_start = time.time()
        # 创建全尺寸输出张量并填充有效框的特征
        final_features = torch.zeros((len(bboxes), feature_dim), device=self.device, dtype=pooled_features.dtype)
        final_features[original_indices] = pooled_features
        if is_cuda: torch.cuda.synchronize()
        assembly_time = time.time() - assembly_start
        
        total_time = time.time() - total_start_time
        if self.verbose:
            print(f"    特征张量组装耗时: {assembly_time:.4f}s")
            parts_time = preprocess_time + forward_time + bbox_transform_time + roi_time + assembly_time
            other_time = total_time - parts_time
            print(f"    耗时分解: 预处理({preprocess_time:.4f}s) + 前向推理({forward_time:.4f}s) + 坐标变换({bbox_transform_time:.4f}s) + RoI池化({roi_time:.4f}s) + 张量组装({assembly_time:.4f}s) + 其它({other_time:.4f}s)")
        # 总耗时（默认仅打印这一行）
        print(f"    DINOv3特征提取总耗时: {total_time:.4f}s")

        return final_features

if __name__ == '__main__':
    # 测试用例
    logging.basicConfig(level=logging.INFO)
    try:
        dummy_image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        dummy_bboxes = np.array([
            [100, 150, 200, 300],  # [x1, y1, x2, y2]
            [400, 50, 550, 250],
        ])

        print("初始化DINOv3FeatureExtractor...")
        # 示例中如果没有GPU则使用CPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        extractor = DINOv3FeatureExtractor(device=device)
        
        print("提取特征中...")
        features = extractor.extract_features(dummy_image, dummy_bboxes)
        
        print(f"成功为 {len(dummy_bboxes)} 个框提取特征。")
        print(f"特征张量形状: {features.shape}")
        # ViT-L/16的预期形状是(N, 1024)
        
    except Exception as e:
        print(f"示例运行过程中发生错误: {e}")
