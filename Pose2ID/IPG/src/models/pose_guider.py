from typing import Tuple

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

from src.models.motion_module import zero_module
from src.models.resnet import InflatedConv3d


class PoseGuider(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int = 320,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding

from torchvision.models import vit_b_16
# class PoseGuider_dit(ModelMixin):
#     def __init__(self):
#         super().__init__()
#         self.vit = vit_b_16(pretrained=True)  # 使用预训练的ViT模型
#         self.vit.heads = nn.Identity()  # 去掉ViT的分类头，只提取特征
#     def forward(self, conditioning):
#         embedding = self.vit(conditioning)  # 使用ViT编码图像
#         return embedding  # 返回768维特征向量


# input :[24, 4, 1, 64, 32]
# output:[24, 4, 1, 64, 32]

# class PoseGuider_dit(ModelMixin):
#     def __init__(
#         self,
#     ):
#         super().__init__()
#         self.conv_in = InflatedConv3d(
#             4, 16, kernel_size=3, padding=1
#         )
#         self.blocks = nn.ModuleList([])
#         self.blocks.append(
#             InflatedConv3d(16, 16, kernel_size=3, padding=1)
#         )
#         self.blocks.append(
#             InflatedConv3d(16, 32, kernel_size=3, padding=1)
#         )
#         self.blocks.append(
#             InflatedConv3d(32, 16, kernel_size=3, padding=1)
#         )
#         self.blocks.append(
#             InflatedConv3d(16, 4, kernel_size=3, padding=1)
#         )
        
       

#     def forward(self, conditioning):
#         embedding = self.conv_in(conditioning)
#         embedding = F.silu(embedding)
#         for block in self.blocks:
#             embedding = block(embedding)
#             embedding = F.silu(embedding)
#         return embedding


class PoseGuider_dit(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int = 4,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 32, 16),
    ):
        super().__init__()
        self.conv_in = InflatedConv3d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                InflatedConv3d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                InflatedConv3d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )

        self.conv_out = zero_module(
            InflatedConv3d(
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
  