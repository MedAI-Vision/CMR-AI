import torch.nn as nn
from mmcv.cnn import normal_init
import torch
import torch.nn.functional as F

from ..builder import HEADS
from .base import BaseHead
import os
import numpy as np
import pickle as pkl

@HEADS.register_module()

class fusion_ConcatHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_mod,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_mod = num_mod
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        # self.fc_cls = nn.Linear(self.in_channels*num_mod+128, self.num_classes)
        self.fc_cls = nn.Linear(self.in_channels*num_mod, self.num_classes)
        # self.fc_code = nn.Linear(13, 128)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)
        # normal_init(self.fc_code, std=self.init_std)

    def forward(self, x, labels):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # print(x.shape)
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            for i in range(self.num_mod):
                x[i] = self.avg_pool(x[i])
        # [N, in_channels, 1, 1, 1]
        x = torch.cat(x, dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        # print(x.shape)
        x = x.view(x.shape[0], -1)
        # print(x.shape)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        # print(cls_score.shape)
        return cls_score


@HEADS.register_module()
class fusion_AttnHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 num_heads=2,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        modality_num = 2
        attn_linear1 = []
        attn_linear2 = []
        self.num_heads = num_heads
        self.modality_num = modality_num
        for i in range(num_heads):
            attn_linear1.append(
                nn.Linear(self.in_channels, self.in_channels, bias=True))
            attn_linear2.append(nn.Linear(self.in_channels, 1, bias=False))
        self.attn_linear1 = nn.ModuleList(attn_linear1)
        self.attn_linear2 = nn.ModuleList(attn_linear2)
        self.alpha = nn.Parameter(torch.ones(num_heads), requires_grad=True)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for i in range(self.num_heads):
            normal_init(self.attn_linear1[i], std=self.init_std)
            normal_init(self.attn_linear2[i], std=self.init_std)
        normal_init(self.fc_cls, std=self.init_std)

    def attn_pooling(self, x):
        batch_size = x.size(0)
        emb_attn = torch.zeros(batch_size, self.num_heads, self.modality_num).to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in range(self.num_heads):
            emb_squish = torch.tanh(self.attn_linear1[i](x))
            emb_attn[:, i, :] = self.attn_linear2[i](emb_squish).squeeze(2)

        alpha_limit = F.softmax(self.alpha, dim=0)
        emb_attn = torch.matmul(alpha_limit, emb_attn)
        emb_attn_norm = F.softmax(emb_attn, dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(
            1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors

    def forward(self, x1, x2):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
        # [N, in_channels, 1, 1, 1]
        x1 = x1.view(-1, 1, self.in_channels)
        x2 = x2.view(-1, 1, self.in_channels)
        x = torch.cat([x1, x2], dim=1)
        x = x.view(-1, self.in_channels)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(-1, 2, self.in_channels)
        x = self.attn_pooling(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


@HEADS.register_module()
class fusion_CnnHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 in_planes=132,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.in_planes = in_planes
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels+in_planes, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x1, x2):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x1 = self.avg_pool(x1)
        # [N, in_channels, 1, 1, 1]
        x2 = x2.view(x2.size(0), self.in_planes, 1, 1, 1)
        x = torch.cat([x1, x2], dim=1)
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


@HEADS.register_module()
class fusion_DecisionHead(BaseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 num_heads=2,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls1 = nn.Linear(self.in_channels, self.num_classes)
        self.fc_cls2 = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        modality_num = 2
        attn_linear1 = []
        attn_linear2 = []
        self.num_heads = num_heads
        self.modality_num = modality_num
        for i in range(num_heads):
            attn_linear1.append(
                nn.Linear(self.num_classes, self.num_classes, bias=True))
            attn_linear2.append(nn.Linear(num_classes, 1, bias=False))
        self.attn_linear1 = nn.ModuleList(attn_linear1)
        self.attn_linear2 = nn.ModuleList(attn_linear2)
        self.alpha = nn.Parameter(torch.ones(num_heads), requires_grad=True)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for i in range(self.num_heads):
            normal_init(self.attn_linear1[i], std=self.init_std)
            normal_init(self.attn_linear2[i], std=self.init_std)
        normal_init(self.fc_cls1, std=self.init_std)
        normal_init(self.fc_cls2, std=self.init_std)

    def attn_pooling(self, x):
        batch_size = x.size(0)
        emb_attn = torch.zeros(batch_size, self.num_heads, self.modality_num).to(
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in range(self.num_heads):
            emb_squish = torch.tanh(self.attn_linear1[i](x))
            emb_attn[:, i, :] = self.attn_linear2[i](emb_squish).squeeze(2)

        alpha_limit = F.softmax(self.alpha, dim=0)
        emb_attn = torch.matmul(alpha_limit, emb_attn)
        emb_attn_norm = F.softmax(emb_attn, dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(
            1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors

    def forward(self, x1, x2):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x1 = self.avg_pool(x1)
            x2 = self.avg_pool(x2)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x1 = self.dropout(x1)
            x2 = self.dropout(x2)
        # [N, in_channels, 1, 1, 1]
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        # [N, in_channels]
        cls_score1 = self.fc_cls1(x1)
        cls_score2 = self.fc_cls2(x2)
        # [N, num_classes]
        cls_score1 = cls_score1.view(-1, 1, self.num_classes)
        cls_score2 = cls_score2.view(-1, 1, self.num_classes)
        cls_score = torch.cat([cls_score1, cls_score2], dim=1)
        cls_score = self.attn_pooling(cls_score)
        return cls_score
