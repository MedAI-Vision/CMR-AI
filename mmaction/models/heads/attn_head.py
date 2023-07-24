import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class AttnHead(BaseHead):
    """Classification head for Attention.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

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
            
        d_model = in_channels
        max_len = 7
        attn_linear1 = []
        attn_linear2 = []
        self.num_heads = num_heads
        self.seq_len = max_len
        for i in range(num_heads):
            attn_linear1.append(nn.Linear(d_model, d_model, bias=True))
            attn_linear2.append(nn.Linear(d_model, 1, bias=False))
        self.attn_linear1 = nn.ModuleList(attn_linear1)
        self.attn_linear2 = nn.ModuleList(attn_linear2)
        self.alpha = nn.Parameter(torch.ones(num_heads), requires_grad=True)
#        self.pos_emb = nn.Parameter(torch.zeros(1, 7, 1024), requires_grad=True)
        
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        for i in range(self.num_heads):
            normal_init(self.attn_linear1[i], std=self.init_std)
            normal_init(self.attn_linear2[i], std=self.init_std)
        normal_init(self.fc_cls, std=self.init_std)
        
    def attn_pooling(self, x):
        batch_size = x.size(0)
        emb_attn = torch.zeros(batch_size, self.num_heads, self.seq_len).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for i in range(self.num_heads):
            emb_squish = torch.tanh(self.attn_linear1[i](x))
            emb_attn[:, i, :] = self.attn_linear2[i](emb_squish).squeeze(2)

        alpha_limit = F.softmax(self.alpha, dim=0)
        emb_attn = torch.matmul(alpha_limit, emb_attn)
        emb_attn_norm = F.softmax(emb_attn, dim=0)
        emb_attn_vectors = torch.bmm(x.transpose(1, 2), emb_attn_norm.unsqueeze(2)).squeeze(2)
        return emb_attn_vectors

    def forward(self, x):
    
        batch_size, num_channels, num_frames, width, height = x.size()
        x = x.contiguous()
        x = x.view(-1, num_frames, width, height)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        
        x = x.view(batch_size, num_frames, -1)
        x = self.attn_pooling(x)
        
        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(batch_size, -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
