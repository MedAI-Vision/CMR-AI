"""
Simple models for encoding dense representations of sequences
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


################################################################################
# Summing Models
################################################################################

class SeqSumPoolingEncoder(nn.Module):
    """Sum all frame representations into a single feature vector"""

    def __init__(self, n_classes, input_size):
        super(SeqSumPoolingEncoder, self).__init__()
        self.linear = nn.Linear(input_size, n_classes, bias=False)

    def init_hidden(self, batch_size):
        return None

    def embedding(self, x, hidden=None):
        """Get learned (summed) representation"""
        x = torch.sum(x, 1)
        return x

    def forward(self, x, hidden=None):
        x = torch.sum(x, 1)
        x = self.linear(x)
        return x