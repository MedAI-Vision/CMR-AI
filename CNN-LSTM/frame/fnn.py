import torch.nn as nn
import torch.nn.functional as F

################################################################################
# FNN Models
################################################################################

class FNNFrameEncoder(nn.Module):

    def __init__(self, input_size=1024, layers=[64, 32]):
        super(FNNFrameEncoder, self).__init__()
        layers = [input_size] + layers
        for i, size in enumerate(layers[:-1]):
            self.add_module('fc{}'.format(i+1), nn.Linear(layers[i], layers[i+1]))

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
