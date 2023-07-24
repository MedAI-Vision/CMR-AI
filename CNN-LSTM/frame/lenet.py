import torch.nn as nn
import torch.nn.functional as F


################################################################################
# Simple CNN Models
################################################################################

class LeNetFrameEncoder(nn.Module):

    def __init__(self, input_shape=(1, 32, 32), kernel_size=5, output_size=84):
        super(LeNetFrameEncoder, self).__init__()

        n_channels, width, height = input_shape
        conv_output     = self.calculate_conv_output(input_shape, kernel_size)
        self.conv1      = nn.Conv2d(n_channels, 6, kernel_size)
        self.conv2      = nn.Conv2d(6, 16, kernel_size)
        self.fc1        = nn.Linear(conv_output, 120)
        self.fc2        = nn.Linear(120, output_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        return out

    def calculate_conv_output(self, input_shape, kernel_size):
        output1 = (6, input_shape[1]-kernel_size+1, input_shape[2]-kernel_size+1)
        output2 = (6, int(output1[1]/2), int(output1[2]/2))
        output3 = (16, output2[1]-kernel_size+1, output2[2]-kernel_size+1)
        output4 = (16, int(output3[1]/2), int(output3[2]/2))

        return output4[0]*output4[1]*output4[2]

