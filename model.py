########################################################################
#                                                                      #
# This code is adapted from yingweiy's excellent GitHub repo:          #
#   https://github.com/yingweiy/drlnd_project1_navigation.git          #
#                                                                      #
########################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_UNITS = [128, 256, 256]
FC1_SIZE     = 1024

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed  = torch.manual_seed(seed)
        
        self.conv1 = nn.Conv3d(              3, HIDDEN_UNITS[0], kernel_size=(1,3,3), stride=(1,3,3))
        self.bn1   = nn.BatchNorm3d(HIDDEN_UNITS[0])
        self.conv2 = nn.Conv3d(HIDDEN_UNITS[0], HIDDEN_UNITS[1], kernel_size=(1,3,3), stride=(1,3,3))
        self.bn2   = nn.BatchNorm3d(HIDDEN_UNITS[1])
        self.conv3 = nn.Conv3d(HIDDEN_UNITS[1], HIDDEN_UNITS[2], kernel_size=(4,3,3), stride=(1,3,3))
        self.bn3   = nn.BatchNorm3d(HIDDEN_UNITS[2])

        cnn_out_size = self._get_cnn_out_size(state_size)
        self.fc1   = nn.Linear(cnn_out_size, FC1_SIZE)
        self.fc2   = nn.Linear(FC1_SIZE, action_size)

    def forward(self, state):
        x = self._cnn(state)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _get_cnn_out_size(self, shape):
        x = torch.rand(shape)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        return n_size

    def _cnn(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x
