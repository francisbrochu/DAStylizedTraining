import torch.nn as nn


class GradientReversalLayer(nn.Module):

    def forward(self, *input):
        return input
