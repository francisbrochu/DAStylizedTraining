import torch.nn as nn


class GradientReversalLayer(nn.Module):

    def __init__(self, c = 1):
        super(GradientReversalLayer, self).__init__()

        self.c = -1 * c

    def forward(self, x):
        h = x.register_hook(lambda grad: grad * self.c)
        return x
