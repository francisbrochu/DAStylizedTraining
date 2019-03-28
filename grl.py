import torch.nn as nn


class GradientReversalLayer(nn.Module):

    def forward(self, x):
        h = x.register_hook(lambda grad: grad * -1)
        return x
