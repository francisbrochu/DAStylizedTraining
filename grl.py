import torch.nn as nn
from torch.autograd import Function


class LambdaLayer(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(LambdaLayer, self).__init__()
        self.lambda_param = lambda_param

    def forward(self, x):
        h = x.register_hook(lambda grad: grad * self.lambda_param)
        return x


class ReverseLayerF(Function):
    """
    Credit goes to Github user fungtion for this implementation of the gradient reversal.
    fbrochu mofified his function slightly
    """
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()

        return output, None
