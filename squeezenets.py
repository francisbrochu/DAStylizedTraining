import torch
import torch.nn as nn
import torchvision


# for dog breed identification
class DBISqueezeNet(nn.Module):

    def __init__(self):
        super(DBISqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, 120, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# for Dogs vs Cats
class DCSqueezeNet(nn.Module):

    def __init__(self):
        super(DCSqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, 2, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# for dice
class DiceSqueezeeNet(nn.Module):

    def __init__(self):
        super(DiceSqueezeeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, 6, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# for Food101
class Food101SqueezeNet(nn.Module):

    def __init__(self):
        super(Food101SqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, 101, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


def load_squeezenet_model(dataset_name):

    if dataset_name == "DBI":
        return DBISqueezeNet()

    elif dataset_name == "DogsCats":
        return DCSqueezeNet()

    elif dataset_name == "Dice":
        return DiceSqueezeeNet()

    else:
        return Food101SqueezeNet()
