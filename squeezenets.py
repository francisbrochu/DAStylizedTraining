import torch
import torch.nn as nn
import torchvision


# for dog breed identification
class DBISqueezeNet(nn.Module):

    def __init__(self):
        super(DBISqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 120
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# for Dogs vs Cats
class DCSqueezeNet(nn.Module):

    def __init__(self):
        super(DCSqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 2
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# for dice
class DiceSqueezeNet(nn.Module):

    def __init__(self):
        super(DiceSqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 6
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


# for Food101
class Food101SqueezeNet(nn.Module):

    def __init__(self):
        super(Food101SqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 101
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)


def load_squeezenet_model(dataset_name):

    if dataset_name == "DBI":
        return DBISqueezeNet()

    elif dataset_name == "DogsCats":
        return DCSqueezeNet()

    elif dataset_name == "Dice":
        return DiceSqueezeNet()

    else:
        return Food101SqueezeNet()
