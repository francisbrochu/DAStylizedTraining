import torch
import torch.nn as nn
import torchvision


# for dog breed identification
class DBIVGG(nn.Module):

    def __init__(self):
        super(DBIVGG, self).__init__()

        self.model = torchvision.models.vgg16_bn(pretrained=True)
        input_fc_dim = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(input_fc_dim, 120)

    def forward(self, x):
        return self.model(x)


# for Dogs vs Cats
class DCVGG(nn.Module):

    def __init__(self):
        super(DCVGG, self).__init__()

        self.model = torchvision.models.vgg16_bn(pretrained=True)
        input_fc_dim = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        return self.model(x)


# for dice
class DiceVGG(nn.Module):

    def __init__(self):
        super(DiceVGG, self).__init__()

        self.model = torchvision.models.vgg16_bn(pretrained=True)
        input_fc_dim = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(input_fc_dim, 6)

    def forward(self, x):
        return self.model(x)


# for Food101
class Food101VGG(nn.Module):

    def __init__(self):
        super(Food101VGG, self).__init__()

        self.model = torchvision.models.vgg16_bn(pretrained=True)
        input_fc_dim = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(input_fc_dim, 101)

    def forward(self, x):
        return self.model(x)


def load_vgg_model(dataset_name):

    if dataset_name == "DBI":
        return DBIVGG()

    elif dataset_name == "DogsCats":
        return DCVGG()

    elif dataset_name == "Dice":
        return DiceVGG()

    else:
        return Food101VGG()
