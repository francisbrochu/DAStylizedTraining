import torch
import torch.nn as nn
import torchvision


# for dog breed identification
class DBIResNet(nn.Module):

    def __init__(self):
        super(DBIResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 120)

    def forward(self, x):
        return self.model(x)


# for Dogs vs Cats
class DCResNet(nn.Module):

    def __init__(self):
        super(DCResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        return self.model(x)


# for dice
class DiceResNet(nn.Module):

    def __init__(self):
        super(DiceResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 8)

    def forward(self, x):
        return self.model(x)


# for Food101
class Food101ResNet(nn.Module):

    def __init__(self):
        super(Food101ResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 101)

    def forward(self, x):
        return self.model(x)


def load_resnet_model(dataset_name):

    if dataset_name == "DBI":
        return DBIResNet()

    elif dataset_name == "DogsCats":
        return DCResNet()

    elif dataset_name == "Dice":
        return DiceResNet()

    else:
        return Food101ResNet()
