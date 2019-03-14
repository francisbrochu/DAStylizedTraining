import torch
import torch.nn as nn
import torchvision


# for dog breed identification
class DBIDenseNet(nn.Module):

    def __init__(self):
        super(DBIDenseNet, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier = nn.Linear(input_fc_dim, 120)

    def forward(self, x):
        return self.model(x)


# for Dogs vs Cats
class DCDenseNet(nn.Module):

    def __init__(self):
        super(DCDenseNet, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        return self.model(x)


# for dice
class DiceDenseNet(nn.Module):

    def __init__(self):
        super(DiceDenseNet, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier = nn.Linear(input_fc_dim, 6)

    def forward(self, x):
        return self.model(x)


# for Food101
class Food101DenseNet(nn.Module):

    def __init__(self):
        super(Food101DenseNet, self).__init__()

        self.model = torchvision.models.densenet121(pretrained=True)
        input_fc_dim = self.model.classifier.in_features
        self.model.classifier = nn.Linear(input_fc_dim, 101)

    def forward(self, x):
        return self.model(x)


def load_densenet_model(dataset_name):

    if dataset_name == "DBI":
        return DBIDenseNet()

    elif dataset_name == "DogsCats":
        return DCDenseNet()

    elif dataset_name == "Dice":
        return DiceDenseNet()

    else:
        return Food101DenseNet()
