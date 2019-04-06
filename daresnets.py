import torch
import torch.nn as nn
import torchvision
from grl import GradientReversalLayer


# for dog breed identification
class DBIResNet(nn.Module):

    def __init__(self, c=1.0):
        super(DBIResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 120)

        self.grl = GradientReversalLayer(c=c)
        self.domainfc = nn.Linear(input_fc_dim, 1)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


# for Dogs vs Cats
class DCResNet(nn.Module):

    def __init__(self, c=1.0):
        super(DCResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 2)

        self.grl = GradientReversalLayer(c=c)
        self.domainfc = nn.Linear(input_fc_dim, 1)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


# for dice
class DiceResNet(nn.Module):

    def __init__(self, c=1.0):
        super(DiceResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 6)

        self.grl = GradientReversalLayer(c=c)
        self.domainfc = nn.Linear(input_fc_dim, 1)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


# for Food101
class Food101ResNet(nn.Module):

    def __init__(self, c=1.0):
        super(Food101ResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 101)

        self.grl = GradientReversalLayer(c=c)
        self.domainfc = nn.Linear(input_fc_dim, 1)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


def load_resnet_model(dataset_name, c=1.0):

    if dataset_name == "DBI":
        return DBIResNet(c=c)

    elif dataset_name == "DogsCats":
        return DCResNet(c=c)

    elif dataset_name == "Dice":
        return DiceResNet(c=c)

    else:
        return Food101ResNet(c=c)
