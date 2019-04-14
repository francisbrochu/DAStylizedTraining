import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from grl import GradientReversalLayer, LambdaLayer


# for dog breed identification
class DBIDAResNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(DBIDAResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 120)

        self.grl = GradientReversalLayer()
        self.ll = LambdaLayer(lambda_param=lambda_param)
        self.domainfc = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.ll(domain_output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


# for Dogs vs Cats
class DCDAResNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(DCDAResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 2)

        self.grl = GradientReversalLayer()
        self.ll = LambdaLayer(lambda_param=lambda_param)
        self.domainfc = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.ll(domain_output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


# for dice
class DiceDAResNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(DiceDAResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 6)

        self.grl = GradientReversalLayer()
        self.ll = LambdaLayer(lambda_param=lambda_param)
        self.domainfc = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.ll(domain_output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


# for Food101
class Food101DAResNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(Food101DAResNet, self).__init__()

        self.model = torchvision.models.resnet34(pretrained=True)
        input_fc_dim = self.model.fc.in_features
        self.model.fc = nn.Linear(input_fc_dim, 101)

        self.grl = GradientReversalLayer()
        self.ll = LambdaLayer(lambda_param=lambda_param)
        self.domainfc = nn.Linear(input_fc_dim, 2)

    def forward(self, x):
        output = self.model.conv1(x)
        output = self.model.bn1(output)
        output = self.model.relu(output)
        output = self.model.maxpool(output)

        output = self.model.layer1(output)
        output = self.model.layer2(output)
        output = self.model.layer3(output)
        output = self.model.layer4(output)

        output = self.model.avgpool(output)
        output = output.view(output.size(0), -1)

        classif_output = self.model.fc(output)

        domain_output = self.grl(output)
        domain_output = self.ll(domain_output)
        domain_output = self.domainfc(domain_output)

        return classif_output, domain_output


def load_resnet_model(dataset_name, lambda_param=0.1):

    if dataset_name == "DBI":
        return DBIDAResNet(lambda_param)

    elif dataset_name == "DogsCats":
        return DCDAResNet(lambda_param)

    elif dataset_name == "Dice":
        return DiceDAResNet(lambda_param)

    else:
        return Food101DAResNet(lambda_param)
