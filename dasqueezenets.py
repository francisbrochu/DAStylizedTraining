import torch
import torch.nn as nn
import torchvision
from grl import GradientReversalLayer


# for dog breed identification
class DBISqueezeNet(nn.Module):

    def __init__(self, c=1.0):
        super(DBISqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 120
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.final_conv_domain = nn.Conv2d(512, 2, kernel_size=1)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(c=c),
            nn.Dropout(p=0.5),
            self.final_conv_domain,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


# for Dogs vs Cats
class DCSqueezeNet(nn.Module):

    def __init__(self, c=1.0):
        super(DCSqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 2
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.final_conv_domain = nn.Conv2d(512, 2, kernel_size=1)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(c=c),
            nn.Dropout(p=0.5),
            self.final_conv_domain,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


# for dice
class DiceSqueezeNet(nn.Module):

    def __init__(self, c=1.0):
        super(DiceSqueezeeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 6
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.final_conv_domain = nn.Conv2d(512, 2, kernel_size=1)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(c=c),
            nn.Dropout(p=0.5),
            self.final_conv_domain,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


# for Food101
class Food101SqueezeNet(nn.Module):

    def __init__(self, c=1.0):
        super(Food101SqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 101
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        self.final_conv_domain = nn.Conv2d(512, 2, kernel_size=1)
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(c=c),
            nn.Dropout(p=0.5),
            self.final_conv_domain,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


def load_squeezenet_model(dataset_name, c=1.0):

    if dataset_name == "DBI":
        return DBISqueezeNet(c=c)

    elif dataset_name == "DogsCats":
        return DCSqueezeNet(c=c)

    elif dataset_name == "Dice":
        return DiceSqueezeNet(c=c)

    else:
        return Food101SqueezeNet(c=c)