import torch
import torch.nn as nn
import torchvision
from grl import LambdaLayer,


# for dog breed identification
class DBIDASqueezeNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(DBIDASqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 120
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        domain_final_conv = nn.Conv2d(input_fc_dim, 2, kernel_size=(1, 1), stride=(1, 1))
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Dropout(p=0.5),
            domain_final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            LambdaLayer(lambda_param=lambda_param)
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


# for Dogs vs Cats
class DCDASqueezeNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(DCDASqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 2
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        domain_final_conv = nn.Conv2d(input_fc_dim, 2, kernel_size=(1, 1), stride=(1, 1))
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Dropout(p=0.5),
            domain_final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            LambdaLayer(lambda_param=lambda_param)
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


# for dice
class DiceDASqueezeNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(DiceDASqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 6
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        domain_final_conv = nn.Conv2d(input_fc_dim, 2, kernel_size=(1, 1), stride=(1, 1))
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Dropout(p=0.5),
            domain_final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            LambdaLayer(lambda_param=lambda_param)
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


# for Food101
class Food101DASqueezeNet(nn.Module):

    def __init__(self, lambda_param=0.1):
        super(Food101DASqueezeNet, self).__init__()

        self.model = torchvision.models.squeezenet1_1(pretrained=True)
        self.model.num_classes = 101
        input_fc_dim = self.model.classifier[1].in_channels
        self.model.classifier[1] = nn.Conv2d(input_fc_dim, self.model.num_classes, kernel_size=(1, 1), stride=(1, 1))

        domain_final_conv = nn.Conv2d(input_fc_dim, 2, kernel_size=(1, 1), stride=(1, 1))
        self.domain_classifier = nn.Sequential(
            GradientReversalLayer(),
            nn.Dropout(p=0.5),
            domain_final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            LambdaLayer(lambda_param=lambda_param)
        )

    def forward(self, x):
        output = self.model.features(x)

        classif_output = self.model.classifier(output)
        domain_output = self.domain_classifier(output)

        return classif_output, domain_output


def load_squeezenet_model(dataset_name, lambda_param=0.1):

    if dataset_name == "DBI":
        return DBIDASqueezeNet(lambda_param=lambda_param)

    elif dataset_name == "DogsCats":
        return DCDASqueezeNet(lambda_param=lambda_param)

    elif dataset_name == "Dice":
        return DiceDASqueezeNet(lambda_param=lambda_param)

    else:
        return Food101DASqueezeNet(lambda_param=lambda_param)
