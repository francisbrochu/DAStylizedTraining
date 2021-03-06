import torch
import torchvision
import torch.utils.data
import numpy as np


def base_tagger(target):
    return target, 0


def style_tagger(target):
    return target, 1


def load_dataset(dataset_name, batch_size=32, num_workers=3):

    if dataset_name == "DBI":
        dataset_path = "dog-breed-identification/"

        train_base_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.5, 1.0)),
                                                 torchvision.transforms.RandomCrop(224, pad_if_needed=True)
                                                 ]),
            torchvision.transforms.ToTensor()
        ])

        train_style_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.5, 1.0))
                                                 ]),
            torchvision.transforms.ToTensor()
        ])
    elif dataset_name == "DogsCats":
        dataset_path = "dogs-vs-cats/"

        train_base_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.33, 1.0)),
                                                 torchvision.transforms.RandomCrop(224, pad_if_needed=True)
                                                 ]),
            torchvision.transforms.ToTensor()
        ])

        train_style_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.5, 1.0))
                                                 ]),
            torchvision.transforms.ToTensor()
        ])
    elif dataset_name == "Dice":
        dataset_path = "dice/"

        train_base_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomVerticalFlip(0.1),
            torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.Compose([
                                                     torchvision.transforms.RandomRotation(45),
                                                     torchvision.transforms.CenterCrop(224)
                                                 ]),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.33, 1.0)),
                                                 torchvision.transforms.RandomCrop(224)
                                                 ]),
            torchvision.transforms.ToTensor()
        ])

        train_style_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomVerticalFlip(0.1),
            torchvision.transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.5, 1.0))
                                                 ]),
            torchvision.transforms.ToTensor()
        ])
    else:  # food 101
        dataset_path = "food101/"

        train_base_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.5, 1.0))
                                                 ]),
            torchvision.transforms.ToTensor()
        ])

        train_style_transformer = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomChoice([torchvision.transforms.Resize((224, 224)),
                                                 torchvision.transforms.RandomResizedCrop(224,
                                                                                          scale=(0.5, 1.0))
                                                 ]),
            torchvision.transforms.ToTensor()
        ])

    valid_base_transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    valid_style_transformer = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    basepath = "../data/" + dataset_path
    stylepath = "../data/stylized/" + dataset_path

    base_train_dataset = torchvision.datasets.ImageFolder(basepath + "train/", transform=train_base_transformer,
                                                          target_transform=base_tagger)
    base_valid_dataset = torchvision.datasets.ImageFolder(basepath + "val/", transform=valid_base_transformer,
                                                          target_transform=base_tagger)

    style_train_dataset = torchvision.datasets.ImageFolder(stylepath + "train/", transform=train_style_transformer,
                                                           target_transform=style_tagger)
    style_valid_dataset = torchvision.datasets.ImageFolder(stylepath + "val/", transform=valid_style_transformer,
                                                           target_transform=style_tagger)

    train_dataset = torch.utils.data.ConcatDataset([base_train_dataset, style_train_dataset])
    valid_dataset = torch.utils.data.ConcatDataset([base_valid_dataset, style_valid_dataset])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    return train_loader, valid_loader
