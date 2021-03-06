import torch
import torchvision
import numpy as np
import torch.utils.data
from deeputils import TopKAccuracy

dataset = "Dice"
k = 1
base_dataset = True  # True for base, False for stylized

model_file = "da_experiments/finalModel_Dice_resnet_da_10.p"

test_loader = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

testpath = "../data/"

if not base_dataset:
    testpath = testpath + "stylized/"

if dataset == "DBI":
    testpath = testpath + "dog-breed-identification/test/"
elif dataset == "DogsCats":
    testpath = testpath + "dogs-vs-cats/test/"
elif dataset == "Dice":
    testpath = testpath + "dice/test/"
else:  # food101
    testpath = testpath + "food101/test/"

model = torch.load(model_file)  # since it was saved from cuda, it should load in cuda

model.eval()

test_dataset = torchvision.datasets.ImageFolder(testpath, transform=test_loader)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=3)

batch_accuracies = []

with torch.no_grad():
    print("Model loaded and ready, predicting on the test set.")

    for i, batch in enumerate(test_loader):
        images, targets = batch

        images = images.cuda()
        targets = targets.cuda()

        predictions = model(images)

        batch_accuracies.append(TopKAccuracy(predictions, targets, k=k))

avg_acc = np.mean(batch_accuracies)

print("Accuracy on the test set is of {}".format(round(avg_acc * 100, 2)))




