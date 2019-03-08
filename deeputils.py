import torch
import torchvision
import torch.nn
import numpy as np
from sklearn.metrics import accuracy_score


def training_epoch(model, train_loader, optimizer, criterion, scheduler=None):
    loss_history = []

    model.train()

    if scheduler is not None:
        scheduler.step()

    for i, batch in enumerate(train_loader):
         inputs, targets = batch

         inputs = inputs.cuda()
         targets = targets.cuda()

         optimizer.zero_grad()

         predictions = model(inputs)
         loss = criterion(predictions, targets)

         loss.backward()
         optimizer.step()

         loss_history.append(loss.item())

    return loss_history


def evaluate(model, loader, criterion):
    loss_history = []
    error_history = []

    model.eval()

    for i, batch in enumerate(loader):
        inputs, targets = batch

        inputs = inputs.cuda()
        targets = targets.cuda()

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        predictions = predictions.max(dim=1)[1]

        loss_history.append(loss.item())
        error_history.append(1 - accuracy_score(targets, predictions))

    return loss_history, error_history
