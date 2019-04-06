import torch
import torchvision
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import json


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
        error_history.append(1 - accuracy_score(targets.cpu(), predictions.cpu()))

    return loss_history, error_history


def generate_parameter_lists(model, model_type):
    convs = []
    lin = []
    if model_type == "resnet":
        for name, param in model.model.named_parameters():
            if name in {"model.fc.weight", "model.fc.bias", "domainfc.weight", "domainfc.bias"}:
                lin.append(param)
            else:
                convs.append(param)
    else:  # densenet
        for name, param in model.model.named_parameters():
            if name in {"model.classifier.1.weight", "model.classifier.1.bias",
                        "final_conv_domain.weight", "final_conv_domain.bias"}:
                lin.append(param)
            else:
                convs.append(param)

    return convs, lin


def plot_history(history, end_epoch, dataset, model_type, experiment_type, id):
    epochs = list(range(end_epoch))

    fig = plt.figure()

    subfig1 = fig.add_subplot(2, 1, 1)
    subfig1.plot(epochs, history[0], color="blue", label="Train")
    subfig1.plot(epochs, history[1], color="orange", label="Validation")
    subfig1.set_title("Losses by epoch")
    subfig1.legend()

    subfig2 = fig.add_subplot(2, 1, 2)
    subfig2.plot(epochs, history[2], color="blue", label="Train")
    subfig2.plot(epochs, history[3], color="orange", label="Validation")
    subfig2.set_title("Error by epoch")
    subfig2.legend()

    plt.savefig("history_{}_{}_{}_{}.png".format(dataset, model_type, experiment_type, id))


def read_config(filename):
    conf = {}

    cfg = configparser.ConfigParser()
    cfg.read(filename)

    conf["dataset"] = cfg["Experiment"]["dataset"]
    conf["model_type"] = cfg["Experiment"]["model_type"]
    conf["experiment_type"] = cfg["Experiment"]["experiment_type"]
    conf["id"] = cfg["Experiment"]["id"]

    conf["early_stopping"] = cfg.getboolean("Parameters", "early_stopping")
    conf["patience"] = int(cfg["Parameters"]["patience"])
    conf["batch_size"] = int(cfg["Parameters"]["batch_size"])
    conf["learning_rate"] = float(cfg["Parameters"]["learning_rate"])
    conf["classif_lr"] = float(cfg["Parameters"]["classif_lr"])
    conf["weight_decay"] = float(cfg["Parameters"]["weight_decay"])
    conf["n_epochs"] = int(cfg["Parameters"]["n_epochs"])
    conf["n_workers"] = int(cfg["Parameters"]["n_workers"])
    conf["gamma"] = float(cfg["Parameters"]["gamma"])

    epochs_list = []
    if cfg.getboolean("Parameters", "scheduler"):
        epochs_list = json.loads(cfg.get("Scheduler", "epochs_list"))

    conf["epochs_list"] = epochs_list

    return conf


def read_da_config(filename):
    conf = {}

    cfg = configparser.ConfigParser()
    cfg.read(filename)

    conf["dataset"] = cfg["Experiment"]["dataset"]
    conf["model_type"] = cfg["Experiment"]["model_type"]
    conf["id"] = cfg["Experiment"]["id"]

    conf["early_stopping"] = cfg.getboolean("Parameters", "early_stopping")
    conf["patience"] = int(cfg["Parameters"]["patience"])
    conf["batch_size"] = int(cfg["Parameters"]["batch_size"])
    conf["learning_rate"] = float(cfg["Parameters"]["learning_rate"])
    conf["classif_lr"] = float(cfg["Parameters"]["classif_lr"])
    conf["weight_decay"] = float(cfg["Parameters"]["weight_decay"])
    conf["n_epochs"] = int(cfg["Parameters"]["n_epochs"])
    conf["n_workers"] = int(cfg["Parameters"]["n_workers"])
    conf["gamma"] = float(cfg["Parameters"]["gamma"])
    conf["c_param"] = float(cfg["Parameters"]["c_param"])

    epochs_list = []
    if cfg.getboolean("Parameters", "scheduler"):
        epochs_list = json.loads(cfg.get("Scheduler", "epochs_list"))

    conf["epochs_list"] = epochs_list

    return conf


def training_da_epoch(model, train_loader, optimizer, criterion_classif, criterion_domain, scheduler=None):
    classif_loss_history = []
    domain_loss_history = []

    model.train()

    if scheduler is not None:
        scheduler.step()

    for i, batch in enumerate(train_loader):
        inputs, targets = batch

        ctargets = targets[0]
        dtargets = targets[1]

        inputs = inputs.cuda()
        ctargets = ctargets.cuda()
        dtargets = dtargets.cuda()

        optimizer.zero_grad()

        predictions_class, predictions_domain = model(inputs)
        classif_loss = criterion_classif(predictions_class, ctargets)
        domain_loss = criterion_domain(predictions_domain, dtargets)

        loss = classif_loss + domain_loss

        loss.backward()
        optimizer.step()

        classif_loss_history.append(classif_loss.item())
        domain_loss_history.append(domain_loss.item())

    return classif_loss_history, domain_loss_history


def evaluate_da(model, loader):
    error_classif_history = []
    error_domain_history = []

    model.eval()

    for i, batch in enumerate(loader):
        inputs, targets = batch

        ctargets = targets[0]
        dtargets = targets[1]

        inputs = inputs.cuda()
        ctargets = ctargets.cuda()
        dtargets = dtargets.cuda()

        predictions_classif, predictions_domain = model(inputs)

        predictions_classif = predictions_classif.max(dim=1)[1]
        predictions_domain = binarize_predictions(predictions_domain)

        error_classif_history.append(1. - accuracy_score(ctargets.cpu(), predictions_classif.cpu()))
        error_domain_history.append(1. - accuracy_score(dtargets.cpu(), predictions_domain))

    return error_classif_history, error_domain_history


def binarize_predictions(predictions):
    pred = []
    for val in predictions:
        if val >= 0.5:
            pred.append(1)
        else:
            pred.append(0)

    return np.asarray(pred)
