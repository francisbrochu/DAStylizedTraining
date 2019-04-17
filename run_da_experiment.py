import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np
import torch.utils.data
from deeputils import generate_parameter_lists, read_da_config
from daloaders import load_dataset
import time
from daresnets import load_resnet_model
import os
import sys
from sklearn.metrics import accuracy_score

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# experiment parameters

if len(sys.argv) != 2:
    raise RuntimeError("Pass a config file.")
else:
    fname = sys.argv[1]

conf = read_da_config(fname)

dataset = conf["dataset"]  # DBI, DogsCats, Dice, Food101
model_type = conf["model_type"]  # resnet or squeezenet
id = conf["id"]

early_stopping = conf["early_stopping"]
patience = conf["patience"]

batch_size = conf["batch_size"]
learning_rate = conf["learning_rate"]
classif_lr = conf["classif_lr"]
weight_decay = conf["weight_decay"]
n_epochs = conf["n_epochs"]
num_workers = conf["n_workers"]
lambda_param = conf["lambda_param"]

epochs_list = conf["epochs_list"]

train_loader, validation_loader = load_dataset(dataset, batch_size=batch_size, num_workers=num_workers)

# define model
model = load_resnet_model(dataset, lambda_param)
model.cuda()

# define optimizer and loss
# generate the parameter lists first
convolutions, linear = generate_parameter_lists(model, model_type)
optimizer = torch.optim.Adam([{"params": convolutions},
                              {"params": linear, "lr": classif_lr}], lr=learning_rate, weight_decay=weight_decay)
criterion_classif = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()

# scheduling options
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epochs_list, gamma=conf["gamma"])

# define early stopping memory and epoch histories
epoch_counter = 0
minimum_error = 1.0
end_epoch = n_epochs

epoch_history = [[], [], [], []]
end_by_earlystop = False
model_i = 0

logfile = open("{}_{}_da_{}.log".format(dataset, model_type, id), "w")

starttime_string = "Starting training at {}".format(time.strftime("%H:%M:%S (%d %b)"))
info_string = "Starting DA experiment #{} on dataset {}, using {} architecture".format(id, dataset,
                                                                                       model_type)
print(info_string)
print(starttime_string)
logfile.write(info_string + "\n")
logfile.write(starttime_string + "\n")

# train
for i in range(n_epochs):
    start_time = time.time()

    # training epoch
    model.train()

    if scheduler is not None:
        scheduler.step()

    for j, batch in enumerate(train_loader):
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

    # evaluate on training set
    model.eval()
    train_classif_error = []
    train_domain_error = []

    for j, batch in enumerate(train_loader):
        inputs, targets = batch

        ctargets = targets[0]
        dtargets = targets[1]

        inputs = inputs.cuda()
        ctargets = ctargets.cuda()
        dtargets = dtargets.cuda()

        predictions_classif, predictions_domain = model(inputs)

        predictions_classif = predictions_classif.max(dim=1)[1]
        predictions_domain = predictions_domain.max(dim=1)[1]

        train_classif_error.append(1. - accuracy_score(ctargets.cpu(), predictions_classif.cpu()))
        train_domain_error.append(1. - accuracy_score(dtargets.cpu(), predictions_domain.cpu()))

    # evaluate on validation set
    val_classif_error = []
    val_domain_error = []

    for j, batch in enumerate(train_loader):
        inputs, targets = batch

        ctargets = targets[0]
        dtargets = targets[1]

        inputs = inputs.cuda()
        ctargets = ctargets.cuda()
        dtargets = dtargets.cuda()

        predictions_classif, predictions_domain = model(inputs)

        predictions_classif = predictions_classif.max(dim=1)[1]
        predictions_domain = predictions_domain.max(dim=1)[1]

        val_classif_error.append(1. - accuracy_score(ctargets.cpu(), predictions_classif.cpu()))
        val_domain_error.append(1. - accuracy_score(dtargets.cpu(), predictions_domain.cpu()))

    end_time = time.time()

    avg_train_classif_error = np.mean(train_classif_error)
    avg_valid_classif_error = np.mean(val_classif_error)
    avg_train_domain_error = np.mean(train_domain_error)
    avg_valid_domain_error = np.mean(val_domain_error)

    epoch_history[0].append(avg_train_classif_error)
    epoch_history[1].append(avg_train_domain_error)
    epoch_history[2].append(avg_valid_classif_error)
    epoch_history[3].append(avg_valid_domain_error)

    # print epoch info
    epoch_string = "[-] Epoch {} in {} seconds : " \
                   "Training Error for Classification = {}, " \
                   "Training Error for Domain = {}; " \
                   "Validation Error for Classification = {}, " \
                   "Validation Error for Domain = {} [-]".format(i + 1,
                                                                 round(end_time - start_time, 2),
                                                                 round(avg_train_classif_error * 100, 2),
                                                                 round(avg_train_domain_error * 100, 2),
                                                                 round(avg_valid_classif_error * 100, 2),
                                                                 round(avg_valid_domain_error * 100, 2))
    print(epoch_string)
    logfile.write(epoch_string + "\n")

    # early stopping
    if early_stopping:
        if avg_valid_classif_error < minimum_error:
            epoch_counter = 0
            minimum_error = avg_valid_classif_error
            model_i = i + 1
            torch.save(model, "./best_ckpt_{}_{}_da_{}.p".format(dataset, model_type, id))
        else:
            epoch_counter += 1

        if epoch_counter >= patience:
            earlystop_string = "Stopping early at epoch {}, using model of epoch {}".format(i + 1, model_i)
            print(earlystop_string)
            logfile.write(earlystop_string + "\n")

            model = torch.load("./best_ckpt_{}_{}_da_{}.p".format(dataset, model_type, id))
            end_by_earlystop = True
            break

if early_stopping:
    if (not end_by_earlystop) and (epoch_history[3][-1] > minimum_error):
        # load stored best model if it is better than the last scheduled epoch
        torch.load("./best_ckpt_{}_{}_da_{}.p".format(dataset, model_type, id))
        chosenmodel_string = "Loading best model (epoch {}) on validation as final model.".format(model_i)
        print(chosenmodel_string)
        logfile.write(chosenmodel_string + "\n")

    # clear checkpoint
    os.remove("./best_ckpt_{}_{}_da_{}.p".format(dataset, model_type, id))

# save model
torch.save(model, "./finalModel_{}_{}_da_{}.p".format(dataset, model_type, id))

endtime_string = "Experiment ended at {}".format(time.strftime("%H:%M:%S (%d %b)"))
print(endtime_string)
logfile.write(endtime_string)

logfile.close()
