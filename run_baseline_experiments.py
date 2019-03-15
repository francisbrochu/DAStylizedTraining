import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from deeputils import training_epoch, evaluate, generate_parameter_lists, plot_history, read_config
from datasetloaders import load_base_dataset, load_style_dataset, load_mixed_dataset
import time
from resnets import load_resnet_model
from squeezenets import load_squeezenet_model
import os
import sys

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# experiment parameters

if len(sys.argv) != 2:
    raise RuntimeError("Pass a config file.")
else:
    fname = sys.argv[1]

conf = read_config(fname)

dataset = conf["dataset"]  # DBI, DogsCats, Dice, Food101
model_type = conf["model_type"]  # resnet or squeezenet
experiment_type = conf["experiment_type"]  # base, style, mixed
id = conf["id"]

early_stopping = conf["early_stopping"]
patience = conf["patience"]

batch_size = conf["batch_size"]
learning_rate = conf["learning_rate"]
classif_lr = conf["classif_lr"]
weight_decay = conf["weight_decay"]
n_epochs = conf["n_epochs"]
num_workers = conf["n_workers"]

epochs_list = conf["epochs_list"]

if experiment_type == "base":
    train_loader, validation_loader = load_base_dataset(dataset, batch_size=batch_size, num_workers=num_workers)
elif experiment_type == "style":
    train_loader, validation_loader = load_style_dataset(dataset, batch_size=batch_size, num_workers=num_workers)
else:
    train_loader, validation_loader = load_mixed_dataset(dataset, batch_size=batch_size, num_workers=num_workers)

# define model
if model_type == "resnet":
    model = load_resnet_model(dataset)
    model.cuda()
else:
    model = load_squeezenet_model(dataset)
    model.cuda()

# define optimizer and loss
# generate the parameter lists first
convolutions, linear = generate_parameter_lists(model, model_type)
optimizer = torch.optim.Adam([{"params": convolutions},
                              {"params": linear, "lr": classif_lr}], lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# scheduling options
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=epochs_list, gamma=conf["gamma"])

# define early stopping memory and epoch histories
epoch_counter = 0
minimum_error = 1.0
end_epoch = n_epochs

epoch_history = [[], [], [], []]
end_by_earlystop = False

# train
for i in range(n_epochs):
    start_time = time.time()

    # training epoch
    _ = training_epoch(model, train_loader, optimizer, criterion, scheduler)

    # evaluate on training set
    train_loss, train_error = evaluate(model, train_loader, criterion)

    # evaluate on validation set
    validation_loss, validation_error = evaluate(model, validation_loader, criterion)

    end_time = time.time()

    avg_train_loss, avg_train_error = np.mean(train_loss), np.mean(train_error)
    avg_valid_loss, avg_valid_error = np.mean(validation_loss), np.mean(validation_error)

    epoch_history[0].append(avg_train_loss)
    epoch_history[1].append(avg_valid_loss)
    epoch_history[2].append(avg_train_error)
    epoch_history[3].append(avg_valid_error)

    # print epoch info
    print("[-] Epoch {} in {} seconds : Training Loss = {}, Training Error = {}; "
          "Validation Loss = {}, Validation Error = {} [-]".format(i+1,
                                                               round(end_time - start_time, 2),
                                                               round(avg_train_loss, 4),
                                                               round(avg_train_error * 100, 2),
                                                               round(avg_valid_loss, 4),
                                                               round(avg_valid_error * 100, 2)))

    # early stopping
    if early_stopping:

        if avg_valid_error < minimum_error:
            epoch_counter = 0
            minimum_error = avg_valid_error
            torch.save(model, "./best_ckpt_{}_{}_{}_{}.p".format(dataset, model_type, experiment_type, id))
        else:
            epoch_counter += 1

        if epoch_counter >= patience:
            end_epoch = i + 1
            print("Stopping early at epoch {}".format(end_epoch + 1))
            model = torch.load("./best_ckpt_{}_{}_{}_{}.p".format(dataset, model_type, experiment_type, id))
            end_by_earlystop = True
            break

if early_stopping:
    if (not end_by_earlystop) and (epoch_history[3][-1] > minimum_error):
        torch.load("./best_ckpt_{}_{}_{}_{}.p".format(dataset, model_type, experiment_type, id))
        print("Loading best model on validation as final model.")

    # clear checkpoint
    os.remove("./best_ckpt_{}_{}_{}_{}.p".format(dataset, model_type, experiment_type, id))

# save model
torch.save(model, "./finalModel_{}_{}_{}_{}.p".format(dataset, model_type, experiment_type, id))

# plot history
plot_history(epoch_history, end_epoch, dataset, model_type, experiment_type)
