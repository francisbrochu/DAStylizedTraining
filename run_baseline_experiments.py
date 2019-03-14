import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from deeputils import training_epoch, evaluate, generate_parameter_lists, plot_history
from datasetloaders import load_base_dataset, load_style_dataset, load_mixed_dataset
import time
from resnets import load_resnet_model
from densenets import load_densenet_model
import os

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# experiment parameters

dataset = "DBI"  # DBI, DogsCats, Dice, Food101
model_type = "resnet"  # resnet or densenet

experiment_type = "base"  # base, style, mixed

early_stopping = False
patience = 10

batch_size = 16
learning_rate = 1e-4
classif_lr = 1e-3
weight_decay = 1e-5
n_epochs = 50

epochs_list = []

if experiment_type == "base":
    train_loader, validation_loader = load_base_dataset(dataset, batch_size=batch_size)
elif experiment_type == "style":
    train_loader, validation_loader = load_style_dataset(dataset, batch_size=batch_size)
else:
    train_loader, validation_loader = load_mixed_dataset(dataset, batch_size=batch_size)

# define model
if model_type == "resnet":
    model = load_resnet_model(dataset)
    model.cuda()
else:
    model = load_densenet_model(dataset)
    model.cuda()

# define optimizer and loss
# generate the parameter lists first
convolutions, linear = generate_parameter_lists(model, model_type)
optimizer = torch.optim.Adam([{"params": convolutions},
                              {"params": linear, "lr": classif_lr}], lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

# scheduling options
scheduler = None

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
    print("[-] Epoch {} in {} seconds : Training Loss".format(i+1, round(end_time - start_time, 2)))

    # early stopping
    if early_stopping:

        if avg_valid_error < minimum_error:
            epoch_counter = 0
            minimum_error = avg_valid_error
            torch.save(model, "./best_ckpt_{}_{}_{}.p".format(dataset, model_type, experiment_type))
        else:
            epoch_counter += 1

        if epoch_counter >= patience:
            end_epoch = i + 1
            print("Stopping early at epoch {}".format(end_epoch + 1))
            model = torch.load("./best_ckpt_{}_{}_{}.p".format(dataset, model_type, experiment_type))
            end_by_earlystop = True
            break

if early_stopping:
    if (not end_by_earlystop) and (epoch_history[3][-1] > minimum_error):
        torch.load("./best_ckpt_{}_{}_{}.p".format(dataset, model_type, experiment_type))
        print("Loading best model on validation as final model.")

# clear checkpoint
os.remove("./best_ckpt_{}_{}_{}.p".format(dataset, model_type, experiment_type))

# save model
torch.save(model, "./bestModel_{}_{}_{}.p".format(dataset, model_type, experiment_type))

# plot history
plot_history(epoch_history, end_epoch, dataset, model_type, experiment_type)
