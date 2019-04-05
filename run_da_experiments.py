import torch
import torch.nn as nn
import torchvision
import torch.optim
import numpy as np
import torch.utils.data
from deeputils import training_da_epoch, evaluate_da, generate_parameter_lists, plot_history, read_da_config
from daloaders import load_dataset
import time
from daresnets import load_resnet_model
from dasqueezenets import load_squeezenet_model
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
c_param = conf["c_param"]

epochs_list = conf["epochs_list"]

train_loader, validation_loader = load_dataset(dataset, batch_size=batch_size, num_workers=num_workers)

# define model
if model_type == "resnet":
    model = load_resnet_model(dataset, c_param)
    model.cuda()
else:
    model = load_squeezenet_model(dataset, c_param)
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

epoch_history = [[], [], [], [], [], [], [], []]
end_by_earlystop = False
model_i = 0

logfile = open("{}_{}_da_{}.log".format(dataset, model_type, id), "w")

starttime_string = "Starting training at {}".format(time.strftime("%H:%M:%S (%d %b)"))
info_string = "Starting experiment #{} on dataset {}, using {} architecture".format(id, dataset,
                                                                                    model_type)
print(info_string)
print(starttime_string)
logfile.write(info_string + "\n")
logfile.write(starttime_string + "\n")

# train
for i in range(n_epochs):
    start_time = time.time()

    # training epoch
    _, _ = training_da_epoch(model, train_loader, optimizer, criterion_classif, criterion_domain, scheduler)

    # evaluate on training set
    #train_classif_loss, train_domain_loss, train_classif_error, train_domain_error = evaluate_da(model, train_loader,
    train_classif_error, train_domain_error = evaluate_da(model, train_loader) # ,criterion_classif, criterion_domain)

    # evaluate on validation set
    #val_classif_loss, val_domain_loss, val_classif_error, val_domain_error = evaluate_da(model, validation_loader,
    val_classif_error, val_domain_error = evaluate_da(model, validation_loader) # ,criterion_classif, criterion_domain)

    end_time = time.time()

    #avg_train_classif_loss, avg_train_classif_error = np.mean(train_classif_loss), np.mean(train_classif_error)
    #avg_valid_classif_loss, avg_valid_classif_error = np.mean(val_classif_loss), np.mean(val_classif_error)
    #avg_train_domain_loss, avg_train_domain_error = np.mean(train_domain_loss), np.mean(train_domain_error)
    #avg_valid_domain_loss, avg_valid_domain_error = np.mean(val_domain_loss), np.mean(val_domain_error)
    avg_train_classif_error = np.mean(train_classif_error)
    avg_valid_classif_error = np.mean(val_classif_error)
    avg_train_domain_error = np.mean(train_domain_error)
    avg_valid_domain_error = np.mean(val_domain_error)

    #epoch_history[0].append(avg_train_classif_loss)
    #epoch_history[1].append(avg_train_domain_loss)
    #epoch_history[2].append(avg_valid_classif_loss)
    #epoch_history[3].append(avg_valid_domain_loss)
    epoch_history[4].append(avg_train_classif_error)
    epoch_history[5].append(avg_train_domain_error)
    epoch_history[6].append(avg_valid_classif_error)
    epoch_history[7].append(avg_valid_domain_error)

    # print epoch info
    # Training Loss = {}, " \
    # "Validation Loss = {}, " \
    epoch_string = "[-] Epoch {} in {} seconds : " \
                   "Training Error for Classification = {}, " \
                   "Training Error for Domain = {}; " \
                   "Validation Error for Classification = {}, " \
                   "Validation Error for Domain = {} [-]".format(i + 1,
                                                                 round(end_time - start_time, 2),
                                                                 #round(avg_train_classif_loss, 4),
                                                                 round(avg_train_classif_error * 100, 2),
                                                                 round(avg_train_domain_error * 100, 2),
                                                                 #round(avg_valid_classif_loss, 4),
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
