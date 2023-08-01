###################################################################################################
###  DISCLAIMER:                                                                                  #
###    I do not own any of the code in this file. This code is taken from the dain repository     #
###    of Github user Nikolaos Passalis (passalis): https://github.com/passalis/dain/tree/master  #
###    and is from their paper "Deep Adaptive Input Normalization for Price Forecasting using     #
###    Limit Order Book Data", published in the "IEEE Transactions on Neural Networks and         #
###    Learning Systems" journal in 2019.                                                         #
###                                                                                               #
###  The code has been modified to fit the needs of this project.                                 #
###                                                                                               #
###  Authors of paper:                                                                            #
###  * Passalis, Nikolaos                                                                         #
###  * Tefas, Anastasios                                                                          #
###  * Kanniainen, Juho                                                                           #
###  * Gabbouj, Moncef                                                                            #
###  * Iosifidis, Alexandros                                                                      #
####################################################################################################
import numpy as np
import torch
from torch.autograd import Variable
from tqdm.auto import tqdm
import torch.optim as optim
from src.lib.lob_loader import get_wf_lob_loaders
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score


def lob_epoch_train_one_epoch(
        model,
        train_loader,
        preprocess,
        model_optimizer,
        device,
    ):
    model.train()

    criterion = CrossEntropyLoss()
    train_loss, counter = 0, 0

    for (inputs, targets) in train_loader:
        # apply preprocesses to input
        if preprocess is not None:
            # Data is on form (N, T, D)
            inputs = torch.from_numpy(preprocess.transform(inputs.numpy()))

        model_optimizer.zero_grad()

        # move to GPU
        inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))
        targets = torch.squeeze(targets)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        model_optimizer.step()

        train_loss += loss.item()
        counter += 1.

    loss = (loss / counter).cpu().item()
    return loss


def lob_evaluator(model, loader, preprocess, device):
    model.eval()

    true_labels = []
    predicted_labels = []
    avg_val_loss, counter = 0, 0

    criterion = CrossEntropyLoss(reduction='mean')

    for (inputs, targets) in tqdm(loader):
        # apply preprocesses to input
        if preprocess is not None:
            # Data is on form (N, T, D)
            inputs = torch.from_numpy(preprocess.transform(inputs.numpy()))

        with torch.no_grad():
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            val_loss = criterion(outputs, torch.squeeze(targets))

            predicted_labels.append(predicted.cpu().numpy())
            true_labels.append(targets.cpu().data.numpy())
            avg_val_loss += val_loss.cpu().item()
            counter += 1.

    true_labels = np.squeeze(np.concatenate(true_labels))
    predicted_labels = np.squeeze(np.concatenate(predicted_labels))

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average=None)
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                           average='macro')
    kappa = cohen_kappa_score(true_labels, predicted_labels)
    avg_val_loss /= counter

    metrics = {}
    metrics['accuracy'] = np.sum(true_labels == predicted_labels) / len(true_labels)

    metrics['precision'], metrics['recall'], metrics['f1'] = precision, recall, f1

    metrics['precision_avg'], metrics['recall_avg'], metrics['f1_avg'] = precision_avg, recall_avg, f1_avg

    metrics['kappa'] = kappa

    metrics['val_loss'] = avg_val_loss

    return metrics
