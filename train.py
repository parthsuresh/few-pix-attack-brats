import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import os
import random

models_path = "./bn_models/"


class Train_Model:
    def __init__(self, device, model):
        self.device = device
        self.model = model
        self.model.to(device)

        # initialize optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        self.file = open(models_path + "log.txt", "a")

    def train_batch(self, x, labels):
        self.optimizer.zero_grad()
        preds = self.model(x)
        loss = self.criterion(preds, labels.long()) / labels.shape[0]
        _, pred_labels = preds.max(1, keepdim=True)
        accuracy = (
            pred_labels.eq(labels.view_as(pred_labels)).sum().item() / labels.shape[0]
        )
        loss.backward()
        self.optimizer.step()
        return loss.item(), accuracy

    def valid_batch(self, x, labels):
        preds = self.model(x)
        loss = self.criterion(preds, labels.long())
        _, pred_labels = preds.max(1, keepdim=True)
        accuracy = (
            pred_labels.eq(labels.view_as(pred_labels)).sum().item() / labels.shape[0]
        )
        return loss.item(), accuracy

    def train(self, train_dataloader, valid_dataloader, epochs):
        self.model.train()
        for epoch in range(1, epochs + 1):
            loss_sum = 0
            accuracy_sum = 0
            for i, data in tqdm(
                enumerate(train_dataloader, start=0), desc=f"Epoch {epoch}"
            ):
                images, labels = data["X"], data["y"]
                images, labels = images.to(self.device), labels.to(self.device)

                loss_batch, accuracy_batch = self.train_batch(images, labels)
                loss_sum += loss_batch
                accuracy_sum += accuracy_batch

            num_batch = len(train_dataloader)
            avg_loss = loss_sum / num_batch
            avg_accuracy = accuracy_sum / num_batch

            # print statistics
            self.file.write(
                f"Epoch: {epoch} | Train Loss : {avg_loss:.3f} | Train Accuracy : {avg_accuracy:.3f}\n"
            )
            print(
                f"Epoch: {epoch} | Train Loss : {avg_loss:.3f} | Train Accuracy : {avg_accuracy:.3f}"
            )

            self.valid(valid_dataloader)

            # save model
            model_file_name = models_path + "model_epoch_" + str(epoch) + ".pth"
            torch.save(self.model.state_dict(), model_file_name)

        self.file.close()

    def valid(self, valid_dataloader):
        self.model.eval()
        loss_sum = 0
        accuracy_sum = 0
        for i, data in enumerate(valid_dataloader, start=0):
            images, labels = data["X"], data["y"]
            images, labels = images.to(self.device), labels.to(self.device)

            loss_batch, accuracy_batch = self.valid_batch(images, labels)
            loss_sum += loss_batch
            accuracy_sum += accuracy_batch

        num_batch = len(valid_dataloader)
        avg_loss = loss_sum / num_batch
        avg_accuracy = accuracy_sum / num_batch

        # print statistics
        self.file.write(
            f"Valid Loss : {avg_loss:.3f} | Valid Accuracy : {avg_accuracy:.3f}\n"
        )
        print(f"Valid Loss : {avg_loss:.3f} | Valid Accuracy : {avg_accuracy:.3f}")
