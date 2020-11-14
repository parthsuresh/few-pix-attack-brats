import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Model, BN_Model
from train import Train_Model
from data import MRIDataset

use_cuda = True
epochs = 100
batch_size = 5

# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
if use_cuda:
    torch.cuda.empty_cache()
model = Model()
bn_model = BN_Model()

# MRI train, valid datasets,dataloaders
train_dataset = MRIDataset(csv_file="train.csv")
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
)
valid_dataset = MRIDataset(csv_file="valid.csv")
valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)
train_bn_model = Train_Model(device=device, model=bn_model)
train_bn_model.train(train_dataloader, valid_dataloader, epochs)
