import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Model, BN_Model
from data import MRIDataset

use_cuda = True

print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
if use_cuda:
    torch.cuda.empty_cache()
bn_model = BN_Model()
bn_model.load_state_dict(torch.load("bn_models/model_epoch_3.pth"))
bn_model.eval()

# MRI train, valid datasets,dataloaders
test_dataset = MRIDataset(csv_file="test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

with torch.no_grad():
    acc_sum = 0
    for i, data in enumerate(test_dataloader):
        x, labels = data["X"], data["y"]
        preds = bn_model(x)
        _, pred_labels = preds.max(1, keepdim=True)
        accuracy = (
            pred_labels.eq(labels.view_as(pred_labels)).sum().item() / labels.shape[0]
        )
        acc_sum += accuracy
    avg_acc = acc_sum / len(test_dataloader)
    print(f"Test Accuracy : {avg_acc}")
