import numpy as np
import pandas as pd
from mpi4py import MPI
import os
import json

import torch
from torch.utils.data import Dataset, DataLoader

from perturb import attack
from data import MRIDataset
from model import BN_Model

comm = MPI.COMM_WORLD

nprocs = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

test_set_complete = pd.read_csv("test.csv")

DATA_SIZE = len(test_set_complete)
OUTPUT_PATH = '/home/parthsuresh/few-pix-attack-brats/outputs/'

# Create individual csv file for each process
data_per_process = {"path":[], "label":[]}
for i in range(comm.rank, DATA_SIZE, comm.size):
    path = test_set_complete.iloc[i]["path"]
    label = test_set_complete.iloc[i]["label"]
    if (i % nprocs == comm.rank):
        data_per_process["path"].append(path)
        data_per_process["label"].append(label)

df = pd.DataFrame.from_dict(data_per_process)
df.to_csv(f"test_{comm.rank}.csv", index=False)

test_dataset = MRIDataset(csv_file=f"test_{comm.rank}.csv")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

use_cuda = False
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
if use_cuda:
    torch.cuda.empty_cache()
bn_model = BN_Model()
bn_model.load_state_dict(torch.load("bn_models/model_epoch_3.pth"))
bn_model.eval()

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        x, label, img_name = data["X"], data["y"], data["img_name"]
        img_name = img_name[0]
        preds = bn_model(x)
        _, pred_labels = preds.max(1, keepdim=True)
        accuracy = (
            pred_labels.eq(label.view_as(pred_labels)).sum().item() / label.shape[0]
        )
        if accuracy != 1:
            print(f"{img_name} is incorrectly classified")
            continue
        print(f"Starting attack on {img_name}")
        is_success, best_solution, best_score, true_label, prediction, \
            label_probs, true_label_prob, mod_true_label_prob = attack(bn_model, img_name, x, label)
        print(f"Attack on {img_name} successful: {is_success}")

        results = {}

        results['is_success'] = str(is_success)
        results['best_solution'] = best_solution.tolist()
        results['best_score'] = best_score.tolist()
        results['true_label'] = true_label
        results['prediction'] = prediction
        results['label_probs'] = label_probs
        results['true_label_prob'] = true_label_prob
        results['mod_true_label_prob'] = mod_true_label_prob

        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)

        with open(OUTPUT_PATH + img_name + '.out', 'w') as outfile:
            json.dump(results, outfile)



comm.Barrier()
t_diff = MPI.Wtime() - t_start
if comm.rank==0: print (t_diff)