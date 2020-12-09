import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd

from data import MRIDataset
from model import BN_Model

import json
import os

from mpi4py import MPI
from queue import Queue
from threading import Thread
from time import sleep

THREAD_COUNT = 2

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()

comm.Barrier()
t_start = MPI.Wtime()

# start = MPI.Wtime()
queue = Queue()
rQueue = Queue()
threads = []

class DownloadWorker(Thread):

#   queue should have input to process
    def __init__(self, queue, returnQueue, rank, threadNum, model):
        Thread.__init__(self)
        self.queue = queue
        self.rQueue = returnQueue
        self.rank = rank
        self.threadNum = threadNum
        self.img = None
        self.label = None
        self.model = model
    
    def _set_img(self, img):
        self.img = img
    
    def _set_label(self, label):
        self.label = label

    def run(self):
        value = 0
        while True:
            try:
            # Get the work from the queue and expand the tuple
            # Sleep was added to simulate intensive work
                # sleep(0.000001) 
                value = self.queue.get()
                value = evaluate(value, self.img, self.label, self.model)
                # print(self.rank, self.threadNum, value)
            finally:
                self.rQueue.put(value)
                self.queue.task_done()

def perturb(p_list, img):
    img_size_x, img_size_y, img_size_z = img.shape[2], img.shape[3], img.shape[4]
    p_img = img.clone()
    for p in p_list:
        x = (p[0].copy() * img_size_x).astype(int)
        x = np.clip(x, 0, img_size_x - 1)
        y = (p[1].copy() * img_size_y).astype(int)
        y = np.clip(y, 0, img_size_y - 1)
        z = (p[2].copy() * img_size_z).astype(int)
        z = np.clip(z, 0, img_size_z - 1)
        vox = p[3].copy()
        vox = np.clip(vox, 0, 1)
        p_img[:, 0, x, y, z] = vox
    return p_img


def orig(best_solution, img_real, label, model, target_label=None):
    true_label_prob = "{}".format(
        F.softmax(model(img_real).squeeze(), dim=0)[label].item()
    )
    img = perturb(best_solution, img_real)
    true_label = "{} {}".format(labels[label], label.item())
    prediction = "{} {}".format(
        labels[model(img).max(-1)[1]], model(img).max(-1)[1][0].item(),
    )
    label_probs = "{}".format(F.softmax(model(img).squeeze(), dim=0))
    mod_true_label_prob = "{}".format(
        F.softmax(model(img).squeeze(), dim=0)[label].item()
    )
    if target_label is not None:
        print(
            "\nTarget Label Probability:",
            F.softmax(model(img).squeeze(), dim=0)[target_label].item(),
        )
    return true_label, prediction, label_probs, true_label_prob, mod_true_label_prob


def evaluate(candidate, img, label, model):
    model.eval()
    p_img = perturb(candidate[1], img)
    preds = (candidate[0], F.softmax(model(p_img).squeeze(), dim=0)[label].item())
    return preds


def evolve(candidates, F=0.5, strategy="clip"):
    gen2 = candidates.copy()
    num_candidates = len(candidates)
    for i in range(num_candidates):
        x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
        x_next = x1 + F * (x2 - x3)
        if strategy == "clip":
            gen2[i] = np.clip(x_next, 0, 1)
        elif strategy == "resample":
            x_oob = np.logical_or((x_next < 0), (1 < x_next))
            x_next[x_oob] = np.random.random((NUM_PIXELS, 4))[x_oob]
            gen2[i] = x_next
    return gen2


def attack(
    model, img_name, img, true_label, target_label=None, iters=5, pop_size=5, verbose=False
):
    # Targeted: maximize target_label if given (early stop > 50%)
    # Untargeted: minimize true_label otherwise (early stop < 5%)
    candidates = np.random.random((pop_size, NUM_PIXELS, 4))
    candidates[:, :, 3] = np.clip(np.random.normal(0.5, 0.5, (pop_size, NUM_PIXELS)), 0, 1)
    is_targeted = target_label is not None
    label = target_label if is_targeted else true_label

    # Evaluate new solutions
    candidates_tuple = []
    for i in range(len(candidates)):
        candidates_tuple.append((i, candidates[i]))

    # new_gen_fitness = evaluate(new_gen_candidates, img, label, model)
    # Put the tasks into the queue as a tuple
    for i in range(len(candidates_tuple)):
        queue.put(candidates_tuple[i])

    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()

    fitness = np.zeros(len(candidates))
    result_fitness = rQueue.queue
    for c_tuple in result_fitness:
        fitness[c_tuple[0]] = c_tuple[1]

    def is_success():
        return (is_targeted and fitness.max() > 0.5) or (
            (not is_targeted) and fitness.min() < 0.5
        )

    for iteration in range(iters):
        # Early Stopping
        if is_success():
            break
        if verbose and iteration % 10 == 0:  # Print progress
            print(
                "Target Probability [Iteration {}]:".format(iteration),
                fitness.max() if is_targeted else fitness.min(),
            )
        # Generate new candidate solutions
        new_gen_candidates = evolve(candidates, strategy="resample")
        
        # Evaluate new solutions
        candidates_tuple = []
        for i in range(len(new_gen_candidates)):
            candidates_tuple.append((i, new_gen_candidates[i]))

        # new_gen_fitness = evaluate(new_gen_candidates, img, label, model)
        # Put the tasks into the queue as a tuple
        for i in range(len(candidates_tuple)):
            queue.put(candidates_tuple[i])

        # Causes the main thread to wait for the queue to finish processing all the tasks
        queue.join()

        new_gen_fitness = np.zeros(len(candidates))
        result_fitness = rQueue.queue
        for c_tuple in result_fitness:
            new_gen_fitness[c_tuple[0]] = c_tuple[1]

        # Replace old solutions with new ones where they are better
        successors = (
            new_gen_fitness > fitness if is_targeted else new_gen_fitness < fitness
        )
        candidates[successors] = new_gen_candidates[successors]
        fitness[successors] = new_gen_fitness[successors]

    best_idx = fitness.argmax() if is_targeted else fitness.argmin()
    best_solution = candidates[best_idx]
    best_score = fitness[best_idx]

    true_label, prediction, label_probs, true_label_prob, mod_true_label_prob = orig(
        best_solution, img, true_label, model
    )
    
    return (
        is_success(),
        best_solution,
        best_score,
        true_label,
        prediction,
        label_probs,
        true_label_prob,
        mod_true_label_prob,
    )

labels = ["HGG", "LGG"]

NUM_PIXELS = int(0.001 * (240 * 240 * 155))

test_set_complete = pd.read_csv("test.csv")

DATA_SIZE = len(test_set_complete)
OUTPUT_PATH = '/home/parthsuresh/few-pix-attack-brats/outputs-threads/'

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

# Create 4 worker threads
for x in range(THREAD_COUNT):
    worker = DownloadWorker(queue, rQueue, comm.rank, x, bn_model)
    # Setting daemon to True will let the main thread exit even though the workers are blocking
    worker.daemon = True
    worker.start()
    threads.append(worker)

with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        x, label, img_name = data["X"], data["y"], data["img_name"]
        img_name = img_name[0]
        # Change worker thread params
        for t in range(THREAD_COUNT):
            threads[t]._set_img(x)
            threads[t]._set_label(label)
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
if comm.rank==0: 
    with open(OUTPUT_PATH + "log.txt", "w") as outfile:
        outfile.write(f"Time taken : {t_diff}")
