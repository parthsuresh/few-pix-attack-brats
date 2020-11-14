import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data import MRIDataset
from model import BN_Model

labels = ["HGG", "LGG"]


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
            "Target Label Probability:",
            F.softmax(model(img).squeeze(), dim=0)[target_label].item(),
        )
    return true_label, prediction, label_probs, true_label_prob, mod_true_label_prob


def evaluate(candidates, img, label, model):
    preds = []
    model.eval()
    with torch.no_grad():
        for i, xs in enumerate(candidates):
            p_img = perturb(xs, img)
            preds.append(F.softmax(model(p_img).squeeze(), dim=0)[label].item())
    return np.array(preds)


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
            x_next[x_oob] = np.random.random((5, 4))[x_oob]
            gen2[i] = x_next
    return gen2


def attack(
    model, img, true_label, target_label=None, iters=100, pop_size=400, verbose=True
):
    # Targeted: maximize target_label if given (early stop > 50%)
    # Untargeted: minimize true_label otherwise (early stop < 5%)
    candidates = np.random.random((pop_size, 5, 4))
    candidates[:, :, 3] = np.clip(np.random.normal(0.5, 0.5, (pop_size, 5)), 0, 1)
    is_targeted = target_label is not None
    label = target_label if is_targeted else true_label
    fitness = evaluate(candidates, img, label, model)

    def is_success():
        return (is_targeted and fitness.max() > 0.5) or (
            (not is_targeted) and fitness.min() < 0.05
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
        new_gen_fitness = evaluate(new_gen_candidates, img, label, model)
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


if __name__ == "__main__":
    test_dataset = MRIDataset(csv_file="test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    use_cuda = True

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    if use_cuda:
        torch.cuda.empty_cache()
    bn_model = BN_Model()
    bn_model.load_state_dict(torch.load("bn_models/model_epoch_3.pth"))
    bn_model.eval()

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            x, label = data["X"], data["y"]
            attack(bn_model, x, label)
