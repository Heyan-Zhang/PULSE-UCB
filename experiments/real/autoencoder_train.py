import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

import numpy as np
import os

import utils
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = os.getenv("NCB_DATASET", "taobao")
dataset_path = os.getenv("NCB_DATASET_PATH", os.path.join(ROOT_DIR, "data", DATASET))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", nargs='?', type=int, default=0)
    args = parser.parse_args()
    seed = args.seed

    X = np.vstack([
        np.load(os.path.join(dataset_path, "preprocess", f"X0_{DATASET}.npy")),
        np.load(os.path.join(dataset_path, "preprocess", f"X1_{DATASET}.npy")),
    ])
    np.random.shuffle(X)

    model = utils.AE_train(X, seed=seed)

    save_dir = os.path.join(os.path.dirname(__file__), "autoencoders")
    os.makedirs(save_dir, exist_ok=True) 
    
    torch.save(model.state_dict(), f'{save_dir}/AE_{DATASET}_s{seed}.pt')
    
    print(f"Model saved to {save_dir}/AE_{DATASET}_s{seed}.pt")
