import numpy as np
import os

RANDOM_SEED = 42  # For reproducibility.
PRETRAIN_SPLIT_RATIO = 0.2  # Proportion reserved for pre-training.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = os.getenv("NCB_DATASET", "taobao")
BASE_DATA_PATH = os.getenv("NCB_BASE_DATA_PATH", os.path.join(ROOT_DIR, "data", DATASET, "preprocess"))


if __name__ == "__main__":
    print("--- Starting Data Splitting Process ---")
    print(f"Pre-train split ratio: {PRETRAIN_SPLIT_RATIO}")

    # Load 32-dimensional features.
    path_x0 = os.path.join(BASE_DATA_PATH, f"X0_{DATASET}_32d.npy")
    path_x1 = os.path.join(BASE_DATA_PATH, f"X1_{DATASET}_32d.npy")

    print(f"Loading 32-dimensional data from:")
    print(f"  - {path_x0}")
    print(f"  - {path_x1}")

    try:
        x0_32d = np.load(path_x0)
        x1_32d = np.load(path_x1)
    except FileNotFoundError:
        print("\nError: 32-dimensional data files not found!")
        print("Please make sure you have successfully run 'generate_embeddings.py' first.")
        exit()

    # Merge data and create reward labels.
    print("\nStep 1: Merging data and creating reward labels...")
    
    # Stack the two classes vertically.
    Y_true = np.vstack([x0_32d, x1_32d])
    
    # Build reward labels: X0 -> 0, X1 -> 1.
    R_rewards = np.concatenate([
        np.zeros(x0_32d.shape[0], dtype=int),
        np.ones(x1_32d.shape[0], dtype=int)
    ])
    
    total_samples = Y_true.shape[0]
    print(f"Total samples merged: {total_samples}")

    # Shuffle data.
    print("\nStep 2: Shuffling data and rewards...")
    
    # Set random seed for reproducibility.
    np.random.seed(RANDOM_SEED)
    
    # Generate shuffled indices.
    shuffled_indices = np.random.permutation(total_samples)
    
    # Apply the same permutation to features and rewards.
    Y_shuffled = Y_true[shuffled_indices]
    R_shuffled = R_rewards[shuffled_indices]
    
    print("Shuffling complete.")

    # Split into pre-train and online sets.
    print("\nStep 3: Splitting data into pre-train and online sets...")
    
    # Compute split index.
    split_index = int(total_samples * PRETRAIN_SPLIT_RATIO)
    
    # Apply split.
    Y_pretrain = Y_shuffled[:split_index]
    
    Y_online = Y_shuffled[split_index:]
    R_online = R_shuffled[split_index:]
    
    print(f"Pre-train set size: {Y_pretrain.shape[0]} samples")
    print(f"Online evaluation set size: {Y_online.shape[0]} samples")

    # Save split datasets.
    print("\nStep 4: Saving the final datasets...")
    
    # Output paths.
    path_pretrain = os.path.join(BASE_DATA_PATH, f"Y_pretrain_{DATASET}.npy")
    path_online_y = os.path.join(BASE_DATA_PATH, f"Y_online_{DATASET}.npy")
    path_online_r = os.path.join(BASE_DATA_PATH, f"R_online_{DATASET}.npy")

    # Write files.
    np.save(path_pretrain, Y_pretrain)
    np.save(path_online_y, Y_online)
    np.save(path_online_r, R_online)
    
    print(f"  - Pre-train features saved to: {path_pretrain}")
    print(f"  - Online features saved to: {path_online_y}")
    print(f"  - Online rewards saved to: {path_online_r}")
    
    print("\n--- Process Complete! ---")
