import torch
import numpy as np
from tqdm import tqdm
import os

from models import Autoencoder_BN

# Data conversion utilities
def convert_features(model, high_dim_data, device, batch_size=10000):
    """
    to avoid memory overflow, we convert the features in batches
    """
    model.eval()  # Set the model to evaluation mode
    low_dim_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, high_dim_data.shape[0], batch_size), desc="Converting Features"):
            batch = high_dim_data[i:i+batch_size]
            
            # Convert to tensor and move to the selected device.
            batch_tensor = torch.from_numpy(batch).to(torch.float).to(device)
            
            # Run the encoder to get low-dimensional embeddings.
            encoded_batch = model.encoding_result(batch_tensor)
            
            # Move embeddings back to CPU as NumPy arrays.
            low_dim_embeddings.append(encoded_batch.cpu().numpy())
            
    return np.vstack(low_dim_embeddings)

if __name__ == "__main__":
    SEED = 0
    EMB_DIM = 32  # goal embedding dimension
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DATASET = os.getenv("NCB_DATASET", "taobao")
    BASE_DATA_PATH = os.getenv("NCB_BASE_DATA_PATH", os.path.join(ROOT_DIR, "data", DATASET, "preprocess"))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading high-dimensional data...")
    X0_high_dim = np.load(os.path.join(BASE_DATA_PATH, f"X0_{DATASET}.npy"))
    X1_high_dim = np.load(os.path.join(BASE_DATA_PATH, f"X1_{DATASET}.npy"))
    print(f"X0 shape: {X0_high_dim.shape}, X1 shape: {X1_high_dim.shape}")
    
    # Input feature dimension.
    raw_dim = X0_high_dim.shape[1]

    # Load the trained autoencoder.
    model_path = os.path.join(os.path.dirname(__file__), "autoencoders", f"AE_{DATASET}_s{SEED}.pt")
    print(f"Loading autoencoder model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Instantiate model and load weights.
    model = Autoencoder_BN(raw_dim=raw_dim, emb_dim=EMB_DIM).to(device)
    model.load_state_dict(state_dict)

    # Generate 32-dimensional embeddings.
    X0_32d = convert_features(model, X0_high_dim, device)
    X1_32d = convert_features(model, X1_high_dim, device)

    # Save 32-dimensional features.
    output_path_X0 = os.path.join(BASE_DATA_PATH, f"X0_{DATASET}_32d.npy")
    output_path_X1 = os.path.join(BASE_DATA_PATH, f"X1_{DATASET}_32d.npy")
    
    print(f"Saving 32-dimensional data...")
    print(f"X0_32d shape: {X0_32d.shape} -> {output_path_X0}")
    print(f"X1_32d shape: {X1_32d.shape} -> {output_path_X1}")
    
    np.save(output_path_X0, X0_32d)
    np.save(output_path_X1, X1_32d)
    
    print("Conversion complete!")
