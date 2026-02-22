import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os

from models import InferenceModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET = os.getenv("NCB_DATASET", "taobao")
BASE_DATA_PATH = os.getenv("NCB_BASE_DATA_PATH", os.path.join(ROOT_DIR, "data", DATASET, "preprocess"))
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "pretrain_models")

OBSERVED_DIMS = 16  # dimension of observed features S_t

# Neural network training parameters.
HIDDEN_DIMS = 128
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 0.001

if __name__ == "__main__":
    # Select compute device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure model save directory exists.
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # Load and split pre-training data.
    print("\nStep 1: Loading and splitting pre-train data...")
    pretrain_data_path = os.path.join(BASE_DATA_PATH, f"Y_pretrain_{DATASET}.npy")
    
    try:
        y_pretrain = np.load(pretrain_data_path)
    except FileNotFoundError:
        print(f"\nError: Pre-train data file not found at {pretrain_data_path}")
        print("Please make sure you have successfully run 'prepare_final_data.py' first.")
        exit()

    # S_t: first 16 dimensions; S'_t: last 16 dimensions.
    X_train_np = y_pretrain[:, :OBSERVED_DIMS]
    y_train_np = y_pretrain[:, OBSERVED_DIMS:]
    
    print(f"Total pre-train samples: {X_train_np.shape[0]}")
    print(f"Observed feature dimensions (S_t): {X_train_np.shape[1]}")
    print(f"Target feature dimensions (S'_t): {y_train_np.shape[1]}")

    # Convert NumPy arrays to PyTorch tensors.
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()
    
    # Build batched training loader.
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize model, loss, and optimizer.
    print("\nStep 2: Initializing model and optimizer...")
    
    # Instantiate model.
    model = InferenceModel(
        input_dim=OBSERVED_DIMS,
        output_dim=y_train_np.shape[1],
        hidden_dim=HIDDEN_DIMS
    ).to(device)
    
    # Regression loss.
    criterion = nn.MSELoss()
    
    # Adam optimizer.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train.
    print("\nStep 3: Starting model training...")
    model.train()  
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass.
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and update.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.6f}")

    print("Training complete.")

    # Save trained model.
    print("\nStep 4: Saving the trained model...")
    save_path = os.path.join(MODEL_SAVE_PATH, f'MLP.pth')
    
    # Save model state dict.
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    print("\n--- Pre-training Process Complete! ---")
