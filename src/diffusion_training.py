import os
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.diffusion_model import SimpleUNetSmall, EDMConfig, EDMResidual


def prepare_diffusion_data(data_pack, save_dir="models/diffusion", train_split=0.8):
    """
    Extracts data, performs train/test split, and saves split indices.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1. Extract Data
    x0 = data_pack.get("x0")
    spatial_cond = data_pack.get("spatial_cond") or data_pack.get("condition")
    global_cond = data_pack.get("global_cond") or data_pack.get("scalars")

    if x0 is None or spatial_cond is None or global_cond is None:
        raise ValueError("data_pack is missing required keys ('x0', 'spatial_cond', 'global_cond')")

    # 2. Perform Split
    dataset_size = len(x0)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(train_split * dataset_size)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Save indices so you know exactly which samples went where
    split_info = {
        "train_indices": train_indices,
        "test_indices": test_indices
    }
    torch.save(split_info, os.path.join(save_dir, "dataset_split.pt"))
    print(f"Dataset split saved to {save_dir}/dataset_split.pt")

    # Create Subsets
    full_dataset = TensorDataset(x0, spatial_cond, global_cond)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    return train_dataset, test_dataset, global_cond.shape[1] 

def train_diffusion_model(train_dataset, 
                         test_dataset,
                         cond_dim,
                         model=None,        # Pass your loaded model here
                         save_dir="models/diffusion", 
                         epochs=100, 
                         batch_size=16, 
                         lr=1e-4,
                         device=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n--- Starting Diffusion Training on {device} ---")
    os.makedirs(save_dir, exist_ok=True)

    # 1. EDM Configuration & Statistics
    # ------------------------------------------------------------------
    # We still need r0_std for the checkpoint metadata and EDM config
    print("Calculating/Retrieving residual statistics...")
    x0_train, spatial_train, _ = train_dataset.dataset[train_dataset.indices]
    pred_train = spatial_train[:, 0:1]
    valid_mask = torch.isfinite(x0_train) & torch.isfinite(pred_train)
    r0 = torch.nan_to_num(x0_train, nan=0.0) - torch.nan_to_num(pred_train, nan=0.0)
    r0_std = r0[valid_mask].std().item()

    # If model exists, we assume it has its own internal cfg (EDMResidual)
    if model is not None:
        print("Using existing loaded model. Ensuring it is on the correct device...")
        model = model.to(device)
        cfg = model.cfg # Extract config from the existing model wrapper
    else:
        print("No model provided. Initializing new UNet and EDM wrapper...")
        sigma_data = r0_std
        cfg = EDMConfig(
            sigma_min=0.004 * r0_std,
            sigma_max=160.0 * r0_std,
            sigma_data=sigma_data,
            p_mean=-1.2 + math.log(r0_std / 0.5),
            p_std=1.2
        )
        unet = SimpleUNetSmall(
            in_channels=5, out_channels=1, base_ch=32,
            time_emb_dim=128, cond_dim=cond_dim
        ).to(device)
        model = EDMResidual(unet, cfg, device=device).to(device)

    # 2. Setup Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Training Loop
    # ------------------------------------------------------------------
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for x_b, sc_b, gc_b in train_loader:
            x_b, sc_b, gc_b = x_b.to(device), sc_b.to(device), gc_b.to(device)

            loss = model.p_losses(x_b, sc_b, gc_b)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_v, sc_v, gc_v in test_loader:
                x_v, sc_v, gc_v = x_v.to(device), sc_v.to(device), gc_v.to(device)
                val_loss += model.p_losses(x_v, sc_v, gc_v).item()
        
        avg_val_loss = val_loss / len(test_loader)
        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        # 4. Save Checkpoint
        # ------------------------------------------------------------------
        checkpoint = {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'config_dict': cfg.__dict__, 
            'r0_std': r0_std,
            'epoch': epoch
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_diffusion.pt"))
            
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(save_dir, "latest_diffusion.pt"))

    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.6f}")
    return model


def load_diffusion_model(model_path, device="cuda"):
    ckpt = torch.load(model_path, map_location=device)

    # 1. Initialize U-Net
    unet = SimpleUNetSmall(in_channels=5, out_channels=1, base_ch=32, time_emb_dim=128, cond_dim=5).to(device)
    
    # 2. Load Config using your key 'config_dict'
    cfg = EDMConfig(**ckpt['config_dict'])
    
    # 3. Initialize Wrapper
    model = EDMResidual(unet, cfg, device=device).to(device)

    # 4. Load Weights using your key 'model_state'
    model.load_state_dict(ckpt['model_state'])

    # 5. Get r0_std
    r0_std = ckpt['r0_std']
    if torch.is_tensor(r0_std):
        r0_std = r0_std.item()

    model.eval()
    return model, r0_std