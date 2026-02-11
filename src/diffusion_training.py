import os
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from src.diffusion_model import SimpleUNetSmall, EDMConfig, EDMResidual

def train_diffusion_model(data_pack, 
                          save_dir="models/diffusion", 
                          epochs=100, 
                          batch_size=16, 
                          lr=1e-4,
                          device=None):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n--- Starting Diffusion Training from Memory on {device} ---")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Extract Data from Memory (Handle potential key name variations)
    # ------------------------------------------------------------------
    x0 = data_pack.get("x0")
    
    # Check for "spatial_cond" (your code) OR "condition" (my previous code)
    spatial_cond = data_pack.get("spatial_cond")
    if spatial_cond is None:
        spatial_cond = data_pack.get("condition")

    # Check for "global_cond" OR "scalars"
    global_cond = data_pack.get("global_cond")
    if global_cond is None:
        global_cond = data_pack.get("scalars")

    if x0 is None or spatial_cond is None or global_cond is None:
        raise ValueError("Error: data_pack is missing required keys ('x0', 'spatial_cond', 'global_cond')")

    print(f"Loaded {len(x0)} samples from memory.")

    # 2. Dynamic EDM Configuration (Calculate r0_std)
    # ------------------------------------------------------------------
    print("Calculating residual statistics for EDM scaling...")
    
    # Spatial cond is [Pred, Elev]. We need Pred (channel 0).
    pred = spatial_cond[:, 0:1] 
    
    # Identify valid pixels (where both Truth and Pred exist)
    valid_mask = torch.isfinite(x0) & torch.isfinite(pred)
    
    # Calculate Residual = Truth - Pred
    x0_safe = torch.nan_to_num(x0, nan=0.0)
    pred_safe = torch.nan_to_num(pred, nan=0.0)
    r0 = x0_safe - pred_safe
    
    # Get standard deviation of the residual (only valid pixels)
    r0_valid = r0[valid_mask]
    
    if r0_valid.numel() == 0:
        raise ValueError("Error: No valid pixels found to calculate statistics!")
        
    r0_std = r0_valid.std().item()
    print(f"   Residual STD (r0_std): {r0_std:.6f}")

    # --- Apply Your Formulas ---
    sigma_data = r0_std
    sigma_min = 0.004 * r0_std
    sigma_max = 160.0 * r0_std
    p_mean = -1.2 + math.log(r0_std / 0.5)
    p_std = 1.2
    
    print(f"   EDM Config: sigma_data={sigma_data:.4f}")
    print(f"               Range=[{sigma_min:.6f}, {sigma_max:.4f}]")

    # Initialize Config
    cfg = EDMConfig(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sigma_data=sigma_data,
        p_mean=p_mean,
        p_std=p_std
    )

    # 3. Setup Model & Dataset
    # ------------------------------------------------------------------
    # Initialize UNet
    unet = SimpleUNetSmall(
        in_channels=4,          # [Noisy_Res, Pred, Elev, Mask]
        out_channels=1,         # [Predicted_Res]
        base_ch=32,
        time_emb_dim=128,
        cond_dim=global_cond.shape[1]
    ).to(device)
    
    # Wrap in EDM
    model = EDMResidual(unet, cfg, device=device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Prepare DataLoaders
    full_dataset = TensorDataset(x0, spatial_cond, global_cond)
    
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 4. Training Loop
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
                loss = model.p_losses(x_v, sc_v, gc_v)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        
        print(f"Epoch {epoch+1:03d} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}")

        # Save Checkpoint (Including r0_std for inference later)
        checkpoint = {
            'model_state': model.state_dict(),
            'config_dict': cfg.__dict__, 
            'r0_std': r0_std,
            'epoch': epoch
        }

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_diffusion.pt"))
            
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, os.path.join(save_dir, "latest_diffusion.pt"))

    print(f"✅ Training Complete. Best Val Loss: {best_val_loss:.6f}")
    return model