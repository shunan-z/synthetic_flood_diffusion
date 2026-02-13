import torch
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. Helper Functions (MSE & Grid)
# ---------------------------------------------------------
def mse_similarity(map1, map2):
    """Calculates Mean Squared Error ignoring NaNs."""
    # Ensure inputs are float arrays
    m1 = map1.astype(np.float32)
    m2 = map2.astype(np.float32)
    
    # Create mask where BOTH are valid
    valid_mask = np.isfinite(m1) & np.isfinite(m2)
    
    if not np.any(valid_mask):
        return float('nan')
        
    diff = m1[valid_mask] - m2[valid_mask]
    return np.mean(diff ** 2)

def to_hw(tensor):
    """Converts (1, 1, H, W) tensor to (H, W) numpy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().squeeze()
    return tensor.squeeze()

def ensure_grid_indices(df, h=64, w=64):
    """Ensures the dataframe has 'row' and 'col' columns for reshaping."""
    if "row" not in df.columns or "col" not in df.columns:
        # Assuming data is ordered row-major or similar. 
        # If your data is random, this needs your specific grid logic.
        df = df.copy()
        n = len(df)
        df["row"] = np.arange(n) // w
        df["col"] = np.arange(n) % w
    return df

# ---------------------------------------------------------
# 2. Core Sampling Logic (EDM Loop)
# ---------------------------------------------------------
@torch.no_grad()
def edm_sample_residual_from_wrapper(
    edm,
    spatial_cond: torch.Tensor,
    global_cond: torch.Tensor,
    shape=(64, 64),
    miss_mask: torch.Tensor = None,
    num_steps: int = 40,
    rho: float = 7.0   # Karras constant
):
    device = spatial_cond.device
    B, _, H, W = spatial_cond.shape
    cfg = edm.cfg

    # 1. Sanitize Inputs
    sc_sane = torch.nan_to_num(spatial_cond, nan=0.0)
    if miss_mask is None:
        miss_mask = torch.zeros((B, 1, H, W), device=device)

    # 2. Karras Sigma Schedule (Power-law distribution)
    step_indices = torch.arange(num_steps, device=device)
    t_max = cfg.sigma_max ** (1 / rho)
    t_min = cfg.sigma_min ** (1 / rho)
    sigmas = (t_max + step_indices / (num_steps - 1) * (t_min - t_max)) ** rho
    # Append 0.0 for the final step
    sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

    # 3. Initialize with noise at sigma_max
    r = torch.randn((B, 1, H, W), device=device) * sigmas[0]

    # 4. Iterative Denoising (Euler)
    for i in range(num_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = torch.full((B,), sigma, device=device)

        # Predict CLEAN residual r0
        r0_hat = edm.denoise(r, sigma_batch, sc_sane, miss_mask, global_cond)

        # Euler Step
        d_i = (r - r0_hat) / sigma
        r = r + (sigma_next - sigma) * d_i

    # 5. Final Composition
    # r at the end of loop is effectively r0_hat (clean residual)
    r_hat = r * (1.0 - miss_mask)
    pred_map = sc_sane[:, 0:1]
    x_hat = pred_map + r_hat # changed line

    # 6. Re-apply NaN mask
    valid_bool = (miss_mask < 0.5)
    nan_val = torch.tensor(float("nan"), device=device)
    x_hat = torch.where(valid_bool, x_hat, nan_val)

    return x_hat, pred_map, r_hat

# ---------------------------------------------------------
# 3. Scenario Builder (XGB + Diffusion)
# ---------------------------------------------------------
def sample_one_scenario_from_globals_edm(
    edm,
    regression_model,              # XGBoost model
    target_data,
    feature_cols,
    scenario_name,
    category,
    speed,
    tide,
    direction_deg,
    device,
    miss_mask=None,
    H=64, W=64,
    dtype=torch.float32
):
    """
    Generates a flood map for a specific scenario using XGBoost (base) + EDM (residual).
    """
    # 1) Prepare Grid & Globals
    gdf = ensure_grid_indices(target_data)
    df = gdf[gdf["target_name"] == scenario_name].copy()
    
    # Safety check
    if len(df) != H * W:
        # Warning: size mismatch. Attempting to crop or pad if critical, 
        # but usually raising error is safer.
        raise ValueError(f"Scenario {scenario_name} has {len(df)} rows, expected {H*W}")

    df = df.sort_values(["row", "col"])

    # 2) Overwrite Global Conditions
    theta = np.deg2rad(direction_deg)
    dir_sin = np.sin(theta)
    dir_cos = np.cos(theta)

    df["Category"]      = category
    df["Speed"]         = speed
    df["Tide"]          = float(tide)
    df["Direction_sin"] = dir_sin
    df["Direction_cos"] = dir_cos

    # 3) Regression Prediction
    X = df[feature_cols]
    is_valid = (miss_mask.flatten() == 0).cpu().numpy()
    pred_flat = np.full(len(X), np.nan)
    if is_valid.any():
        pred_flat[is_valid] = regression_model.predict(X[is_valid])
    pred_map = pred_flat.reshape(64, 64)

    # 4) Elevation Map
    elev_map = df["elev_mean"].to_numpy().reshape(H, W)
    elev_var = df["elev_var"].to_numpy().reshape(H, W)

    # 5) Build Tensors
    spatial_cond = torch.tensor(
        np.stack([pred_map, elev_map, elev_var], axis=0), 
        dtype=dtype, device=device
    ).unsqueeze(0)  # (1, 3, H, W)

    global_cond = torch.tensor(
        [[category, speed, tide, dir_sin, dir_cos]],
        dtype=dtype, device=device
    ) # (1, 5)

    if miss_mask is not None:
        miss_mask = miss_mask.to(device)
    else:
        miss_mask = torch.zeros((1, 1, H, W), dtype=dtype, device=device)

    # 6) Run EDM Sampling
    edm.eval()
    return edm_sample_residual_from_wrapper(
        edm,
        spatial_cond,
        global_cond,
        shape=(H, W),
        miss_mask=miss_mask,
        num_steps=40
    )

# ---------------------------------------------------------
# 4. Main Evaluation Loop
# ---------------------------------------------------------
def evaluate_scenarios(
    scenario_labels,
    edm_model,
    xgb_model,
    target_data,
    feature_cols,
    data_pack,
    device
):
    """
    Iterates over a list of scenario labels, generates maps, and calculates MSE.
    """
    mse_pred_list = []
    mse_xhat_list = []
    processed_names = []

    # Unpack ground truth data
    all_names = data_pack["scenario_names"]
    all_images = data_pack["x0"]

    print(f"Starting evaluation for {len(scenario_labels)} scenarios...")

    for target_label in scenario_labels:
        try:
            # --- A. Setup ---
            if target_label not in all_names:
                print(f"Skipping {target_label}: Not found in data_pack")
                continue

            idx = all_names.index(target_label)
            scenario_name = all_names[idx] # Usually same as target_label

            # Get parameters from dataframe
            row = target_data[target_data["target_name"] == target_label].iloc[0]
            category      = int(row["Category"])
            speed         = float(row["Speed"])
            tide          = int(row["Tide"])
            
            # Handle direction column name flexibility
            if "Direction_deg" in row:
                direction_deg = float(row["Direction_deg"])
            else:
                direction_deg = float(row.get("Direction", 0.0))

            # --- B. Prepare Mask from Ground Truth ---
            # We use the ground truth to determine where valid pixels are
            orig_tensor = all_images[idx] # (1, H, W)
            if not isinstance(orig_tensor, torch.Tensor):
                orig_tensor = torch.tensor(orig_tensor)
            
            orig_tensor = orig_tensor.to(device)
            valid_mask = torch.isfinite(orig_tensor).float()
            
            # Fix dimensions if needed
            if valid_mask.ndim == 3:
                valid_mask = valid_mask[0:1] # Ensure (1, H, W)
            elif valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(0)
                
            miss_mask = (1.0 - valid_mask).unsqueeze(0) # Add Batch dim -> (1, 1, H, W)

            # --- C. Generate ---
            x_hat, pred, res = sample_one_scenario_from_globals_edm(
                edm=edm_model,
                model=xgb_model,
                target_data=target_data,
                feature_cols=feature_cols,
                scenario_name=scenario_name,
                category=category,
                speed=speed,
                tide=tide,
                direction_deg=direction_deg,
                device=device,
                miss_mask=miss_mask
            )

            # --- D. Compare (MSE) ---
            # Convert to numpy (H, W) and orient correctly
            orig_hw  = to_hw(orig_tensor)
            pred_hw  = to_hw(pred)
            xhat_hw  = to_hw(x_hat)
            
            # Calculate MSE
            mse_pred = mse_similarity(pred_hw, orig_hw)
            mse_xhat = mse_similarity(xhat_hw, orig_hw)

            mse_pred_list.append(mse_pred)
            mse_xhat_list.append(mse_xhat)
            processed_names.append(target_label)
            
        except Exception as e:
            print(f"Error evaluating {target_label}: {str(e)}")
            continue

    print(f"Completed. Evaluated {len(processed_names)} scenarios.")
    return processed_names, mse_pred_list, mse_xhat_list