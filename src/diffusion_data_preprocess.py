import numpy as np
import pandas as pd
import geopandas as gpd
import torch

def add_regression_pred(target_data, model, pred_col="regression_pred"):
    """
    Runs the XGBoost model on the dataframe and adds a prediction column.
    """
    print("Running XGBoost inference on all data...")

    feature_cols = [
    "elev_mean",
    "elev_var",
    "Category",
    "Speed",
    "Tide",
    "Direction_sin",
    "Direction_cos",
    "latitude",
    "longitude"
]
    gdf = target_data.copy()
    
    # Ensure we don't crash on missing data
    valid_mask = gdf[feature_cols].notna().all(axis=1)
    
    # Predict
    if valid_mask.sum() > 0:
        preds = model.predict(gdf.loc[valid_mask, feature_cols])
        gdf.loc[valid_mask, pred_col] = preds
    else:
        print("WARNING: No valid rows found for prediction!")
        
    return gdf

def _infer_grid_indices(gdf):
    """
    Helper: Converts spatial coordinates (x, y) into grid indices (row, col)
    assuming a regular 64x64 grid.
    """
    out = gdf.copy()
    
    # Get centroids
    cent = out.geometry.centroid
    xs = np.round(cent.x.to_numpy(), 5) # Round to avoid float precision errors
    ys = np.round(cent.y.to_numpy(), 5)

    # Find unique coords to establish grid lines
    x_unique = np.sort(np.unique(xs))
    y_unique = np.sort(np.unique(ys))

    # Map actual coords to integer indices (0..63)
    # Note: We usually want y to be row (0 is bottom or top) and x to be col
    # Here we map x -> col, y -> row
    out["col"] = np.searchsorted(x_unique, xs)
    out["row"] = np.searchsorted(y_unique, ys)
    
    return out

def build_diffusion_tensors(target_data, 
                            scenario_col="target_name",
                            target_val_col="target_value",
                            pred_col="regression_pred",
                            elev_col="elev_mean",
                            global_cols=["Category", "Speed", "Tide", "Direction_sin", "Direction_cos"],
                            grid_size=(64, 64)):
    """
    Converts the DataFrame into PyTorch tensors for the Diffusion Model.
    
    Returns a Dictionary:
      - 'x0': (N, 1, 64, 64) -> Ground Truth (Target)
      - 'condition': (N, 2, 64, 64) -> [REGRESSION_Pred, Elevation]
      - 'scalars': (N, 5) -> [Cat, Speed, Tide, Dir_sin, Dir_cos]
    """
    H, W = grid_size
    expected_pixels = H * W
    
    # 1. Infer Grid Indices (Row/Col)
    print("Inferring grid indices from geometry...")
    df_grid = _infer_grid_indices(target_data)
    
    # Containers
    x0_list = []        # Ground Truth
    cond_list = []      # Spatial Conditions (XGB + Elev)
    scalar_list = []    # Global Scalars
    names_list = []
    
    print(f"Processing scenarios grouped by '{scenario_col}'...")
    
    # 2. Group by Scenario (e.g., 'ene210i1', 'ene210i2'...)
    grouped = df_grid.groupby(scenario_col)
    
    for name, group in grouped:
        # Sanity Check: Must be exactly 64x64
        if len(group) != expected_pixels:
            print(f"Skipping {name}: Has {len(group)} pixels, expected {expected_pixels}")
            continue
            
        # Sort by row/col to ensure image alignment
        # We sort by row first, then col to fill the matrix correctly
        group = group.sort_values(["row", "col"])
        
        # --- A. Extract Maps (H, W) ---
        # Ground Truth
        truth_map = group[target_val_col].values.reshape(H, W)
        
        # Condition 1: XGBoost Prediction
        pred_map = group[pred_col].values.reshape(H, W)
        
        # Condition 2: Elevation
        elev_map = group[elev_col].values.reshape(H, W)
        
        # --- B. Extract Scalars (Vector) ---
        # Take the first row's values (assuming they are constant for the whole map)
        scalars = group[global_cols].iloc[0].values.astype(float)
        
        # --- C. Collect ---
        x0_list.append(truth_map)
        
        # Stack [Pred, Elev] -> (2, H, W)
        spatial_stack = np.stack([pred_map, elev_map], axis=0) 
        cond_list.append(spatial_stack)
        
        scalar_list.append(scalars)
        names_list.append(name)

    # 3. Convert to PyTorch Tensors
    # x0: (N, H, W) -> (N, 1, H, W)
    x0_tensor = torch.tensor(np.array(x0_list), dtype=torch.float32).unsqueeze(1)
    
    # condition: (N, 2, H, W)
    cond_tensor = torch.tensor(np.array(cond_list), dtype=torch.float32)
    
    # scalars: (N, 5)
    scalar_tensor = torch.tensor(np.array(scalar_list), dtype=torch.float32)
    
    print(f"✅ Successfully created tensors for {len(names_list)} scenarios.")
    print(f"   x0 shape: {x0_tensor.shape}")
    print(f"   condition shape: {cond_tensor.shape}")
    
    return {
        "x0": x0_tensor,          # The target image to generate/denoise
        "condition": cond_tensor, # The conditioning images
        "scalars": scalar_tensor, # The global parameters
        "names": names_list
    }