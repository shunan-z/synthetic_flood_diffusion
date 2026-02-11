import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
# Import the 3 specific functions
from src.data_loader import load_meow_data, load_substation_data, load_elevation_data
from src.regression_data_preprocess import preprocess_data
from src.regression_model import FloodXGBModel
from src.diffusion_data_preprocess import add_regression_pred, build_diffusion_tensors
from src.diffusion_training import train_diffusion_model
from src.sampling_evaluation import sample_one_scenario_from_globals_edm
from src.visualization import plot_scenario_comparison, to_hw


def main():
    print("--- 1. Loading Data ---")
    gdf = load_meow_data()
    substation = load_substation_data()
    dem_data, dem_transform, dem_crs = load_elevation_data()

    print("\n--- 2. Processing Pipeline ---")
    # Now returns BOTH formats
    tensor_images, tabular_data = preprocess_data(gdf, substation, dem_data, dem_transform)

    print("\n--- 3. Outputs Ready ---")
    

    # --- A. Check Tabular Data (XGBoost) ---
    print(f"XGBoost DataFrame: {tabular_data.shape} rows")
    print("First 5 rows of Tabular Data:")
    pd.set_option('display.max_columns', None)
    
    # --- B. Check Tensor Output (Diffusion) ---
    final_tensor = torch.from_numpy(tensor_images).float()
    print(f"\nDiffusion Tensor: {final_tensor.shape} (Channels, H, W)")
    
    # 1. Train Regression Model
    print("\n--- 3. Running Baseline Regression ---")
    xgb_model = FloodXGBModel()
    metrics = xgb_model.train(tabular_data)

    # --- NEW SECTION: Prepare for Diffusion ---
    print("\n--- 4. Preparing Data for Diffusion Model ---")

    # 1. Add XGBoost predictions to the dataframe
    # (This adds the 'xgb_pred' column)
    tabular_data = add_regression_pred(tabular_data, xgb_model)

    # 2. Build the Tensor Dictionary
    diffusion_data = build_diffusion_tensors(
    target_data=tabular_data,
    scenario_col="target_name",
    grid_size=(64, 64),
    global_cols=["Category", "Speed", "Tide", "Direction_sin", "Direction_cos"]
)
    diffusion_model = train_diffusion_model(
    data_pack=diffusion_data,   # <--- Pass the dictionary directly
    save_dir="models/diffusion",
    epochs=2,
    batch_size=16
)

    # ---------------------------------------------------------
    # 6. Evaluate & Visualize a Specific Scenario
    # ---------------------------------------------------------
    print("\n--- 6. Visualizing Result ---")
    
    # A. Define the Feature Columns explicitly
    # These must match exactly what you used to train the XGBoost model
    feature_cols = [
        "row", "col", 'latitude', 'longitude',           # Spatial coords            
        "elev_mean", 'elev_var',            # Elevation
        "Category", "Speed", "Tide", "Direction_sin", "Direction_cos" # Global Params
    ]

    # B. Select a Scenario
    target_label = "nw410i2"
    
    # --- FIX: Use "names" instead of "scenario_names" ---
    labels = diffusion_data["names"] 
    
    # Safety check: if label doesn't exist, pick the first one
    if target_label not in labels:
        print(f"Warning: '{target_label}' not found. Defaulting to first available scenario.")
        target_label = labels[0]

    idx = labels.index(target_label)
    print(f"Visualizing Scenario: {target_label} (Index {idx})")

    # C. Get Metadata
    # Retrieve the specific conditions (speed, tide, etc.) for this scenario
    row = tabular_data[tabular_data["target_name"] == target_label].iloc[0]
    
    # D. Prepare Masks
    # Get Ground Truth image
    orig_image = diffusion_data["x0"][idx] 
    
    # Move to GPU for processing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    orig_tensor = torch.tensor(orig_image).to(device)
    
    # Create mask: 1 where valid, 0 where NaN
    valid_mask = torch.isfinite(orig_tensor).float()
    miss_mask = (1.0 - valid_mask).unsqueeze(0)# Shape: (1, 1, H, W)

    # E. Generate Prediction (XGB + Diffusion)
    # This runs the sampling loop you defined
    x_hat, pred, res = sample_one_scenario_from_globals_edm(
        edm=diffusion_model,
        regression_model=xgb_model,
        target_data=tabular_data,
        feature_cols=feature_cols,   
        scenario_name=target_label,
        category=int(row["Category"]),
        speed=float(row["Speed"]),
        tide=int(row["Tide"]),
        direction_deg=float(row.get("Direction_deg", row.get("Direction", 0.0))),
        device=device,
        miss_mask=miss_mask
    )

    # F. Plot
    os.makedirs("results", exist_ok=True)
    
    plot_scenario_comparison(
        orig=orig_image,      
        pred=pred,            
        x_hat=x_hat,          
        scenario_name=target_label,
        save_dir="results"
    )
    
    print(f"Done! Result saved to results/{target_label}_maps.png")
if __name__ == "__main__":
    main()