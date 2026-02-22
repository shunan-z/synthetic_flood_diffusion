import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
# Import the 3 specific functions
from src.data_loader import load_meow_data, load_substation_data, load_elevation_data
from src.regression_data_preprocess import preprocess_data
from src.regression_model import FloodXGBModel,FloodLinearModel
from src.diffusion_data_preprocess import add_regression_pred, build_diffusion_tensors
from src.diffusion_training import train_diffusion_model, load_diffusion_model, prepare_diffusion_data
from src.sampling_evaluation import sample_one_scenario_from_globals_edm
from src.visualization import plot_scenario_comparison, to_hw, check_regression_prediction


def main():
    print("--- 1. Loading Data ---")
    gdf = load_meow_data()
    substation = load_substation_data()
    dem_data, dem_transform, dem_crs = load_elevation_data()

    print("\n--- 2. Processing Pipeline ---")
    # Now returns BOTH formats
    tabular_data = preprocess_data(gdf, substation, dem_data, dem_transform)


    print("\n--- 3. Outputs Ready ---")
    

    # --- A. Check Tabular Data (XGBoost) ---
    print(f"Regression DataFrame: {tabular_data.shape} rows")
    print("First 5 rows of Tabular Data:")
    pd.set_option('display.max_columns', None)
    
    # 1. Train Regression Model
    print("\n--- 3. Running Baseline Regression ---")
    regression_model = FloodLinearModel()
    metrics = regression_model.train(tabular_data)

    # --- NEW SECTION: Prepare for Diffusion ---
    print("\n--- 4. Preparing Data for Diffusion Model ---")

    # 1. Add XGBoost predictions to the dataframe
    # (This adds the 'xgb_pred' column)
    tabular_data = add_regression_pred(tabular_data, regression_model)
    # 2. Build the Tensor Dictionary
    diffusion_data = build_diffusion_tensors(
    target_data=tabular_data,
    scenario_col="target_name",
    grid_size=(64, 64),
    global_cols=["Category", "Speed", "Tide", "Direction_sin", "Direction_cos"]
)
    feature_cols = [
        "row", "col", 'latitude', 'longitude',           # Spatial coords            
        "elev_mean", 'elev_var',            # Elevation
        "Category", "Speed", "Tide", "Direction_sin", "Direction_cos" # Global Params
    ]
    
    check_regression_prediction(30, regression_model, diffusion_data, tabular_data, feature_cols)
 
    print("\n--- 4. Training Diffusion Model ---")

    
    # 1. Load Model AND r0_std
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #path = "models/diffusion/latest_diffusion.pt"
    #diffusion_model, r0_std = load_diffusion_model(path, device=device)
    
    
    train_ds, test_ds, cond_dim = prepare_diffusion_data(
        data_pack=diffusion_data, 
        save_dir="models/diffusion",
        train_split=0.8
    )
    
    # 2. Train from Scratch (100 Epochs)
    # By not passing a 'model' argument, the function initializes a new one
    diffusion_model = train_diffusion_model(
        train_dataset=train_ds,
        test_dataset=test_ds,
        cond_dim=cond_dim,
        model=None,               # Explicitly None to train from scratch
        save_dir="models/diffusion",
        epochs=50,               # Updated to 100 epochs
        batch_size=16,
        device=device
    )
    
    # Now you can pass 'model' to your generate_samples function
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
    
    labels = diffusion_data["scenario_names"] 
    
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
        regression_model=regression_model,
        target_data=tabular_data,
        feature_cols=feature_cols,   
        scenario_name=target_label,
        category=row["Category"],
        speed=row["Speed"],
        tide=row["Tide"],
        direction_deg=float(row.get("Direction_deg", row.get("Direction", 0.0))),
        device=device,
        miss_mask=miss_mask
    )
    categories = list(range(1, 9))  # 1 to 8
    results = []

    print(f"--- Generating sensitivity analysis for Category 1-8 ---")

    for cat in categories:
    
        # Run your existing function (Unchanged)
        # We ignore pred and res here since you only asked for x_hat
        x_hat_raw, _, _ = sample_one_scenario_from_globals_edm(
            edm=diffusion_model,
            regression_model=regression_model,
            target_data=tabular_data,
            feature_cols=feature_cols,   
            scenario_name=target_label,
            category=cat,            # Varying this
            speed=row["Speed"],
            tide=row["Tide"],
            direction_deg=float(row.get("Direction_deg", row.get("Direction", 0.0))),
            device=device,
            miss_mask=miss_mask
        )
        
        # Move to CPU and remove batch dims for plotting
        # Also apply the r0_std fix manually here if your function returns normalized values
        # x_hat_corrected = (x_hat_raw - pred) * r0_std + pred (if needed)
        # For now, we take the result as is:
        results.append(x_hat_raw.detach().cpu().numpy().squeeze())

    # 2. Plotting in a 2x4 Grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, cat in enumerate(categories):
        im = axes[i].imshow(results[i], cmap="viridis", origin="lower")
        axes[i].set_title(f"Category: {cat}")
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.suptitle(f"Sensitivity Analysis: Category Variation for {target_label}", fontsize=16)
    plt.tight_layout()

    plt.show()

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