import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import geopandas as gpd
import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(self, target_data, targets_to_plot, feature_cols):
        """
        Generates 3 side-by-side plots (True, Predicted, Residual) for specific targets.
        Uses a shared color scale for True/Predicted to make comparison easy.
        """
        print(f"\n--- Visualizing {len(targets_to_plot)} targets ---")

        # 1. Calculate Shared Scale (Global Min/Max)
        # We need this so 10 meters of water looks the same color in every plot
        all_true = []
        all_pred = []

        # Temporary loop to gather stats
        for t_name in targets_to_plot:
            subset = target_data[target_data["target_name"] == t_name].copy()
            subset = subset.dropna(subset=feature_cols + ["geometry"])
            
            if subset.empty:
                print(f"Skipping {t_name} (No valid data)")
                continue

            preds = self.predict(subset[feature_cols])
            all_true.extend(subset["target_value"].values)
            all_pred.extend(preds)

        if not all_true:
            print("No valid data found to plot.")
            return

        shared_vmin = min(np.min(all_true), np.min(all_pred))
        shared_vmax = max(np.max(all_true), np.max(all_pred))
        
        print(f"Shared Color Scale: Min={shared_vmin:.2f}, Max={shared_vmax:.2f}")

        # 2. Generate Plots
        for t_name in targets_to_plot:
            subset = target_data[target_data["target_name"] == t_name].copy()
            subset = subset.dropna(subset=feature_cols + ["geometry"])
            
            if subset.empty:
                continue

            # Predict
            subset["predicted"] = self.predict(subset[feature_cols])
            subset["residual"] = subset["target_value"] - subset["predicted"]

            # Check if it's a GeoDataFrame for mapping
            if not isinstance(subset, gpd.GeoDataFrame):
                print(f"Warning: Data for {t_name} is not a GeoDataFrame. Cannot map.")
                continue

            # Plotting
            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            
            # (A) TRUE
            subset.plot(column="target_value", ax=axes[0], cmap="viridis", 
                       vmin=shared_vmin, vmax=shared_vmax, legend=True)
            axes[0].set_title(f"TRUE: {t_name}")
            axes[0].axis("off")

            # (B) PREDICTED
            subset.plot(column="predicted", ax=axes[1], cmap="viridis", 
                       vmin=shared_vmin, vmax=shared_vmax, legend=True)
            axes[1].set_title(f"PREDICTED: {t_name}")
            axes[1].axis("off")

            # (C) RESIDUAL (Red = Underestimated, Blue = Overestimated)
            # We use a centered scale for residuals
            max_resid = max(abs(subset["residual"].min()), abs(subset["residual"].max()))
            subset.plot(column="residual", ax=axes[2], cmap="coolwarm", 
                       vmin=-max_resid, vmax=max_resid, legend=True)
            axes[2].set_title("RESIDUAL (True - Pred)")
            axes[2].axis("off")

            plt.tight_layout()
            plt.show()


def to_hw(x):
    """Converts tensor/array to (H,W) numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    
    # Handle various shapes
    if x.ndim == 4:   # (1, 1, H, W)
        return x[0, 0]
    if x.ndim == 3:   # (1, H, W) or (C, H, W) where C=1
        return x[0]
    return x          # (H, W)

def plot_scenario_comparison(
    orig, 
    pred, 
    x_hat, 
    scenario_name="Scenario", 
    save_dir=None
):
    """
    Plots a 3x2 comparison: Original, XGBoost, Diffusion, and their residuals.
    Also plots histograms of the residuals.
    
    Args:
        orig: Ground truth tensor/array (H,W) or (1,H,W)
        pred: XGBoost prediction tensor/array
        x_hat: Diffusion prediction tensor/array
        scenario_name: Title for the plot
        save_dir: Optional directory to save the figures
    """
    
    # 1. Prepare Data & Align Orientations
    # Note: Applying flipud to predictions to match Original based on your snippet
    orig_hw = to_hw(orig)
    pred_hw = to_hw(pred)
    xhat_hw = to_hw(x_hat)
    
    # 2. Compute Residuals
    # Residual = Truth - Prediction
    res_pred = orig_hw - pred_hw
    res_diff = orig_hw - xhat_hw
    
    # Determine symmetric limits for colorbars (so 0 is always white/centered)
    # We find the max absolute deviation across BOTH methods for a fair comparison
    max_val = np.nanmax([np.abs(res_pred), np.abs(res_diff)])
    
    # ---------------------------
    # Figure 1: Spatial Maps
    # ---------------------------
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    plt.suptitle(f"Scenario: {scenario_name}", fontsize=16)

    # Turn off the top-right corner (empty slot)
    axes[0, 1].axis("off")

    # Helper for colorbars
    def add_cb(im, ax):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 0: Original
    im0 = axes[0, 0].imshow(orig_hw, origin="lower", cmap="viridis")
    axes[0, 0].set_title("Original (Ground Truth)")
    axes[0, 0].axis("off")
    add_cb(im0, axes[0, 0])

    # Row 1: XGBoost
    im1 = axes[1, 0].imshow(pred_hw, origin="lower", cmap="viridis")
    axes[1, 0].set_title("XGBoost Prediction")
    axes[1, 0].axis("off")
    add_cb(im1, axes[1, 0])

    im2 = axes[1, 1].imshow(res_pred, origin="lower", cmap="bwr", vmin=-max_val, vmax=max_val)
    axes[1, 1].set_title("Residual: Original - XGBoost")
    axes[1, 1].axis("off")
    add_cb(im2, axes[1, 1])

    # Row 2: Diffusion
    im3 = axes[2, 0].imshow(xhat_hw, origin="lower", cmap="viridis")
    axes[2, 0].set_title("Diffusion Prediction")
    axes[2, 0].axis("off")
    add_cb(im3, axes[2, 0])

    im4 = axes[2, 1].imshow(res_diff, origin="lower", cmap="bwr", vmin=-max_val, vmax=max_val)
    axes[2, 1].set_title("Residual: Original - Diffusion")
    axes[2, 1].axis("off")
    add_cb(im4, axes[2, 1])

    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/{scenario_name}_maps.png", dpi=150)
    
    plt.show()

    # ---------------------------
    # Figure 2: Histograms
    # ---------------------------
    # Flatten and remove NaNs
    res_pred_flat = res_pred[np.isfinite(res_pred)].ravel()
    res_diff_flat = res_diff[np.isfinite(res_diff)].ravel()

    # Shared X-axis range for fair comparison
    all_res = np.concatenate([res_pred_flat, res_diff_flat])
    if len(all_res) > 0:
        hist_range = (all_res.min(), all_res.max())
    else:
        hist_range = (-1, 1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    
    # XGB Histogram
    axes2[0].hist(res_pred_flat, bins=50, range=hist_range, color='skyblue', edgecolor='black', alpha=0.7)
    axes2[0].axvline(0, color='red', linestyle='dashed', linewidth=1)
    axes2[0].set_title(f"XGB Residuals (std={np.std(res_pred_flat):.3f})")
    axes2[0].set_xlabel("Error (m)")
    axes2[0].set_ylabel("Count")

    # Diffusion Histogram
    axes2[1].hist(res_diff_flat, bins=50, range=hist_range, color='salmon', edgecolor='black', alpha=0.7)
    axes2[1].axvline(0, color='red', linestyle='dashed', linewidth=1)
    axes2[1].set_title(f"Diffusion Residuals (std={np.std(res_diff_flat):.3f})")
    axes2[1].set_xlabel("Error (m)")
    axes2[1].set_ylabel("Count")

    plt.tight_layout()
    
    if save_dir:
        plt.savefig(f"{save_dir}/{scenario_name}_hist.png", dpi=150)
        
    plt.show()




def check_regression_prediction(
    index, 
    regression_model, 
    data_pack, 
    target_data, 
    feature_cols, 
    save_path=None
):
    """
    Plots Ground Truth vs. Regression Prediction for a specific index in data_pack.
    Handles NaNs by skipping prediction for invalid rows (masking).
    """
    # 1. Get Scenario Info
    # data_pack["names"] might be the key depending on your preprocess, 
    # but based on previous context checking for "names" or "scenario_names"
    scenario_name = data_pack["scenario_names"][index] 
    print(f"--- Checking Index {index}: {scenario_name} ---")

    # 2. Get Ground Truth (from the tensor pack)
    gt_tensor = data_pack["x0"][index]
    if isinstance(gt_tensor, torch.Tensor):
        gt_img = gt_tensor.detach().cpu().numpy().squeeze()
    else:
        gt_img = gt_tensor.squeeze()

    # 3. Get Regression Prediction (With NaN Masking)
    # Filter the big dataframe for this scenario
    df = target_data[target_data["target_name"] == scenario_name].copy()
    
    # Ensure sorting matches the 64x64 grid structure
    if "row" not in df.columns:
        df["row"] = np.arange(len(df)) // 64
        df["col"] = np.arange(len(df)) % 64
        
    df = df.sort_values(["row", "col"])
    
    X = df[feature_cols]

    # --- NAN HANDLING START ---
    # Identify rows where ANY feature is NaN
    valid_rows_mask = X.notna().all(axis=1)
    
    # Create a placeholder for predictions filled with NaNs
    pred_flat = np.full(len(df), np.nan)
    
    # Predict ONLY on the valid rows
    if valid_rows_mask.sum() > 0:
        X_clean = X[valid_rows_mask]
        # This will now work because X_clean has no NaNs
        pred_clean = regression_model.predict(X_clean)
        # Place predictions back into their original spots
        pred_flat[valid_rows_mask] = pred_clean
    else:
        print("⚠️ Warning: All input features for this scenario are NaN.")
    # --- NAN HANDLING END ---

    # Reshape to image
    xgb_img = pred_flat.reshape(64, 64)

    # 4. Calculate Residual (Error)
    # We only calculate error where BOTH Ground Truth AND Prediction are valid
    valid_comparison_mask = np.isfinite(gt_img) & np.isfinite(xgb_img)
    
    diff = np.full_like(gt_img, np.nan) # Start with all NaNs
    diff[valid_comparison_mask] = gt_img[valid_comparison_mask] - xgb_img[valid_comparison_mask]

    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground Truth
    im0 = axes[0].imshow(gt_img, cmap="viridis", origin="lower")
    axes[0].set_title(f"Ground Truth\n({scenario_name})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Prediction
    im1 = axes[1].imshow(xgb_img, cmap="viridis", origin="lower")
    axes[1].set_title("Regression Prediction\n(NaNs masked)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Residual
    # Use 'bwr' (blue-white-red) to show +/- errors clearly
    if np.any(valid_comparison_mask):
        limit = np.nanmax(np.abs(diff)) 
    else:
        limit = 1.0
        
    im2 = axes[2].imshow(diff, cmap="bwr", vmin=-limit, vmax=limit, origin="lower")
    axes[2].set_title("Residual\n(Truth - Pred)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

    # 6. Simple Stats
    if np.any(valid_comparison_mask):
        mse = np.mean(diff[valid_comparison_mask]**2)
        mae = np.mean(np.abs(diff[valid_comparison_mask]))
        print(f"Stats -> MSE: {mse:.5f} | MAE: {mae:.5f}")
    else:
        print("Warning: No valid pixels overlap between Truth and Prediction.")