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
    pred_hw = np.flipud(to_hw(pred))
    xhat_hw = np.flipud(to_hw(x_hat))
    
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
    im0 = axes[0, 0].imshow(orig_hw, origin="upper", cmap="viridis")
    axes[0, 0].set_title("Original (Ground Truth)")
    axes[0, 0].axis("off")
    add_cb(im0, axes[0, 0])

    # Row 1: XGBoost
    im1 = axes[1, 0].imshow(pred_hw, origin="upper", cmap="viridis")
    axes[1, 0].set_title("XGBoost Prediction")
    axes[1, 0].axis("off")
    add_cb(im1, axes[1, 0])

    im2 = axes[1, 1].imshow(res_pred, origin="upper", cmap="bwr", vmin=-max_val, vmax=max_val)
    axes[1, 1].set_title("Residual: Original - XGBoost")
    axes[1, 1].axis("off")
    add_cb(im2, axes[1, 1])

    # Row 2: Diffusion
    im3 = axes[2, 0].imshow(xhat_hw, origin="upper", cmap="viridis")
    axes[2, 0].set_title("Diffusion Prediction")
    axes[2, 0].axis("off")
    add_cb(im3, axes[2, 0])

    im4 = axes[2, 1].imshow(res_diff, origin="upper", cmap="bwr", vmin=-max_val, vmax=max_val)
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