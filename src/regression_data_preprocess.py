import re
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, box
from rasterio.features import geometry_mask

# --- 1. Clipping Logic ---
def clip_to_roi(gdf):
    """Clips global GDF to the specific Texas ROI to reduce processing time."""
    min_lon, min_lat = -95.8, 28.8
    max_lon, max_lat = -94.8, 30.0
    roi_box = box(min_lon, min_lat, max_lon, max_lat)
    gdf_clip = gpd.clip(gdf, roi_box)
    if gdf_clip.empty:
        print("⚠️ WARNING: Clipped GDF is empty!")
    return gdf_clip

# --- 2. Stats Calculation ---
def add_elevation_stats(gdf, dem_data, transform):
    print("   -> Calculating elevation stats (Mean & Variance)...")
    elev_means = []
    elev_vars = []

    for geom in gdf.geometry:
        # all_touched=True ensures we catch pixels even for small grid cells
        mask = geometry_mask(
            [geom],
            transform=transform,
            out_shape=dem_data.shape,
            invert=True,
            all_touched=True 
        )
        
        vals = dem_data[mask]
        
        if vals.size == 0:
            # If the cell is completely outside the raster (rare given your bounds)
            elev_means.append(np.nan)
            elev_vars.append(np.nan)
        else:
            # Calculate Mean and Variance ignoring NaNs in the source data
            elev_means.append(np.nanmean(vals))
            elev_vars.append(np.nanvar(vals))

    gdf["elev_mean"] = elev_means
    gdf["elev_var"] = elev_vars
    
    return gdf
# --- 3. Rescaling (Using YOUR specific constants) ---
def rescale_geometries(gdf, substation):
    print("   -> Rescaling geometries using fixed reference points...")
    
    # CONSTANTS
    lon1 = -94.8
    lat1 = 28.9
    lon2 = -95.7
    lat2 = 29.8
    
    def transform_coords(x, y):
        # Your Formula:
        ax = 1 + 2 * (x - lon2)**3 / lon2
        by = (1 + 2 * (y - lat1)**3 / lat1) * (lon1 - lon2) / (lat2 - lat1)
        return (ax * x, by * y)

    def rescale_polygon(poly):
        if poly.is_empty: return poly
        new_coords = [transform_coords(x, y) for x, y in poly.exterior.coords]
        return Polygon(new_coords)

    # 1. Apply to Polygons
    gdf_scaled = gdf.copy()
    gdf_scaled["geometry"] = gdf.geometry.apply(rescale_polygon)

    # 2. Apply to Substations
    sub = substation.copy()
    
    sub["Longitude_scaled"] = sub.apply(lambda row: 
        (1 + 2*(row["Longitude"] - lon2)**3 / lon2) * row["Longitude"], axis=1)
        
    sub["Latitude_scaled"] = sub.apply(lambda row: 
        (1 + 2*(row["Latitude"] - lat1)**3 / lat1) * row["Latitude"] * (lon1 - lon2) / (lat2 - lat1), axis=1)
    
    gsubs = gpd.GeoDataFrame(
        sub,
        geometry=gpd.points_from_xy(sub["Longitude_scaled"], sub["Latitude_scaled"]),
        crs=gdf.crs
    )
    
    return gdf_scaled, gsubs

# --- 4. Rasterization (Preserves NaNs) ---
def rasterize_to_grid(gdf_scaled, n=64):
    print(f"   -> Rasterizing to {n}x{n} grid...")
    
    # Use Projected CRS (Pseudo-Mercator) for grid
    g = gdf_scaled.to_crs(3857)
    minx, miny, maxx, maxy = g.total_bounds
    
    xs = np.linspace(minx, maxx, n + 1)
    ys = np.linspace(miny, maxy, n + 1)
    
    cells, cell_ids = [], []
    cid = 0
    for j in range(n):
        for i in range(n):
            cells.append(box(xs[i], ys[j], xs[i+1], ys[j+1]))
            cell_ids.append(cid)
            cid += 1
            
    grid = gpd.GeoDataFrame({"cell_id": cell_ids}, geometry=cells, crs=g.crs)
    
    value_cols = [c for c in gdf_scaled.columns if c not in ['geometry', 'id', 'Name']]
    left = gpd.GeoDataFrame(g[value_cols], geometry=g.geometry, crs=g.crs)
    right = gpd.GeoDataFrame(grid[["cell_id"]], geometry=grid.geometry, crs=grid.crs)
    
    inter = gpd.overlay(left, right, how="intersection")
    inter["part_area"] = inter.area
    
    agg = {}
    for c in value_cols:
        if not pd.api.types.is_numeric_dtype(inter[c]): continue
        valid = inter[inter[c].notna()]
        if valid.empty: continue
        
        numerator = (valid[c] * valid["part_area"]).groupby(valid["cell_id"]).sum()
        denominator = valid.groupby("cell_id")["part_area"].sum()
        agg[c] = numerator / denominator

    agg_df = gpd.GeoDataFrame(agg)
    agg_df.index.name = "cell_id"
    
    grid_agg = grid.merge(agg_df, on="cell_id", how="left")
    
    final_cols = list(agg.keys())
    
    # Initialize with NaN 
    '''
    images = np.full((len(final_cols), n, n), np.nan, dtype=np.float32)
    
    for c_idx, col in enumerate(final_cols):
        for cid, val in zip(grid_agg["cell_id"], grid_agg[col]):
            # --- FIX IS HERE ---
            # Old: iy = n - 1 - (cid // n)  <-- This inverted it (Image Coordinates)
            # New: iy = cid // n            <-- This matches your Cartesian Grid (Bottom=0)
            iy = cid // n
            ix = cid % n
            images[c_idx, iy, ix] = val
            
    return images, final_cols, grid_agg
    '''
    return grid_agg

# --- 5. Tabular Dataset Creation ---
def create_tabular_dataset(grid_agg):
    print("   -> Creating tabular dataset...")
    
    pattern = re.compile(r"([a-z]+)(\d)(\d{2})([a-z]\d)")
    base_cols = ["cell_id", "geometry", "elev_mean", "elev_var"]
    present_base = [c for c in base_cols if c in grid_agg.columns]
    target_cols = [c for c in grid_agg.columns if pattern.fullmatch(c)]

    rows = []
    for col in target_cols:
        m = pattern.fullmatch(col)
        direction, category, speed, tide = m.groups()

        tmp = grid_agg[present_base].copy()
        tmp["target_name"] = col
        tmp["target_value"] = grid_agg[col] 
        
        tmp["Direction"] = direction
        tmp["Category"] = int(category)
        tmp["Speed"] = int(speed)
        tmp["Tide"] = 1 if tide == "i2" else 0
        
        rows.append(tmp)

    if not rows:
        return pd.DataFrame()

    target_data = pd.concat(rows, ignore_index=True)
    
    target_data["latitude"]  = target_data.geometry.centroid.y
    target_data["longitude"] = target_data.geometry.centroid.x
    
    direction_to_deg = {
        "n": 0.0, "nne": 22.5, "ne": 45.0, "ene": 67.5,
        "e": 90.0, "ese": 112.5, "se": 135.0, "sse": 157.5,
        "s": 180.0, "ssw": 202.5, "sw": 225.0, "wsw": 247.5,
        "w": 270.0, "wnw": 292.5, "nw": 315.0, "nnw": 337.5,
    }
    
    target_data["Direction_deg"] = target_data["Direction"].map(direction_to_deg).astype(float)
    radians = np.deg2rad(target_data["Direction_deg"])
    target_data["Direction_sin"] = np.sin(radians)
    target_data["Direction_cos"] = np.cos(radians)
    
    return target_data

# --- Master Pipeline ---
def preprocess_data(gdf, substation, dem_data, dem_transform):
    # 1. Clip
    gdf_clip = clip_to_roi(gdf)
    
    # 2. Stats
    gdf_clip = add_elevation_stats(gdf_clip, dem_data, dem_transform)
    
    # 3. Rescale
    gdf_scaled, subs_scaled = rescale_geometries(gdf_clip, substation)
    
    # 4. Rasterize (Unpacks 3 values correctly now!)
    #images, feature_names, grid_agg = rasterize_to_grid(gdf_scaled, n=64)
    grid_agg = rasterize_to_grid(gdf_scaled, n=64)
    # 5. Tabular
    tabular_data = create_tabular_dataset(grid_agg)
    
    return tabular_data


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- The PyTorch Model Definition ---
class SmallCNN(nn.Module):
    def __init__(self, input_dim):
        super(SmallCNN, self).__init__()
        # 1D CNNs expect input shape: (Batch, Channels, Length)
        # We have 1 channel, and 'input_dim' is our sequence length
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(8 * input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (Batch, Features) -> reshape to (Batch, 1, Features)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        return self.fc_layers(x)

# --- The Wrapper Class (Matches your FloodLinearModel) ---
class FloodCNNModel:
    def __init__(self):
        self.feature_cols = [
            "elev_mean", "elev_var", "Category", "Speed", 
            "Tide", "Direction_sin", "Direction_cos", 
            "latitude", "longitude"
        ]
        self.scaler = StandardScaler()
        self.model = SmallCNN(len(self.feature_cols))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, df, epochs=100, batch_size=32, lr=0.001):
        print("\n--- Training 1D-CNN Model ---")
        
        # 1. Handle NaN values (Exactly like your linear model)
        data = df[self.feature_cols + ["target_value"]].dropna().copy()
        X = data[self.feature_cols]
        y = data["target_value"].values.reshape(-1, 1)
        
        # 2. Split & Scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 3. Convert to Tensors
        train_ds = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # 4. Optimizer and Loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # 5. Training Loop
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # 6. Evaluate
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(torch.FloatTensor(X_train_scaled).to(self.device)).cpu().numpy()
            test_pred = self.model(torch.FloatTensor(X_test_scaled).to(self.device)).cpu().numpy()
        
        train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
        test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Test R²: {test_r2:.4f}")
        return {"train_rmse": train_rmse, "test_rmse": test_rmse, "test_r2": test_r2}

    def predict(self, X_new):
        self.model.eval()
        X_subset = X_new[self.feature_cols]
        X_scaled = self.scaler.transform(X_subset)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy()
        return preds.flatten()

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # We save the whole object; note that joblib works for PyTorch objects 
        # but torch.save is preferred for the weights alone. For consistency, we'll use joblib.
        joblib.dump(self, filepath)
        print(f"✅ CNN Model saved to: {filepath}")

    @staticmethod
    def load(filepath):
        model = joblib.load(filepath)
        print(f"✅ CNN Model loaded from: {filepath}")
        return model