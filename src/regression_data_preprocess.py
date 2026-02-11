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
    print("   -> Calculating elevation stats...")
    elev_means = []
    elev_vars = []

    for geom in gdf.geometry:
        mask = geometry_mask([geom], transform=transform, out_shape=dem_data.shape, invert=True)
        vals = dem_data[mask]
        
        if vals.size == 0:
            elev_means.append(np.nan)
            elev_vars.append(np.nan)
        else:
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
    images, feature_names, grid_agg = rasterize_to_grid(gdf_scaled, n=64)
    
    # 5. Tabular
    tabular_data = create_tabular_dataset(grid_agg)
    
    return images, tabular_data