import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine

def load_meow_data():
    """Function to handle all data loading logic."""
    DATA_DIR = "data"
    
    # Load Shapefile
    shp_path = os.path.join(DATA_DIR, "gl3MEOWs.shp")
    if not os.path.exists(shp_path):
        raise FileNotFoundError(f"Cannot find {shp_path}. Check your data folder!")
        
    gdf = gpd.read_file(shp_path)
    gdf = gdf.replace(99.9, 0)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    lon1, lat1, lon2, lat2 = -94.8, 28.9, -95.7, 29.8  
    # bounding box: (min_lon, min_lat, max_lon, max_lat)

    bbox = box(lon1, lat1, lon2, lat2)
    # keep any polygons that touch the box, and clip them to the box
    gdf_clip = gpd.clip(gdf, gpd.GeoDataFrame(geometry=[bbox], crs="EPSG:4326"))
    

    
    return gdf_clip

def load_substation_data():

    DATA_DIR = "data"
    # Load CSV
    csv_path = os.path.join(DATA_DIR, "Final_Input1.csv")
    substation = pd.read_csv(csv_path)
    substation = substation.iloc[:, :9]

    return substation

def load_elevation_data():
    """
    Returns:
        data: Numpy array of elevation
        transform: Affine transform (to map lat/lon to array indices)
        crs: Coordinate reference system
    """
    DATA_DIR = "data"
    etopo_global = os.path.join(DATA_DIR, "ETOPO_2022_v1_60s_N90W180_bed.tif")
    
    # Your Area of Interest
    min_lon, min_lat = -95.8, 28.8
    max_lon, max_lat = -94.8, 30.0

    if not os.path.exists(etopo_global):
        raise FileNotFoundError(f"Missing ETOPO file: {etopo_global}")

    with rasterio.open(etopo_global) as src:
        window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=src.transform)
        data = src.read(1, window=window)
        # Get the transform for this specific window
        transform = src.window_transform(window) 
        crs = src.crs

    return data, transform, crs