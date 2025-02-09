"""
additional script to crop the large zarr file to a field extent to generate an example
RCM Project specific
"""


import geopandas as gpd
import xarray as xr
import numpy as np
import shapely
import pathlib




# Automatically set module root based on the installed package location
module_root = pathlib.Path(__file__).parent.resolve().parent


zarr_file = r"...\S1\GRD\xarray\cube_32632_FRIEN_10m_2017-2021_dB_pattern1D.zarr"

geojson_file = r"{}\tests\data\Fboundary\WG_32632_FRIEN_20190704_59_fid_boundary.geojson".format(module_root)
output_zarr = r"{}\tests\data\output_zarr\WG_32632_FRIEN_20190704_59.zarr".format(module_root)

CRS = '32632'


#geojson_file = r"{}\tests\data\Fboundary\WRg_32632_FRIEN_20200902_59_fid_boundary.geojson".format(module_root)
#output_zarr = r"{}\tests\data\output_zarr\WRg_32632_FRIEN_20200902_59.zarr".format(module_root)

#CRS = '32632'



#zarr_file = r"...\S1\GRD\xarray\cube_32633_DEMM_10m_2017-2021_dB_pattern1D.zarr"

#geojson_file = r"{}\tests\data\Fboundary\ZwFr_32633_MRKN_20210428_A6070_fid_boundary.geojson".format(module_root)
#output_zarr = r"{}\tests\data\output_zarr\ZwFr_32633_MRKN_20210428_A6070.zarr".format(module_root)

#CRS = '32633'



slice_from = "2019-06-28"
slice_to = "2019-07-10"

#slice_from = "2020-08-28"
#slice_to = "2020-09-10"

#slice_from = "2021-04-20"
#slice_to = "2021-05-05"



# Read GeoJSON and Convert WKT to Shapely
gdf = gpd.read_file(geojson_file, engine="fiona")

# Convert WKT geometries to Shapely objects
if "geom" in gdf.columns:  # Ensure 'geom' column exists
    gdf["geometry"] = gdf["geom"].apply(shapely.wkt.loads)

# Drop invalid geometries
gdf = gdf[gdf["geometry"].notnull()]

# Merge multiple geometries into one (if needed)
merged_geom = gdf.unary_union  # Creates a single MultiPolygon/Polygon

# Open the Zarr dataset
ds = xr.open_zarr(zarr_file)


# Select the variables to extract
selected_variables = ["VV", "VH", "fid_mask"]  # List of variables to keep
ds = ds[selected_variables]  # Select only these variables

# Detect coordinate names dynamically
ds_dims = list(ds.dims)
y_name = next((dim for dim in ["y", "northing"] if dim in ds_dims), None)
x_name = next((dim for dim in ["x", "easting"] if dim in ds_dims), None)

if not x_name or not y_name:
    raise ValueError(f"Could not detect spatial dimensions (x/y) in dataset. Found: {ds_dims}")

print(f"Using '{x_name}' for x-coordinates and '{y_name}' for y-coordinates.")

# Spatially Crop the Zarr File Based on Bounding Box
minx, miny, maxx, maxy = merged_geom.bounds  # Extract bounding box

# Crop using bounding box first (faster)
ds_cropped = ds.sel({x_name: slice(minx, maxx), y_name: slice(maxy, miny)})  # Y-axis flip

print(ds)

# Apply Exact Geometry Mask (Optional)
ds_cropped = ds_cropped.rio.write_crs(f"EPSG:{CRS}")  # Assign correct CRS
ds_masked = ds_cropped.rio.clip([merged_geom], ds_cropped.rio.crs, drop=True)

# Filter Data by Time
time_name = "time"
ds_masked[time_name] = np.array(ds_masked[time_name], dtype="datetime64[ns]")  # Ensure datetime format
ds_xx = ds_masked.sel({time_name: slice(slice_from, slice_to)})  # Select only 2021 data

print(ds_xx)

ds_xx = ds_xx.chunk({"time": -1, "x": "auto", "y": "auto"})  # Adjust chunking dynamically
# Save the filtered dataset to disk as a new Zarr file
ds_xx.to_zarr(output_zarr, mode="w", safe_chunks=True)

print(f"Filtered dataset saved to: {output_zarr}")