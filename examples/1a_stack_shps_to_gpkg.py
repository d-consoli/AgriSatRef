"""
Step 1: Conversion of classification in shp file to gpkg layer (layer name -> bbch and date), stacking after FID, year,
crop type and  AOI (study area), includes also cropping after field shape

"""

import pathlib
import re
import geopandas as gpd
import os

from geoprocessing.geometry_utils import GeometryUtils
from dbflow.src import db_utility as dbu
import logging



config_path = dbu.get_custom_paths()["custom_db_structure"]
print(f"Resolved config.ini path: {config_path}")
print(f"Current working directory: {os.getcwd()}")


# Automatically set module root based on the installed package location
module_root = pathlib.Path(__file__).parent.resolve().parent


CROP_TYPE_CODE = 'ZwFr'

s1grid = r'{}\tests\data\S1_grid\MRKN_[589]_S1_grid.gpkg'.format(module_root)
folder_path = pathlib.Path(r'{}\tests\data\Classification\ZwFr'.format(module_root))


CROP_TYPE_CODE = 'WRg'

s1grid = r'{}\tests\data\S1_grid\MRKN_[55]_S1_grid.gpkg'.format(module_root)
folder_path = pathlib.Path(r'{}\tests\data\Classification\WRg'.format(module_root))


output_dir = pathlib.Path(r'{}\tests\data\_output_{}'.format(module_root, CROP_TYPE_CODE))
# Create the directory if it does not exist
output_dir.mkdir(parents=True, exist_ok=True)


# ----------- This is done to extract information to generate a gpkg the layer name and generate a gpkg file name with same information

# Regular expression pattern to match dates in the filename
date_pattern = r'(\d{4}\d{2}\d{2})'

# Dictionary to store file paths grouped by pattern
pattern_groups = {}

# Iterate through files in the folder
for file_path in folder_path.glob('*.shp'):
    print(f'file_path: {file_path}')

    # Extract the pattern from the file path
    sl_nr = file_path.name.split('_')[8]  # Assuming the pattern is before the first underscore

    # Add the file path to the corresponding pattern group
    if sl_nr in pattern_groups:
        pattern_groups[sl_nr].append(file_path)
    else:
        pattern_groups[sl_nr] = [file_path]

# Print the grouped file paths
for pattern, files in pattern_groups.items():
    print(f"Pattern: {pattern}")

    # List to store GeoDataFrames
    gdfs = {}
    for filepath in files:

        # Read the shapefile into a GeoDataFrame
        gdf = gpd.read_file(filepath)
        match = re.search(date_pattern, filepath.as_posix())
        date = match.group(1)

        # Convert to 'YYYY-MM-DD' format
        formatted_date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        sl_nr = filepath.name.split('_')[8]
        AOI = filepath.name.split('_')[5]
        CRS = filepath.name.split('_')[4]

        buff = '-30'
        bbch = 0

        # crate the name for a gpkg file
        geopackage_path = output_dir.joinpath('UAV_Valid_{}_xx_{}_{}_{}_bbch_{}_{}.gpkg'.format(
            CROP_TYPE_CODE, CRS, AOI, date[:4], 'xx', sl_nr))


        # Write each GeoDataFrame to the GeoPackage with the corresponding date as the layer name
        layer_name = f'{bbch}_{date}'

        #---------------------------------------------------------------------------------------------------------------
        gdr = gpd.read_file(r"{}\tests\data\Fboundary\{}_{}_{}_{}_{}_fid_boundary.geojson".format(module_root,
                                                                                                  CROP_TYPE_CODE,
                                                                                                  CRS,
                                                                                                  AOI,
                                                                                                  date,
                                                                                                  sl_nr
                                                                                                  ))

        # Convert WKT strings to Shapely geometry objects, skipping invalid ones
        polygons = [GeometryUtils.safe_loads(wkt) for wkt in gdr['geom'] if GeometryUtils.safe_loads(wkt) is not None]


        # Step 1: Initialize an empty list to store intersected results
        intersected_results = []

        # Step 2: Loop through each polygon in 'polygons' and perform intersection with filtering
        for polygon in polygons:
            # Filter 'gdf' to only keep geometries that intersect with the current polygon
            intersecting_gdf = gdf[gdf.geometry.intersects(polygon)].copy()

            # Apply intersection to retain only overlapping areas
            intersecting_gdf['geometry'] = intersecting_gdf.geometry.apply(lambda x: x.intersection(polygon))

        # Step 4: Check if the cropping was effective
        if len(intersecting_gdf) < len(gdf):
            print("cropped_gdf was cropped to retain only intersecting areas.")
        else:
            print("cropped_gdf was not cropped.")

        # Print the final result to confirm intersecting areas are retained
        print("Cropped GeoDataFrame with only intersecting areas:")
        # print(cropped_gdf)

        # Specify the output GeoPackage file path
        geopackage_path = output_dir.joinpath('UAV_Valid_{}_xx_{}_{}_{}_bbch_{}_{}.gpkg'.format(
            CROP_TYPE_CODE, CRS, AOI, date[:4], 'xx', sl_nr))
        intersecting_gdf.to_file(geopackage_path, layer=layer_name, driver="GPKG")
        print(geopackage_path)

