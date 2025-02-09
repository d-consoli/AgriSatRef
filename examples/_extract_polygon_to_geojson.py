"""
additional script to extract from SQlite DB the field shape Polygon and store it as geojson
RCM Project specific
"""

import pathlib
import re
import geopandas as gpd


from geoprocessing.geometry_utils import GeometryUtils
from dbflow.src import db_utility as dbu
import logging


# Automatically set module root based on the installed package location
module_root = pathlib.Path(__file__).parent.resolve().parent


#CROP_TYPE_CODE = 'ZwFr'
#s1grid = r'{}\tests\data\S1_grid\MRKN_[589]_S1_grid.gpkg'.format(module_root)
#folder_path = pathlib.Path(r'{}\tests\data\Classification\ZwFr'.format(module_root))


#CROP_TYPE_CODE = 'WRg'
#s1grid = r'{}\tests\data\S1_grid\MRKN_[55]_S1_grid.gpkg'.format(module_root)
#folder_path = pathlib.Path(r'{}\tests\data\Classification\WRg'.format(module_root))


CROP_TYPE_CODE = 'ZwFr'
folder_path = pathlib.Path(r'{}\tests\data\Classification\ZwFr'.format(module_root))


output_dir = pathlib.Path(r'{}\tests\data\output_{}'.format(module_root, CROP_TYPE_CODE))
db_path = r'{}\_DB\RCM_main.db'.format(module_root)

CRS = '32633'

# connect to DB
dbarchive = dbu.connect2db(db_path)



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

        CROP_TYPE_CODE = 'SM'

        sql_fid = dbu.create_sql(sql_file='_query_and_buffer_fid.sql', replacements={':CRS': str(CRS),
                                                                                     ':buff': str(buff),
                                                                                      ':CROP_TYPE_CODE': CROP_TYPE_CODE,
                                                                                      ':AOI': AOI,
                                                                                      ':sl_nr': sl_nr,
                                                                                      ':date': date[:4],
                                                                                      })

        gdr = dbu.query_sql(sql=sql_fid, db_engine=dbarchive.archive.engine)
        gdr = gdr.drop_duplicates()

        # Convert WKT strings to Shapely geometry objects, skipping invalid ones
        polygons = [GeometryUtils.safe_loads(wkt) for wkt in gdr['geom'] if GeometryUtils.safe_loads(wkt) is not None]

        # Ensure that 'gdf' is a valid GeoDataFrame with geometries and CRS
        gdr = gpd.GeoDataFrame(gdr, geometry=polygons, crs=f"EPSG:{CRS}")  # Replace with correct CRS if known

        # Convert to WKT
        gdr.to_file(r"{}\tests\data\Fboundary\{}_{}_{}_{}_{}_fid_boundary.geojson".format(module_root, CROP_TYPE_CODE,
                                                                                          CRS, AOI, date, sl_nr), driver="GeoJSON")