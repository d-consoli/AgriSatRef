"""
Step 2: Downscaling of the high resolution classification (UAV classiifcation at 0.050m) to a coarse resolution Grid
(Sentinel-1 at 10m) with zonal statistic, the result is stored as gpkg
"""

import pathlib
import pandas as pd
import geopandas as gpd

from io_handler.geopackage_handler import GeoPackageHandler
from dbflow.src import db_utility as dbu


# definition of pattern of interests
patterns_dict = {
    'bare_soil': ['bare_soil'],
    'undamaged_crop': ['vital_crop',
                       # 'ripening_crop', # only for Corn
                       'flowering_crop', 'dry_crop'
                       ],
    'lodged_crop': ['vital_lodged_crop', 'dry_lodged_crop'],
    'weed_infestation': ['weed_infestation']
}



# Automatically set module root based on the installed package location
module_root = pathlib.Path(__file__).parent.resolve().parent



xroot = r'{}\tests\data\output_zarr'.format(module_root)
ref_directory = r'{}\tests\data\ref_dataset'.format(module_root)


CROP_TYPE_CODE = 'WRg'
output_dir = pathlib.Path(r'{}\tests\data\_output_{}'.format(module_root, CROP_TYPE_CODE))


# this is RCM Project specific - extracting df with fid and sl_nr information
#db_path = r'{}\_DB\RCM_main.db'.format(module_root)

#dbarchive = dbu.connect2db(db_path)

#sql_fid_sl_nr = dbu.create_sql(sql_file='_query_fid_sl_nr.sql')
#dfr = dbu.query_sql(sql=sql_fid_sl_nr, db_engine=dbarchive.archive.engine)


# OR define your DF with fid and sl_nr information
dfr = pd.DataFrame( \
    {'fid': [55, 589, 591],
     'sl_nr': ['59-00', 'A6070-00', 'A6090-00']
    })



s1_grid_dir = pathlib.Path(r'{}\tests\data'.format(module_root)).joinpath('S1_grid')
temp_file = s1_grid_dir.joinpath('_temp.gpkg')



input_dir = output_dir

for ifile in input_dir.glob('**/*.gpkg'):

    print(f'ifile: {ifile}')

    fid_value = ifile.stem.split('_')[9]
    AOI = ifile.stem.split('_')[5]
    CRS = ifile.stem.split('_')[4]


    print(f'fid_value: {fid_value}')
    fid_values = dfr.loc[dfr['sl_nr'].str.startswith(fid_value + '-'), 'fid'].iloc
    print('fid_values: ', fid_values)


    # create an empty GeoDataFrame to collect data
    grd_df_list = []
    for j, fid in enumerate(fid_values):

        grid_file = pathlib.Path(r'{}\{}_[{}]_S1_grid.gpkg'.format(s1_grid_dir.as_posix(), AOI, fid))

        # Check if the path exists
        if not grid_file.exists():
            print(f"The path '{grid_file.as_posix()}' does not exist.")
            pass
        else:
            print(f"The path '{grid_file.as_posix()}' exists.")

            grd_df = gpd.read_file(grid_file)

            # Append the DataFrame to the list
            grd_df_list.append(grd_df)

    # Concatenate all DataFrames in the list
    merged_grd_df = gpd.GeoDataFrame(pd.concat(grd_df_list, ignore_index=True), crs=grd_df_list[0].crs)

    grid_file = temp_file
    merged_grd_df.to_file(grid_file.as_posix(), driver="GPKG")

    ref_gpkg = pathlib.Path(r'{}\tests\ref_dataset'.format(module_root))

    output_dir_ = ref_gpkg.joinpath('_output_{}'.format(CROP_TYPE_CODE))
    output_dir_.mkdir(parents=True, exist_ok=True)


    # Read GeoPackage file
    gdf = gpd.read_file(ifile)
    # layers = fiona.listlayers(ifile)

    handler = GeoPackageHandler(geopackage_file=ifile)
    layers = handler.get_available_layers()
    print(layers)

    for i_layer in layers:
        print(i_layer)

        if ifile.is_file():
            output_file = handler.calculate_poi_coverage_for_layer(
                ifile,
                indir=input_dir,
                outdir=output_dir_,
                lyr_name=i_layer,
                input_crs=CRS,
                z_vector=grid_file,
                fformat='gpkg',
                ofile=output_dir_.joinpath(ifile.stem + '_pattern.gpkg'),
                pattern=None,
                reproject_to_crs=None,
                tempfile=False
            )

            print(f"Generated dataset saved at: {output_file}")
    grid_file.unlink()
    grid_file = None
    merged_grd_df = None
