"""
Step 3: In this part the reference dataset in gpkg format is converted to xarray object and save to disk as zarr file
"""

import pathlib
import logging
import sys

from io_handler.geopackage_handler import GeoPackageHandler
from converters.classification_converter import ClassificationDownscaler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
ref_directory = r'{}\tests\ref_dataset'.format(module_root)



CROP_TYPE_CODE = 'WRg'
CROP_TYPE_CODE_LIST = ['WRg']

output_dir = pathlib.Path(r'{}\tests\ref_dataset'.format(module_root))



for CROP_TYPE_ in CROP_TYPE_CODE_LIST:
    input_dir = output_dir.joinpath('_output_{}'.format(CROP_TYPE_))
    # Check if the directory exists
    if not input_dir.exists():
        print(f"Directory '{input_dir}' does not exist.")
        sys.exit()

    # Extract values from the dictionary and flatten them into a single list
    all_paterns_level_2 = [value for sublist in patterns_dict.values() for value in sublist]

    # Initialize the handler
    handler = GeoPackageHandler()

    # Get layer information from all GeoPackages in the folder
    all_layer_info_df = handler.get_layer_information_from_folder(input_dir, patterns=all_paterns_level_2)


    # Display the combined DataFrame
    print(all_layer_info_df['date'].unique())

    print(f"Files in '{input_dir}':")
    for file in input_dir.glob('*.gpkg'):
        print(file.name)

        xx, xx, crop_type_code, xx, CRS, AOI, year, bbch, xx, fid, xx = file.name.split('_')
        print('fid:', fid)

        CRS_out = '32632'

        zarr_file = pathlib.Path(xroot).joinpath(
            r'WG_{}_{}_20190704_59.zarr'.format(CRS_out, AOI))

        # ref_dataset_file= pathlib.Path(cfg.ref_directory).joinpath(UAVmask_{}.zarr'.format(fid))
        ref_dataset_file = pathlib.Path(ref_directory).joinpath(
            'UAVmask_mean_{}_{}_{}.zarr'.format(CRS_out, AOI,
                                                CROP_TYPE_CODE))


        # --------  Convert Classification to RefDataset
        for pattern_of_interest, PATTERN_list in patterns_dict.items():
            print('-' * 50)
            print(pattern_of_interest, PATTERN_list)
            print('-' * 50)

            # handler = GeoPackageHandler(file)
            # fid_df = handler.get_layer_information(patterns=PATTERN_list, separator="_")
            # print(fid_df.empty)

            print(zarr_file)

            converter = ClassificationDownscaler(
                patterns=PATTERN_list,
                pattern_of_interest=pattern_of_interest,
                # patterns_dict=patterns_dict,
                zarr_file=zarr_file,
                classification_file=file,
                ref_dataset_file=ref_dataset_file,
                crs=CRS_out,
                time_dimension_labels=all_layer_info_df['date'].unique(),
                # this should be at least the minimum of existing number of values for time dimenstion
                study_area=AOI,
                add_averaged_mask=False,
                set_nan_value=0,
                add_quality_control_mask=True
            )

            converter.convert()

    # --------  Add Quality Control Mask to RefDataset
    # Process and sum mask layers, add `total_mask` to Zarr file
    quality_control_mask = converter.generate_quality_control_layers(error_types=['sum_consistency',
                                                                                  'MAE', 'RMSE', 'FE', 'Z_score',
                                                                                  'total_sum'],
                                                                     calculation_types=['per_time_step',
                                                                                        'over_time']
                                                                     )
    if quality_control_mask is not None:
        logger.info("Processed and added 'quality_control_mask' to the Zarr file.")
    else:
        logger.warning("No mask layers to process.")