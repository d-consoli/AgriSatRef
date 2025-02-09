"""
Step 1b: Extract S1 grid from xarray object and write it as gpkg

requirement: your xarray object of coarse resolution zarr file should have a variable that represents field ID mask with
int values like in this case is "fid_mask" = 55, 589, 591
"""

import pathlib
from geoprocessing.zarr_processor import ZarrProcessor


# MRKN
fid_list = [589, 591]

# FRIEN
fid_list = [55]


CROP_TYPE_CODE = 'WRg'
AOI = 'FRIEN'
CRS = '32632'

# Automatically set module root based on the installed package location
module_root = pathlib.Path(__file__).parent.resolve().parent

xroot = r'{}\tests\data'.format(module_root)
zarr_file = r"{}\output_zarr\{}_{}_{}_20200902_59.zarr".format(xroot, CROP_TYPE_CODE, CRS, AOI)


processor = ZarrProcessor(zarr_file)
processor.process_fid_list(fid_list, xroot, "VV", AOI,
                           output_format="gpkg",
                           mask_variable="fid_mask",
                           filname_tail='S1_grid')


