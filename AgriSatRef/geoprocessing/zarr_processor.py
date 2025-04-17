import xarray
import geopandas as gpd
import pathlib
import rasterio
from shapely.geometry import Polygon
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZarrProcessor:
    """
    A class to handle operations related to Zarr datasets, including loading, reprojection,
    and updating datasets with new mask data. Also includes functionality to convert raster grids to vectors.
    """

    def __init__(self, zarr_file):
        """
        Initializes the ZarrProcessor with a given Zarr file.

        Parameters:
            zarr_file (str or pathlib.Path): Path to the Zarr file.
        """
        self.zarr_file = pathlib.Path(zarr_file)
        self.dataset = self.load_zarr()

    def load_zarr(self):
        """
        Loads the Zarr file into an xarray dataset.

        Returns:
            xarray.Dataset: The loaded xarray dataset.

        Raises:
            FileNotFoundError: If the specified Zarr file does not exist.
        """
        if not self.zarr_file.exists():
            raise FileNotFoundError(f"The specified Zarr file does not exist: {self.zarr_file}")
        return xarray.open_zarr(self.zarr_file.as_posix(), consolidated=True)
    


    def save_raster_tiff(self, raster_array, output_filename, epsg=None):
        """
        Saves a 2D raster array as a GeoTIFF file.

        Parameters:
            raster_array (xarray.DataArray): The 2D raster array to be saved.
            output_filename (str or pathlib.Path): Path where the output TIFF file will be saved.
            epsg (int, optional): EPSG code for the coordinate reference system. Default is None.
        """
        # Get the shape, resolution, and bounds of the raster array
        shape = raster_array.shape
        resolution = raster_array.rio.resolution()
        bounds = raster_array.rio.bounds()
        crs = raster_array.rio.crs

        # Calculate the transform
        transform = rasterio.transform.from_origin(bounds[0], bounds[3], resolution[0], abs(resolution[1]))

        # Set the CRS
        if epsg is not None:
            crs = f'EPSG:{epsg}'

        # Define the metadata for the output TIFF
        metadata = {
            'driver': 'GTiff',
            'height': shape[0],
            'width': shape[1],
            'count': 1,  # Number of bands
            'dtype': raster_array.dtype,
            'crs': crs,
            'transform': transform
        }

        # Write the raster array to a GeoTIFF file
        print("##########################")
        print(output_filename)
        print(shape)
        print(resolution)
        print(crs)
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(raster_array.values, 1)  # Write the first band

        print(f"Raster TIFF file written to: {output_filename}")


    def process_fid_list(self, fid_list, xroot, variable_name, AOI, output_format="gpkg", mask_variable=None, filname_tail='_grid'):
        """
        Processes the Zarr dataset for each FID, converts the raster grid to vector format, and saves it as a file.
        This version is flexible to handle different variable names and masks.

        Generated vector file header will have the format: `{AOI}_{fid_list}_{filename_tail}.{output_format}`
        Example: `FRIEN_[4]_S1_grid.gpkg`
        Parameters:
            fid_list (list): List of feature IDs (FIDs) to process.
            xroot (str or pathlib.Path): Path to the directory where the xarray is stored to be used as an dummy for a
            grid retrieval and the generated vector file will be stored.
            variable_name (str): The name of the xarray variable (e.g., 'VV') to process from the dataset.
            AOI (str): Area of Interest identifier.
            output_format (str): The output format for the vector file (default: 'gpkg').
            mask_variable (str, optional): The name of the mask variable (e.g., 'fid_mask', 'fid'). If None, no mask is applied.
            filname_tail (str, optional): A custom suffix for the filename (default is '_grid').
        """
        # Step 1: Load the target data variable (e.g., 'VV')
        if variable_name in self.dataset:
            data_var = self.dataset[variable_name].mean(dim='time').compute()
        else:
            raise KeyError(f"Variable '{variable_name}' not found in the dataset.")

        # Step 2: If a mask variable is provided, check if it exists and apply the mask
        if mask_variable:
            if mask_variable in self.dataset:
                logger.info(f"Applying mask '{mask_variable}' for FID filtering.")
                print(data_var)
                data_var = data_var.where(self.dataset[mask_variable].isin(fid_list).compute(), drop=True)
            else:
                raise KeyError(f"Mask variable '{mask_variable}' not found in the dataset.")
        else:
            logger.info("No mask variable provided. Processing the entire dataset without filtering.")

        # Step 3: Process each FID in the list
        for fid in fid_list:
            fid = [fid]  # Ensure it's a list for the where() operation

            logger.info(f"Processing FID: {fid}")

            # Step 4: Create a temporary directory to store intermediate files
            temp_dir = Path(xroot).joinpath(filname_tail)
            temp_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created temporary directory: {temp_dir}")

            # Step 5: Convert raster grid to vector format and save the file
            if output_format == 'tif':
                try:
                    self.save_raster_tiff(
                        raster_array=data_var,
                        output_filename=temp_dir.joinpath(f'{AOI}_{fid}_{filname_tail}.{output_format}'),
                        epsg=None  # Optional: Change this to a valid EPSG code if needed
                    )
                except Exception as e:
                    logger.error(f'Error converting raster grid to raster: {e}')
                    pass
            else:
                try:
                    self.rastergrid2vector(
                        raster_array=data_var,
                        vector_filename=temp_dir.joinpath(f'{AOI}_{fid}_{filname_tail}.{output_format}'),
                        epsg=None  # Optional: Change this to a valid EPSG code if needed
                    )
                except Exception as e:
                    logger.error(f'Error converting raster grid to vector: {e}')
                    pass

            # Step 6: Clean up and close datasets
            data_var.close()
            self.dataset.close()

            if output_format == 'tif':
                raster_file = temp_dir.joinpath(f'{AOI}_{fid}_{filname_tail}.tif')
                with rasterio.open(raster_file) as dataset:
                    logger.info(f"Generated raster dataset for FID: {fid} - Type: {type(dataset)}")
            else:
                # Step 7: Verify the vector file has been generated by reading it
                vector_file = temp_dir.joinpath(f'{AOI}_{fid}_{filname_tail}.{output_format}')
                grid = gpd.read_file(vector_file)
                logger.info(f"Generated vector grid for FID: {fid} - Type: {type(grid)}")

    def rastergrid2vector(self, raster_array, vector_filename, epsg=None):
        """
        Converts the raster grid to vector format and saves it as a file.

        Parameters:
            raster_array (xarray.DataArray): The 2D raster array to be converted.
            vector_filename (str or pathlib.Path): Path where the output vector file will be saved.
            epsg (int, optional): EPSG code for the coordinate reference system. Default is None.
        """
        shape = raster_array.shape
        resolution = raster_array.rio.resolution()
        bounds = raster_array.rio.bounds()
        crs = raster_array.rio.crs

        length = resolution[0]
        width = abs(resolution[1])

        logger.info(f'Raster array shape: {shape}')
        logger.info(f'Raster array bounds: {bounds}')
        logger.info(f'CRS: {crs}')
        logger.info(f'Resolution: length={length}, width={width}')

        # Create the grid polygons
        cols = np.arange(bounds[0], bounds[2] + width, width)
        rows = np.arange(bounds[1], bounds[3] + length, length)
        rows = rows[::-1]  # Reverse rows so they start from the top

        polygons = []
        for x in cols:
            for y in rows:
                polygons.append(Polygon([(x, y), (x + width, y), (x + width, y - length), (x, y - length)]))

        grid = gpd.GeoDataFrame({'geometry': polygons})

        if epsg is not None:
            grid = grid.set_crs(f'epsg:{epsg}')
        else:
            grid = grid.set_crs(crs)

        # Save to the specified vector format
        grid.to_file(vector_filename)
        logger.info(f"Vector file written to: {vector_filename}")

    def process_zarr_data(self, crs_object=None, write_to_disk=None):
        """
        Processes the Zarr dataset by checking its CRS and reprojecting if necessary. Optionally writes the result to disk.

        Parameters:
            crs_object (rasterio.crs.CRS, optional): The target CRS as a rasterio CRS object.
                                                     If None, no reprojection is done.
            write_to_disk (str or pathlib.Path, optional): Directory path to save the reprojected dataset.
                                                           If None, no file is written to disk.

        Returns:
            xarray.Dataset: The processed (and possibly reprojected) xarray dataset.
        """
        # If no CRS object is provided, return the original dataset without reprojection
        if crs_object is None:
            logger.info("No CRS object provided. Returning the dataset without reprojection.")
            return self.dataset

        # If the dataset CRS doesn't match the target CRS, reproject the dataset
        if self.dataset.rio.crs != crs_object:
            logger.info(f"Reprojecting Zarr dataset from {self.dataset.rio.crs} to {crs_object}")
            reprojected_dataset = self.dataset.rio.reproject(f"EPSG:{crs_object.to_epsg()}")

            # If a path is provided, write the dataset to disk
            if write_to_disk:
                write_to_disk = pathlib.Path(write_to_disk)  # Ensure it's a Path object
                write_to_disk.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

                # Construct the file name with CRS info (e.g., dataset_4326.zarr)
                file_name = f"dataset_{crs_object.to_epsg()}.zarr"
                output_file = write_to_disk / file_name

                logger.info(f"Writing reprojected dataset to disk at {output_file}")
                reprojected_dataset.to_zarr(output_file.as_posix(), mode='w', consolidated=True)

            return reprojected_dataset

        # If the CRS matches, return the original dataset
        logger.info("CRS matches the expected CRS. No reprojection needed.")
        return self.dataset

    def update_zarr_with_masks(self, ref_dataset_file, combined_data_reduced, variable_name, output_crs): # ref_dataset_file is the path to the endproduct of created poi coverage in zarr format
        """
        Updates or creates a Zarr file with new mask data. If the file already exists, it updates the existing data.
        Otherwise, it creates a new Zarr file.

        Parameters:
            ref_dataset_file (pathlib.Path): Path to the mask file.
            combined_data_reduced xarray.Dataset): The new mask data to be added to the Zarr dataset.
            variable_name (str): The variable name for the mask data.
            output_crs (str): The target CRS for the mask data in EPSG format.
        """
        ref_dataset_file = pathlib.Path(ref_dataset_file)
        if ref_dataset_file.exists():
            logger.info("Mask file exists. Updating the existing Zarr file.")
            self.update_existing_zarr_file(ref_dataset_file, combined_data_reduced, output_crs, variable_name)
        else:
            logger.info("Mask file does not exist. Creating a new Zarr dataset.")
            self.create_new_zarr_file(ref_dataset_file, combined_data_reduced, output_crs, variable_name)

    def create_new_zarr_file(self, ref_dataset_file, data_reduced, output_crs, variable_name):
        """
        Creates a new Zarr file and writes the mask data into it.

        Parameters:
            ref_dataset_file (pathlib.Path): Path to the mask file.
            data_reduced (xarray.Dataset): The reduced mask data.
            output_crs (str): The target CRS for the mask data.
            variable_name (str): The variable name for the mask data.
        """
        print(f"Creating new Zarr dataset with variable {variable_name}.")
        cube = xarray.Dataset()
        cube.rio.write_crs(f'EPSG:{output_crs}', inplace=True)

        if data_reduced.rio.crs != cube.rio.crs:
            data_reduced = self.reproject(data_reduced, output_crs, cube.rio.crs)
        cube[variable_name] = data_reduced

        cube.to_zarr(ref_dataset_file.as_posix(), mode='a', consolidated=True)

    def update_existing_zarr_file(self, ref_dataset_file, data_reduced, output_crs, variable_name):
        """
        Updates an existing Zarr file with new mask data by aligning and combining the new data with the existing dataset.

        Parameters:
            ref_dataset_file (pathlib.Path): Path to the mask file.
            data_reduced xarray.Dataset): The new mask data.
            output_crs (str): The target CRS for the mask data.
            variable_name (str): The variable name for the mask data.
        """
        print(f"Updating existing Zarr dataset with variable {variable_name}.")
        with xarray.open_zarr(ref_dataset_file.as_posix(), consolidated=True) as cube:
            cube.load()

            if variable_name in cube.data_vars:
                print(f"The dataset contains a variable named '{variable_name}'. Updating existing data.")
                if cube.rio.crs != data_reduced.rio.crs:
                    data_reduced = self.reproject(data_reduced, output_crs, cube.rio.crs)

                updated_variable = self.update_dataset(cube, data_reduced, variable_name)
                cube[variable_name] = updated_variable
            else:
                print(f"The dataset does not contain a variable named '{variable_name}'. Adding new data.")
                cube[variable_name] = data_reduced

            cube.to_zarr(ref_dataset_file.as_posix(), mode='w', consolidated=True)

    def update_dataset(self, cube, new_data, variable_name):
        """
        Aligns datasets and updates the variable with combined data.

        Parameters:
            cube (xarray.Dataset): The existing dataset.
            new_data (xarray.Dataset): The new data to be combined.
            variable_name (str): The variable name for the mask data.

        Returns:
            xarray.DataArray: The combined dataset.
        """
        existing_variable = cube[variable_name]
        aligned_cube, aligned_existing = xarray.align(cube, existing_variable, join='outer')
        combined_data = aligned_existing.fillna(0) + new_data.fillna(0).astype(float)
        combined_data = combined_data.where(combined_data != 0, other=np.nan)
        return combined_data
    # decide whichreproject is to keep and wich delete
    '''
    def reproject(self, data, target_crs, source_crs):
        """
        Reprojects the data to the target CRS.

        Parameters:
            data (xarray.Dataset): The dataset to be reprojected.
            target_crs (str): The target CRS.
            source_crs (str): The source CRS.

        Returns:
            xarray.Dataset: The reprojected dataset.
        """
        print(f"Reprojecting data from {source_crs} to {target_crs}")
        return data.rio.reproject(target_crs)
    '''

    def reproject_zarr(self, crs_object):
        """
        Reprojects the Zarr dataset to a specified CRS.

        Parameters:
            crs_object (rasterio.crs.CRS): The target CRS as a rasterio CRS object.

        Returns:
            xarray.Dataset: The reprojected dataset.
        """
        epsg_code = crs_object.to_epsg()  # Extract EPSG code from the CRS object
        return self.dataset.rio.reproject(f"EPSG:{epsg_code}")

    def reproject_and_save_to_zarr(self, output_zarr_path, from_to=[4326, 32633]):
        """
        Reproject a dataset from EPSG:4326 to EPSG:32633 and save it in Zarr format.

        Parameters:
        - dataset_path (str): Path to the input dataset in EPSG:4326.
        - output_zarr_path (str): Path to save the reprojected dataset in Zarr format.
        """

        print(self.dataset.rio.crs)  # Check CRS
        print(self.dataset.coords)  # Check coordinates

        # Check if the dataset has CRS set to EPSG:4326
        if self.dataset.rio.crs is None:
            # If CRS is not set, manually set it to EPSG:4326
            dataset = self.dataset.rio.write_crs(f"EPSG:{from_to[0]}", inplace=True)

        print("Original CRS:", dataset.rio.crs)


        # Reproject the dataset to EPSG:32633
        dataset_utm = dataset.rio.reproject(f"EPSG:{from_to[1]}")

        # Verify the new CRS
        print("Reprojected CRS:", dataset_utm.rio.crs)

        # Save the reprojected dataset to Zarr
        dataset_utm.to_zarr(output_zarr_path, mode='w')

        print(f"Reprojected dataset saved to {output_zarr_path}")

    def close(self):
        return self.dataset.close()
