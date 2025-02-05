import pathlib
import numpy as np
import xarray as xr
import geopandas as gpd
import rasterio
from geoprocessing.zarr_processor import ZarrProcessor
from io_handler.geopackage_handler import GeoPackageHandler
import logging
from geocube.api.core import make_geocube

# Setup basic configuration for logging with a specified format
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# use this to write the logfile
# Configure logging to write to a file
#logging.basicConfig(level=logging.INFO,
#                    filename='application.log',  # Specify the file name
#                    filemode='a',  # Use filemode 'a' for append; 'w' for overwrite
#                    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


# Retrieve a logger for the specific module
logger = logging.getLogger(__name__)



class ClassificationDownscaler:
    """
    A class to handle the conversion of reference datasets (classification as a vector) from a GeoPackage format into a Zarr format,
    process mask files, and update the Zarr dataset accordingly.
    """

    def __init__(self, classification_file, zarr_file, ref_dataset_file, crs, patterns, pattern_of_interest,
                 time_dimension_labels, study_area='aoi', add_averaged_mask=False, add_quality_control_mask=True,
                 max_chunk_size=200, set_nan_value=0):
        """
        Initialize the ClassificationDownscaler by specifying files and parameters required for converting classification data
        to a reference dataset using an xarray structure.

        Parameters:
            classification_file (str): Path to the GeoPackage file containing classifications to be converted.
            zarr_file (str): Path to an existing Zarr file, which will be used as a template to define the xarray.Dataset
                            shape for the reference data. The dataset must have crs defined and bit in a Cartesian Coordinate System (examp. in UTM)
            ref_dataset_file (str): Path to the Zarr file to be created as the reference dataset.
            crs (int): Coordinate Reference System (CRS) to handle reprojection.
            patterns (list of str): List of patterns used to match layer names, defining the higher level of specification.
            pattern_of_interest (str): Specific pattern to focus on during processing, providing a lower level of specification.
            time_dimension_labels (list of str): List of unique dates or time steps for the dataset, representing all unique dates across classification files.
            study_area (str, optional): Identifier for the area of interest.
            add_averaged_mask (bool): If True, calculates an average mask (median over time for each pixel defined in pattern_of_interest). Defaults to False.
            add_quality_control_mask (bool): If True, a quality control mask is applied based on specified metrics. Defaults to False.
            max_chunk_size (int): Maximum chunk size allowed based on available memory, influencing performance and resource usage.
            set_nan_value (int): value to be set as NaN, Defaults to np.nan.

        """
        # TODO: make consistent default on UTM coordinates as 'x', 'y' and optional switsch to l'lon', 'lat' but with warning on error of calulation !!
        # TODO: calculateion to average mask does not work yet, the function was deleted, i gess! check in old version !!!
        self.classification_file = pathlib.Path(classification_file)
        self.ref_dataset_file = pathlib.Path(ref_dataset_file)
        self.crs = crs
        self.patterns = patterns
        self.pattern_of_interest = pattern_of_interest
        self.time_dimension_labels = time_dimension_labels
        self.study_area = study_area
        self.add_averaged_mask = add_averaged_mask
        self.add_quality_control_mask = add_quality_control_mask
        self.max_chunk_size = max_chunk_size
        self.set_nan_value = set_nan_value #or np.nan

        logger.info(f"Loading Zarr dataset from {zarr_file} to be used as a template to create an empty Dataset for Reference Dataset")
        self.zarr_data = xr.open_zarr(zarr_file, consolidated=False)

        # Extract the spatial shape from template zarr file
        self.spatial_shape = self.get_spatial_shape_from_dataset()

        # Load or create the mask file with appropriate chunking
        self.load_or_create_mask()


    def get_spatial_shape_from_dataset(self):
        """
        Extracts the spatial shape, specifically the size of the x and y dimensions (or longitude and latitude),
        from the Zarr dataset stored in the object. This method checks for common spatial dimension names
        ('x' and 'y' or 'lon' and 'lat') and retrieves their sizes.

        Returns:
            tuple: A tuple representing the shape of the spatial dimensions, ordered as (y_size, x_size),
                   where 'y_size' is the size of the latitude or 'y' dimension, and 'x_size' is the size
                   of the longitude or 'x' dimension.

        Raises:
            ValueError: If neither 'x' and 'y' dimensions nor 'lon' and 'lat' dimensions are found in the dataset.
        """
        logger.info("Attempting to extract spatial dimensions from the dataset.")
        if 'x' in self.zarr_data.dims and 'y' in self.zarr_data.dims:
            x_size = self.zarr_data.sizes['x']
            y_size = self.zarr_data.sizes['y']
            logger.info(f"Spatial dimensions found - x_size: {x_size}, y_size: {y_size}")
        elif 'lon' in self.zarr_data.dims and 'lat' in self.zarr_data.dims:
            x_size = self.zarr_data.sizes['lon']
            y_size = self.zarr_data.sizes['lat']
            logger.info(f"Spatial dimensions found - lon_size: {x_size}, lat_size: {y_size}")
        else:
            logger.error("Could not find x/y or lon/lat dimensions in the dataset.")
            raise ValueError("Could not find x/y or lon/lat dimensions in the dataset.")

        return (y_size, x_size)

    def load_or_create_mask(self, time_dimension_labels=None):
        """
        Loads an existing mask file or creates a new one if it does not exist, using the spatial dimensions
        from the Zarr dataset. This method adjusts data chunks for optimal processing and sets the datatype
        for the time dimension.

        Parameters:
            time_dimension_labels (list of str, optional): List of time labels (e.g., ['20210506']). If not provided,
                                                           defaults to an internally generated sequence.

        Returns:
            xarray.Dataset: The mask data as an xarray dataset with the specified or default time labels and
                            unified chunks.

        Notes:
            Logs the process of loading or creating the mask file. If the file does not exist, it logs a warning
            and proceeds to create a new mask file.
        """
        logger.info("Preparing to load or create mask.")
        # Default to internal values if none are provided
        time_dimension_labels = time_dimension_labels or [f"t{n}" for n in range(len(self.time_dimension_labels))]

        if not self.ref_dataset_file.exists():
            logger.warning(f"Reference Dataset file {self.ref_dataset_file} does not exist. Creating new reference dataset file.")
            self.create_ref_dataset_file(time_dimension_labels)  # Pass time_dimension_labels instead of time_steps

        # Load the mask data
        logger.info(f"Loading reference dataset from {self.ref_dataset_file}")
        self.ref_dataset = xr.open_zarr(self.ref_dataset_file, consolidated=False)

        # Unify chunks
        self.ref_dataset = self.ref_dataset.unify_chunks()

        # Verify and set optimal chunk sizes
        new_chunk_sizes = self.calculate_optimal_chunks(self.ref_dataset, self.max_chunk_size)
        logger.info(f"Setting new chunk sizes: {new_chunk_sizes}")
        self.ref_dataset = self.ref_dataset.chunk(new_chunk_sizes)

        self.ref_dataset['time'] = self.ref_dataset['time'].astype(str)
        return self.ref_dataset

    def create_ref_dataset_file(self, time_dimension_labels=None):
        """
        Creates a new mask file with predefined time dimensions using the existing spatial dimensions from the Zarr dataset.
        If no specific time dimension labels are provided, a default sequence is generated and used.

        Parameters:
            time_dimension_labels (list of str, optional): List of time labels. If None, defaults to a sequence
                                                           like ['t1', 't2', ...] up to the number of dates in the dataset.

        Notes:
            Logs the start of the mask creation process, the settings used, and the completion of the file creation.
            Also checks and ensures the presence of necessary spatial dimensions before proceeding.

        Raises:
            ValueError: If spatial dimensions (x/y or lon/lat) are not found in the Zarr dataset.
        """
        logger.info("Starting the reference dataset creation process.")
        # If no time labels are provided, create default ones from 1 to self.number_of_dates
        if time_dimension_labels is None:
            time_dimension_labels = [f"t{n + 1}" for n in
                                     range(len(self.time_dimension_labels))]  # Create default labels like ['t1', 't2', ...]
        else:
            time_dimension_labels = self.time_dimension_labels

        logger.debug(f"Time dimension labels used: {time_dimension_labels}")

        # Debug: Log or print the time labels to make sure they're correct
        print("Time labels being used:", time_dimension_labels)

        # Use existing spatial coordinates (x, y or lon, lat) from the Zarr dataset
        if 'x' in self.zarr_data.coords and 'y' in self.zarr_data.coords:
            x_coords = self.zarr_data.coords['x'].values
            y_coords = self.zarr_data.coords['y'].values
        elif 'lon' in self.zarr_data.coords and 'lat' in self.zarr_data.coords:
            x_coords = self.zarr_data.coords['lon'].values
            y_coords = self.zarr_data.coords['lat'].values
        else:
            logger.error("Spatial dimensions (x/y or lon/lat) not found in the Zarr dataset.")
            raise ValueError("Spatial dimensions (x/y or lon/lat) not found in the Zarr dataset.")

        # Use provided or default time labels directly
        time_coords = time_dimension_labels

        # Step 1: Create an empty dataset with predefined coordinates, no variables
        ds = xr.Dataset(coords={"time": time_coords, "y": y_coords, "x": x_coords})

        # Save the dataset to a Zarr file (overwrite or append mode)
        ds.to_zarr(self.ref_dataset_file, mode='w', consolidated=True)
        ds.close()
        logger.info(f"Reference dataset file created and saved at {self.ref_dataset_file}")

    def convert(self):
        """
        Main method to convert the GeoPackage layers to xarray.Dataset and write to Zarr file
        This method handles the conversion of geographic layers into data arrays, processes them according to specified patterns,
        and integrates them into a Zarr dataset for use in further analysis or storage.
        """
        logger.info("Starting the conversion process.")
        logger.info(f"Processing patterns: {self.patterns}")

        # Step 1: Retrieve layer information for the specific pattern list
        handler = GeoPackageHandler(self.classification_file)
        layer_info = handler.get_layer_information(self.patterns, separator="_")
        logger.info(f"Layer information retrieved for patterns {self.patterns}: {layer_info}")


        # If all patterns in pattern_of_interest are not existing, add DataArray with 0 values
        if layer_info.empty:
            logger.warning(
                "No layers found for the specified patterns; will proceed to create zero-filled data arrays.")
            for poi in self.patterns:
                logger.info(f"Creating zero-filled data array for pattern of interest: {poi}")
                combined_data = self.create_zero_filled_dataarray(poi)
                self.update_zarr_with_masks(combined_data, f'{poi}')

        #TODO: test '_median', 'QC_mask' and fill with 0 to each layer, not nan !!!, set to nan during the 'QC_mask'
        else:
            # Initialize list for skipped patterns that were not in layer_info
            skipped_pois = []
            # Step 2: If required, process combined pattern layers into a single mask (average or combined)
            if self.add_averaged_mask:
                logger.info(f"Combining layers into averaged mask for pattern: {self.pattern_of_interest}")
                combined_data = self.proccess_pois(layer_info)
                if combined_data is not None:
                    self.update_zarr_with_masks(combined_data, f'{self.pattern_of_interest}_mean')
                    logger.info(f"Averaged mask data updated in Zarr for {self.pattern_of_interest}")
                else:
                    logger.warning(f"No data generated for combined pattern: {self.pattern_of_interest}_mean")

            # Step 3: Process each pattern of interest (POI) individually
            for poi in self.patterns:
                logger.info(f"Processing individual pattern of interest: {poi}")
                combined_data = self.proccess_poi(layer_info, poi)
                if combined_data is not None:
                    self.update_zarr_with_masks(combined_data, f'{poi}')
                    logger.info(f"Mask data updated in Zarr for {poi}")
                else:
                    # If combined_data is None, check for existing reference dataset file and skip for now
                    if not self.ref_dataset_file.exists():
                        skipped_pois.append(poi)
                        logger.warning(f"No data found for {poi} and no existing reference dataset. Skipping.")
                    else:
                        if poi not in self.ref_dataset.data_vars:
                            logger.warning(
                                f"No data found for {poi} and reference dataset file. Skipping for now.")
                            skipped_pois.append(poi)
                        else:
                            # If ref_dataset_file exists, create zero-filled DataArray for this POI and process it
                            logger.info(f"Creating zero-filled data array for missing data on {poi}.")
                            combined_data = self.create_zero_filled_dataarray(poi)
                            self.update_zarr_with_masks(combined_data, f'{poi}')

            # Step 4: After processing available POIs, check for patterns not found in layer_info and create zero-filled DataArray
            all_pois_in_layer_info = set(layer_info['POI'].unique())
            missing_pois = [poi for poi in self.patterns if poi not in all_pois_in_layer_info]

            if missing_pois:
                logger.info(f"Missing patterns not found in layer info: {missing_pois}")
                for missing_poi in missing_pois:
                    combined_data = self.create_zero_filled_dataarray(missing_poi)
                    self.update_zarr_with_masks(combined_data, f'{missing_poi}')

            # Step 5: If there were skipped POIs, attempt to process them after the mask file has been created
            if skipped_pois:
                logger.info(f"Reprocessing skipped patterns: {skipped_pois}")
                for skipped_poi in skipped_pois:
                    combined_data = self.create_zero_filled_dataarray(skipped_poi)
                    self.update_zarr_with_masks(combined_data, f'{skipped_poi}')

        # Replace 0 values with NaN or vice vers
        for var in self.ref_dataset.data_vars:
            self.ref_dataset[var] = self.ref_dataset[var].where(self.ref_dataset[var] != 0, self.set_nan_value)

        #add_spatial_ref_if_missing(self.ref_dataset, self.crs)

        # Finalize the dataset and save to disk
        self.save_to_zarr()
        logger.info("Conversion and Zarr dataset finalization completed.")
        # Close processes to ensure data integrity
        self.ref_dataset.close()
        self.zarr_data.close()

    def prepare_data_for_saving(self):
        """
        Prepare the dataset for saving by removing problematic attributes, adding spatial references,
        and cleaning up chunk encodings. This ensures the dataset is ready for efficient storage and subsequent use.
        """
        logger.info("Preparing data for saving by cleaning attributes and encodings.")
        for var in self.ref_dataset.data_vars:
            logger.debug(f"Inspecting variable: {var}")

            # Safely remove 'grid_mapping' attribute if it exists
            if 'grid_mapping' in self.ref_dataset[var].attrs:
                logger.info(f"Removing 'grid_mapping' attribute from variable '{var}'")
                self.ref_dataset[var].attrs.pop('grid_mapping', None)

            # Remove 'chunks' from encoding if it exists
            if 'chunks' in self.ref_dataset[var].encoding:
                logger.info(f"Removing 'chunks' encoding from variable '{var}'")
                del self.ref_dataset[var].encoding['chunks']

            # Handle 'spatial_ref' conflicts or harmonize it
            if 'spatial_ref' in self.ref_dataset[var].coords:
                if self.ref_dataset['spatial_ref'] not in self.ref_dataset[var]['spatial_ref']:
                    logger.warning(f"Conflicting 'spatial_ref' found in {var}. Dropping to avoid conflicts.")
                    self.ref_dataset[var] = self.ref_dataset[var].drop_vars('spatial_ref', errors='ignore')
                else:
                    logger.info(f"Harmonizing 'spatial_ref' across the dataset for {var}.")

    def save_to_zarr(self):
        """
        Save the dataset to Zarr after preparing it by removing attributes, adding spatial references, and rechunking.
        The dataset is computed to ensure all lazy-loaded data is processed and saved properly.
        """
        logger.info("Starting the process to save the dataset to Zarr.")
        # Prepare the dataset by cleaning up attributes and encodings # to delete some inconsistent metadate of zarr
        self.prepare_data_for_saving()

        # to make each dataset evenly and consistent chunked before writing to a disk
        optimal_chunks = self.calculate_optimal_chunks(self.ref_dataset, self.max_chunk_size)
        logger.info("Rechunking the dataset for optimal performance.")
        self.ref_dataset = self.ref_dataset.chunk(optimal_chunks)

        # Convert Dask array to NumPy array if this not called the dataset is in memory and will not be written in mode='a' only overwrite in mode='w'
        self.ref_dataset = self.ref_dataset.compute()  # Forces the execution of Dask computations
        logger.info("Computations completed, proceeding to save the dataset.")

        # Save the dataset to Zarr, mode='a' allows to add the new data
        self.ref_dataset.to_zarr(self.ref_dataset_file.as_posix(), mode='a')
        logger.info(f"Dataset successfully saved to {self.ref_dataset_file}. Closing the dataset to avoid file locks.")
        # Explicitly close the dataset after saving to avoid file locks
        self.ref_dataset.close()

    '''
    def create_zero_filled_dataarray(self, poi):
        """
        Create a zero-filled DataArray for a specified pattern of interest (POI). This helps in handling missing data by initializing it to zero.

        Parameters:
            poi (str): The specific pattern of interest to process.

        Returns:
            xarray.DataArray: A DataArray filled with zeros according to the dimensions and coordinates of the main dataset.
        """
        logger.info(f"Creating zero-filled DataArray for {poi}.")
        x_dim = self.ref_dataset.coords['x']
        y_dim = self.ref_dataset.coords['y']

        # Check if the 'time' dimension exists, otherwise skip it
        if 'time' in self.ref_dataset.coords:
            time_dim = self.ref_dataset.coords['time']
            # Create a zero-filled DataArray with the 'time' dimension
            combined_data = xr.DataArray(
                data=np.zeros((len(time_dim), len(y_dim), len(x_dim))),
                dims=['time', 'y', 'x'],
                coords={'time': time_dim, 'y': y_dim, 'x': x_dim},
                name=poi  # Assign the POI name to the DataArray
            )
        else:
            logger.warning("'time' dimension not found in mask_zarr. Creating DataArray without 'time' dimension.")
            # Create a zero-filled DataArray without the 'time' dimension
            combined_data = xr.DataArray(
                data=np.zeros((len(y_dim), len(x_dim))),
                dims=['y', 'x'],
                coords={'y': y_dim, 'x': x_dim},
                name=poi  # Assign the POI name to the DataArray
            )
        return combined_data
    '''

    def create_zero_filled_dataarray(self, poi):
        """
        Create a zero-filled DataArray for a specified pattern of interest (POI). This helps in handling missing data by initializing it to zero.

        Parameters:
            poi (str): The specific pattern of interest to process.

        Returns:
            xarray.DataArray: A DataArray filled with zeros according to the dimensions and coordinates of the main dataset.
        """
        logger.info(f"Creating zero-filled DataArray for {poi}.")
        dimensions = ['y', 'x']  # Default dimensions
        coords = {'y': self.ref_dataset.coords['y'], 'x': self.ref_dataset.coords['x']}

        if 'time' in self.ref_dataset.coords:
            dimensions.insert(0, 'time')  # Insert 'time' at the beginning if it exists
            coords['time'] = self.ref_dataset.coords['time']

        data_shape = tuple(len(coords[dim]) for dim in dimensions)
        data = np.zeros(data_shape, dtype=float)

        zero_filled_data = xr.DataArray(data=data, dims=dimensions, coords=coords, name=poi)
        logger.debug(f"Zero-filled DataArray for {poi} created with dimensions {dimensions}.")

        return zero_filled_data

    def update_zarr_with_masks(self, combined_data, pattern_of_interest):
        """
        Update the Zarr dataset with mask data for the specified pattern of interest. This function handles
        numeric validation, potential data filling, and integration into an existing dataset structure.

        Parameters:
            combined_data (xarray.DataArray): The combined mask data for the pattern.
            pattern_of_interest (str): The name of the pattern to be updated or added in the Zarr file.
        """
        logger.info(f"Updating Zarr file with mask data for pattern: {pattern_of_interest}")
        # Set variable name
        variable_name = f'{pattern_of_interest}'

        # Ensure the data is valid before writing to Zarr
        if combined_data.isnull().any():
            logger.warning("Null values detected in data, filling with zeros.")
            combined_data = combined_data.fillna(0).astype(float)  # Handle missing data

        if combined_data.dtype == object:
            raise ValueError("Data contains unsupported object type. Expected numeric data.")

        # self.update_existing_zarr_file(combined_data_reduced, variable_name)
        if variable_name in self.ref_dataset.data_vars:
            logger.info(f"Variable '{variable_name}' exists in the dataset. Updating data.")
            # If variable exists, expand its dimension or update it
            if 'time' in combined_data.dims:
                existing_variable = self.ref_dataset[variable_name]
                # Align datasets to ensure they can be combined without dimension mismatches
                combined_data, self.ref_dataset, existing_variable = xr.align(combined_data, self.ref_dataset, existing_variable, join='outer')

                #logger.info("Unique times in ref_dataset:", self.ref_dataset.time.values)
                #logger.info("Unique times in existing_variable:", existing_variable.time.values)
                #logger.info("Unique times in combined_data:", combined_data.time.values)

                combined_variable = existing_variable.fillna(0).astype(float) + combined_data.fillna(0).astype(float)
                combined_variable = combined_variable.astype(float)
                self.ref_dataset[variable_name] = combined_variable
            else:
                #raise ValueError("Dataset does not have 'time' dimension!.")
                logger.info("The dataarray is 2-dimensional the 'time' dimension is absent.")
                self.ref_dataset[variable_name] = combined_data

        else:
            logger.info(f"Variable '{variable_name}' does not exist. Creating new variable.")
            self.ref_dataset[variable_name] = combined_data

    def proccess_pois(self, layer_info):
        """
        Processes all Points of Interest (POIs) by summing up data over time and calculating the final mask. This function
        aggregates data across all POIs and then applies a mean calculation to normalize the results.

        Parameters:
            layer_info (DataFrame): Information about the layers in the GeoPackage used to fetch data.

        Returns:
            xarray.Dataset: The reduced dataset containing the computed median mask across all POIs.
        """
        logger.info("Processing all POIs and calculating the final mask.")
        total_mask = None

        for poi in layer_info['POI'].unique():
            combined_data = self.proccess_poi(layer_info, poi)
            current_sum = self.sum_over_time(combined_data)

            if total_mask is None:
                total_mask = current_sum
            else:
                total_mask += current_sum

        mask_median = self.calculate_mean(total_mask, len(self.time_dimension_labels))
        median_dataset = xr.Dataset({'median': mask_median})
        reduced_dataset = self.reduce_coordinates(median_dataset, mask_median)
        logger.info("All POIs processed and final mask calculated.")
        return reduced_dataset

    def proccess_poi(self, layer_info, poi):
        """
        Processes individual POI by creating combined data for that POI. This involves filtering the layer information,
        creating a combined dataset from GeoPackage layers, and updating dataset attributes.

        Parameters:
            layer_info (DataFrame): Information about the layers in the GeoPackage.
            poi (str): The specific POI to process.

        Returns:
            xarray.Dataset: The combined data for the POI, or None if no data is found.
        """
        logger.info(f"Processing POI: {poi}")
        poi_df = layer_info[layer_info['POI'] == poi]
        if poi_df.empty:
            logger.warning(f"No layers found for POI: {poi}")
            return None

        combined_data = self.create_combined_data(poi_df)
        if combined_data is not None:
            self.update_data_attributes(combined_data)
            logger.info(f"Data combined and attributes updated for POI: {poi}")
            return combined_data
        else:
            logger.error(f"Failed to create combined data for POI: {poi}")
            return None

    '''
    def create_combined_data(self, layer_names_df):
        """
        Creates a combined dataset from a list of GeoPackage layers specified in the provided DataFrame. Data normalization
        and aggregation are performed based on the 'sum' measurement from each GeoPackage layer.

        Parameters:
            layer_names_df (DataFrame): DataFrame containing the layer information, including names and dates.

        Returns:
            xarray.Dataset: Combined dataset, or None if the input DataFrame is empty.
        """
        if not layer_names_df.empty:
            append = 'as_3_dimension'

            # Create empty xarray
            combined_data = xr.Dataset()
            data_dict = {}

            # Read in the layers from the GeoPackage and process them
            for i, (index, row) in enumerate(layer_names_df.iterrows()):
                layer_name = row['layer_name']
                date = row['date']
                poi = row['POI']

                layer_01 = gpd.read_file(self.classification_file, layer=layer_name)

                if not layer_01.empty:
                    geocube_data = make_geocube(vector_data=layer_01, measurements=["sum"], like=self.zarr_data)

                    # Normalize the values for comparison
                    geocube_data['sum'] = geocube_data['sum'] / 100

                    data_dict[date] = geocube_data['sum'].isel(y=slice(None, None, -1)).isel(x=slice(None, -1, None))

            # Convert the dictionary to an xarray dataset and concatenate along the time dimension
            combined_data = xr.concat(list(data_dict.values()), dim='time')
            combined_data['time'] = list(data_dict.keys())

            return combined_data
        else:
            print("No layers found for the provided POI.")
            return None
    '''

    def create_combined_data(self, layer_names_df):
        """
        Creates a combined dataset from a list of GeoPackage layers specified in the provided DataFrame. Data normalization
        and aggregation are performed based on the 'sum' measurement from each GeoPackage layer.

        Parameters:
            layer_names_df (DataFrame): DataFrame containing the layer information, including names and dates.

        Returns:
            xarray.Dataset: Combined dataset, or None if the input DataFrame is empty.
        """
        if layer_names_df.empty:
            logger.info("Empty DataFrame provided; no data to process.")
            return None

        data_dict = {}
        logger.info("Creating combined dataset from GeoPackage layers.")
        for index, row in layer_names_df.iterrows():
            layer_name = row['layer_name']
            date = row['date']
            layer_data = gpd.read_file(self.classification_file, layer=layer_name)
            if layer_data.empty:
                logger.warning(f"No data in layer: {layer_name}")
                continue

            # Check if the dataset contains CRS information using rioxarray
            if 'spatial_ref' in self.zarr_data.coords:
                crs = self.zarr_data.rio.crs
                print(f"CRS of the Zarr dataset: {crs}")
            else:
                print("CRS information is not available in this dataset.")
                logger.error(f"No crs is defined in '{self.zarr_data}' !")

            print(self.zarr_data)
            print('!!!!!!!!!!!!!!!!!!!')
            print(layer_data)
            print(type(layer_data))


            if layer_data.crs is not None:
                print("GeoDataFrame has a CRS:", layer_data.crs)
            else:
                print("GeoDataFrame does not have a CRS.")

            # Get the bounding box (total extent) of the GeoDataFrame
            bbox = layer_data.total_bounds
            print("Bounding box (extent):", bbox)

            dataset=self.zarr_data

            if 'x' in dataset.coords and 'y' in dataset.coords:
                x_min = dataset['x'].min().item()
                x_max = dataset['x'].max().item()
                y_min = dataset['y'].min().item()
                y_max = dataset['y'].max().item()
                print(f"Bounding box: xmin={x_min}, xmax={x_max}, ymin={y_min}, ymax={y_max}")
            else:
                print("Dataset does not have 'x' and 'y' coordinates.")

            if 'grid_mapping' in dataset.attrs:
                print(f"Grid mapping is present: {dataset.attrs['grid_mapping']}")
            else:
                print("No grid mapping attribute found.")

            # Check for latitude and longitude
            if 'lat' in dataset.coords and 'lon' in dataset.coords:
                print("Dataset has 'lat' and 'lon' coordinates.")
            else:
                print("No obvious spatial coordinates like 'lat' or 'lon' found.")

            geocube_data = make_geocube(vector_data=layer_data, measurements=["sum"], like=self.zarr_data)
            geocube_data['sum'] /= 100  # Normalize data
            data_dict[date] = geocube_data['sum'].isel(y=slice(None, None, -1)).isel(x=slice(None, -1, None))

        if not data_dict:
            logger.warning("No valid data found in any layers.")
            return None

        combined_data = xr.concat(data_dict.values(), dim='time')
        combined_data['time'] = list(data_dict.keys())
        logger.info("Combined dataset created successfully.")
        return combined_data

    '''
    def update_data_attributes(self, data):
        """
        Updates the attributes of a dataset by modifying specific keys and removing others. This ensures the dataset
        is cleaned up before further processing or storage.

        Parameters:
            data (xarray.Dataset): The dataset whose attributes are to be updated.
        """
        print("\nOld Attributes:")
        for key, value in data.attrs.items():
            print(f"{key}: {value}")

        attributes_to_change = {'name': 'mask'}
        attributes_to_delete = ['_FillValue', 'long_name']

        print("\nChanges:")
        for key, new_value in attributes_to_change.items():
            if key in data.attrs:
                data.attrs[key] = new_value
                print(f'"{key}" renamed to "{new_value}".')
            else:
                print(f'"{key}" not found.')

        for attr in attributes_to_delete:
            if attr in data.attrs:
                del data.attrs[attr]
                print(f'"{attr}" deleted.')
            else:
                print(f'"{attr}" not found.')

        print("\nNew Attributes:")
        for key, value in data.attrs.items():
            print(f"{key}: {value}")
    '''

    def update_data_attributes(self, data):
        """
        Updates the attributes of a dataset by modifying specific keys and removing others. This ensures the dataset
        is cleaned up before further processing or storage.

        Parameters:
            data (xarray.Dataset): The dataset whose attributes are to be updated.
        """
        logger.info("Updating dataset attributes.")
        changes = {'name': 'mask'}
        to_delete = ['_FillValue', 'long_name']

        for key, new_value in changes.items():
            if key in data.attrs:
                data.attrs[key] = new_value
                logger.debug(f"Attribute '{key}' updated to '{new_value}'.")
            else:
                logger.debug(f"Attribute '{key}' not found; no update performed.")

        for attr in to_delete:
            if attr in data.attrs:
                del data.attrs[attr]
                logger.debug(f"Attribute '{attr}' deleted.")
            else:
                logger.debug(f"Attribute '{attr}' not found; no deletion performed.")

    # Helper methods for processing POI
    def sum_over_time(self, data):
        return data.sum(dim="time")

    def calculate_mean(self, total_mask, number_of_dates):
        return total_mask / number_of_dates

    def reduce_coordinates(self, dataset, ref_dataset):
        return dataset['median'].sel(x=ref_dataset.x, y=ref_dataset.y)

    def generate_quality_control_layers(self, error_types=['sum_consistency', 'total_sum'], calculation_types=['per_time_step', 'over_time']):
        """
        Generates quality control (QC) masks using the specified error metrics and writes them to the Zarr file if
        the `add_quality_control_mask` flag is set to True.

        This method calculates QC masks based on the provided error types and calculation types, such as:
        - 'sum_consistency': A mask where the sum of the layers is consistent (e.g., sum equals 1).
        - 'MAE' (Mean Absolute Error): Mask based on the mean absolute error between layers.
        - 'RMSE' (Root Mean Square Error): Mask based on the root mean square error.
        - 'FE' (Fractional Error): Mask based on fractional error.
        - 'Z_score': Mask based on Z-score calculation.
        - 'total_sum': QC mask based on the total sum of values either over time or per time step.

        Parameters:
        ----------
        error_types : list of str, optional
            List of error metrics to use for generating QC masks. The available options are:
            'sum_consistency', 'MAE', 'RMSE', 'FE', 'Z_score'.
            Default is ['sum_consistency'].

        calculation_types : list of str, optional
            List of calculation modes for the QC mask generation. The available options are:
            - 'per_time_step': QC mask will be generated for each time step.
            - 'over_time': QC mask will be calculated over the entire time period.
            Default is ['per_time_step', 'over_time'].

        Returns:
        --------
        dict
            A dictionary of QC masks generated for each error type, where the keys are the variable names
            (e.g., 'QC_sum_consistency_per_time_step', 'QC_MAE_over_time') and the values are the corresponding xarray.DataArray.

        Notes:
        ------
        - This method filters the dataset to exclude layers that end with '_mean' or start with 'QC_'.
        - It aligns the layers and sums them to create the QC masks based on the provided error and calculation types.
        - QC masks are written to the Zarr dataset if the `add_quality_control_mask` flag is set to True.
        - If no layers are found for summing, the method logs a warning and exits without generating QC masks.

        Example:
        --------
        >> converter.generate_quality_control_layers(error_types=['sum_consistency', 'RMSE'], calculation_types=['over_time'])
        """

        # Reload or create the dataset if it was closed
        #if self.ref_dataset is None:
        logger.info("Dataset is not loaded. Reopening or creating mask file.")
        self.ref_dataset = self.load_or_create_mask()

        if self.ref_dataset is None:
            logger.warning("No data loaded. Skipping QC mask generation.")
            return None

        logger.info("Filtering layers that do not end with '_mean'.")
        logger.info(f"Variables in ref_dataset: {list(self.ref_dataset.data_vars)}")

        # Filter layers, ensuring we exclude 'mask'
        layers_to_sum = [var for var in self.ref_dataset.data_vars if
                         not var.endswith('_mean') and not var.startswith('QC_')]

        if not layers_to_sum:
            logger.warning("No layers to sum for QC mask generation.")
            return None

        # Initialize summed_data
        logger.info(f"Summing layers: {layers_to_sum}")
        summed_data = xr.zeros_like(self.ref_dataset[layers_to_sum[0]], dtype=np.float32)

        # Align and sum valid layers
        for layer in layers_to_sum:
            valid_layer = self.ref_dataset[layer].where((self.ref_dataset[layer] > 0) & (self.ref_dataset[layer] <= 1))
            valid_layer, summed_data = xr.align(valid_layer.fillna(0), summed_data.fillna(0))
            summed_data += valid_layer

        # Generate QC masks
        QC_masks = self._calculate_qc_masks(error_types, calculation_types, summed_data)

        # Rename the QC masks and save them to the dataset
        for variable_name, QC_mask in QC_masks.items():
            # Rename the QC mask with the appropriate QC name
            logger.info(f"Renaming QC mask to {variable_name}")

            # Use the `.rename()` function to rename the DataArray
            QC_mask = QC_mask.rename(variable_name)

            # Now write the renamed QC mask to the dataset inside the loop
            if variable_name in self.ref_dataset:
                logger.info(f"Variable {variable_name} already exists. Updating data.")
                self.ref_dataset[variable_name].data = QC_mask.data  # Replace the data
            else:
                logger.info(f"Adding new variable {variable_name} to the dataset.")
                self.ref_dataset[variable_name] = QC_mask

        # Check if the flag allows adding the QC mask
        if self.add_quality_control_mask:
            logger.info("add_quality_control_mask is True, writing QC masks to the Zarr file.")
            self._write_qc_masks_to_zarr(QC_masks)
        else:
            logger.info("add_quality_control_mask is False, skipping QC mask write.")

        return QC_masks

    def _calculate_qc_masks(self, error_types, calculation_types, summed_data):
        """
        Internal method to calculate quality control (QC) masks based on specified error metrics.

        Parameters:
        ----------
        error_types : list of str
            List of error metrics to generate QC masks. Options include:
            - 'sum_consistency': QC mask where sum equals 1.
            - 'MAE': Mean Absolute Error.
            - 'RMSE': Root Mean Square Error.
            - 'FE': Fractional Error (relative error).
            - 'Z_score': Z-score from the mean.
            - 'total_sum': QC mask based on the total sum of values either over time or per time step.

        calculation_types : list of str
            List of calculation modes. Options include:
            - 'per_time_step': QC mask generated for each time step.Fz_sc
            - 'over_time': QC mask calculated over the entire time period.

        summed_data : xarray.DataArray
            The array representing the sum of the valid layers, used to generate QC masks.

        Returns:
        --------
        dict
            A dictionary where keys are QC variable names (e.g., 'QC_sum_consistency_per_time_step')
            and values are the corresponding QC masks (xarray.DataArray).

        Notes:
        ------
        - Each error type is calculated based on the corresponding calculation type.
        - If an error type is not recognized, an error is logged and that error type is skipped.
        """
        # Convert summed_data to a NumPy array and get the unique values
        unique_values = np.unique(summed_data.values)

        # Print the unique values
        print("Unique values in summed_data:", unique_values)

        QC_masks = {}
        for error_type in error_types:
            for calculation_type in calculation_types:
                variable_name = f"QC_{error_type}_{calculation_type}"

                if error_type == 'sum_consistency':
                    logger.info(f"Generating sum consistency QC mask ({calculation_type})")

                    if calculation_type == 'per_time_step':
                        # Calculate sum consistency for each time step
                        # this is just test with mask threshold for publication
                        #QC_mask = xr.where(summed_data == 1.0, 1, 0).astype('float64')
                        #QC_mask = xr.where((summed_data >= 0.85) & (summed_data <= 1.05), 1, 0).astype('float64')

                        #TODO: make the trheshold dynamic (threshold value as accepted downscaling error)
                        QC_mask = xr.where((summed_data >= 0.95) & (summed_data <= 1.05), 1, 0).astype('float64')


                    elif calculation_type == 'over_time':
                        # Compute sum consistency over time
                        total_sum_over_time = summed_data.sum(dim='time', keep_attrs=True)
                        # this is just test with mask threshold for publication
                        #QC_mask = xr.where(total_sum_over_time == 1.0, 1, 0).astype('float64')
                        #QC_mask = xr.where((total_sum_over_time >= 0.85) & (total_sum_over_time <= 1.05), 1, 0).astype('float64')

                        #QC_mask = xr.where((total_sum_over_time >= 0.95) & (total_sum_over_time <= 1.05), 1, 0).astype( 'float64')

                    QC_mask = xr.where((summed_data >= 0.990) & (summed_data <= 1.0), 1, 0) # 0.995 is a threshold -> 0.005% error from conversion probaly

                elif error_type == 'MAE':
                    logger.info(f"Generating MAE QC mask ({calculation_type})")
                    mae_error = abs(summed_data - 1.0)
                    QC_mask = mae_error if calculation_type == 'per_time_step' else mae_error.mean(dim='time')

                elif error_type == 'RMSE':
                    logger.info(f"Generating RMSE QC mask ({calculation_type})")
                    squared_error = (summed_data - 1.0) ** 2
                    QC_mask = np.sqrt(squared_error) if calculation_type == 'per_time_step' else np.sqrt(squared_error.mean(dim='time'))

                elif error_type in ['FE', 'Fractional_Error']:
                    logger.info(f"Generating Fractional Error QC mask ({calculation_type})")
                    fractional_error = abs(summed_data - 1.0) / 1.0
                    QC_mask = fractional_error if calculation_type == 'per_time_step' else fractional_error.mean(dim='time')

                #elif error_type == 'Z_score':
                #    logger.info(f"Generating Z-score QC mask ({calculation_type})")
                #    mean_summed = summed_data.mean(dim='time')
                #    std_summed = summed_data.std(dim='time')
                #    QC_mask = (summed_data - mean_summed) / std_summed if calculation_type == 'per_time_step' else ((summed_data - mean_summed) / std_summed).mean(dim='time')

                elif error_type == 'Z_score':

                    logger.info(f"Generating Z-score QC mask ({calculation_type})")

                    if calculation_type == 'per_time_step':

                        logger.info("Calculating Z-score for each time step individually")

                        # Calculate the sum for each time step

                        total_sum = summed_data.sum(dim='time')

                        # Ensure Z-score is calculated only for pixels where total sum is at least 2.9

                        valid_pixels = total_sum >= 2.9

                        if valid_pixels.any():

                            # Calculate mean and standard deviation across time

                            mean_summed = summed_data.mean(dim='time')

                            std_summed = summed_data.std(dim='time')

                            # Calculate Z-score per time step for valid pixels

                            QC_mask = xr.where(valid_pixels,

                                               (summed_data - mean_summed) / std_summed,

                                               np.nan)

                            # Ensure the dimensions match the original data's dimensions

                            QC_mask = QC_mask.transpose(*summed_data.dims)


                        else:

                            logger.warning("Insufficient valid data for Z-score calculation per time step. Setting Z-score mask to NaN.")

                            QC_mask = xr.full_like(summed_data, np.nan)


                    elif calculation_type == 'over_time':

                        logger.info("Calculating Z-score over the entire time period")

                        # Calculate the total sum across the time dimension and remove the time dimension

                        total_sum_over_time = summed_data.sum(dim='time')

                        # Ensure Z-score is calculated only for pixels where the total sum over time is >= 2.9

                        valid_pixels = total_sum_over_time >= 2.9

                        if valid_pixels.any():

                            # Compute mean and standard deviation over time

                            mean_summed_over_time = summed_data.mean(dim='time')

                            std_summed_over_time = summed_data.std(dim='time')

                            # Calculate Z-score over the entire time period, collapsing the time dimension

                            z_score = (mean_summed_over_time - summed_data) / std_summed_over_time

                            # Remove the time dimension by averaging the Z-score across time

                            QC_mask = xr.where(valid_pixels,

                                               z_score.mean(dim='time'),  # Collapse time dimension by averaging

                                               np.nan)

                            # Ensure the resulting mask is 2D (y, x) by dropping the time dimension

                            QC_mask = QC_mask.squeeze()


                        else:

                            logger.warning("Insufficient valid data for Z-score calculation over time. Setting Z-score mask to NaN.")

                            # Create an empty mask with the same (y, x) dimensions

                            QC_mask = xr.full_like(summed_data.isel(time=0), np.nan)

                    # Store the QC mask in the QC_masks dictionary

                    QC_masks[variable_name] = QC_mask

                # Add the new 'total_sum' mask logic here
                elif error_type == 'total_sum':
                    logger.info(f"Generating total sum QC mask ({calculation_type})")

                    if calculation_type == 'per_time_step':
                        # Use summed_data as the mask for each time step (no further computation needed)
                        QC_mask = summed_data  # This gives the total sum per pixel for each time step
                    elif calculation_type == 'over_time':
                        # Compute the sum over the entire time period for each pixel
                        QC_mask = summed_data.sum(dim='time', keep_attrs=True)  # Sum over time per pixel
                else:
                    logger.error(f"Unknown error type '{error_type}'")
                    continue

                QC_masks[variable_name] = QC_mask

        return QC_masks

    def _write_qc_masks_to_zarr(self, QC_masks):
        """
        Internal method to write the generated QC masks to the Zarr dataset.

        Parameters:
        ----------
        QC_masks : dict
            Dictionary containing the generated QC masks, where keys are the variable names and values
            are the xarray.DataArray QC masks.

        Notes:
        ------
        - If a QC mask variable already exists in the Zarr dataset, it is aligned with the new QC mask
          and the two are summed to update the variable.
        - If the QC mask does not exist in the Zarr dataset, a new variable is created and added.
        """
        logger.info(f"Writing QC masks to Zarr file: {self.ref_dataset_file}")
        if not self.ref_dataset_file.exists():
            logger.warning(f"Reference dataset file {self.ref_dataset_file} does not exist. Skipping write.")
            return
        # Open the existing Zarr dataset
        cube = xr.open_zarr(self.ref_dataset_file, consolidated=True)

        for variable_name, QC_mask in QC_masks.items():
            if variable_name in cube:
                logger.info(f"Variable {variable_name} already exists. Adding new QC mask to the existing data.")

                # Align the new QC mask and the existing variable
                existing_variable, QC_mask_aligned = xr.align(cube[variable_name], QC_mask)

                # Sum the existing variable and the new QC mask
                combined_variable = existing_variable.fillna(0).astype(float) + QC_mask_aligned.fillna(0).astype(float)

                # Update the variable in the cube
                cube[variable_name].data = combined_variable.data
            else:
                logger.info(f"Adding new variable {variable_name} to Zarr.")
                cube[variable_name] = QC_mask

        # Save the updated Zarr dataset
        self.save_to_zarr()

    @staticmethod
    def calculate_optimal_chunks(dataset, max_chunk_size):
        """
        Calculate optimal chunk sizes for a Dataset based on its dimensions.

        Args:
            dataset (xarray.Dataset): The dataset to chunk.
            max_chunk_size (int): The maximum size for any chunk along any dimension.

        Returns:
            dict: A dictionary of chunk sizes for each dimension.
        """
        chunk_sizes = {}
        # Iterate through all dimensions across all data variables in the dataset
        all_dims = {dim: dataset.dims[dim] for var in dataset.variables for dim in dataset[var].dims}

        for dim, size in all_dims.items():
            if size <= max_chunk_size:
                # If the dimension size is smaller than or equal to the max chunk size, use the full size
                chunk_sizes[dim] = size
            else:
                # Calculate an appropriate chunk size
                num_chunks = size // max_chunk_size
                if size % max_chunk_size != 0:
                    num_chunks += 1  # Add an extra chunk if there's a remainder
                chunk_size = size // num_chunks
                if size % num_chunks != 0:
                    chunk_size += 1  # Adjust chunk size so the last chunk isn't larger
                chunk_sizes[dim] = chunk_size

        return chunk_sizes

    @staticmethod
    def crop_zarr_to_extent(input_path, output_path=None, crs=None):
        """
        Static method to crop the Zarr dataset to the extent of non-NaN and non-zero summed values across all variables,
        and overwrite the original Zarr file or save it to a new location.

        Parameters:
        - input_path (str): Path to the input Zarr dataset.
        - output_path (str, optional): Path to save the cropped Zarr dataset. If None, overwrites the input dataset.
        - crs (int, optional): EPSG code of the target CRS to assign to the dataset.
        """
        # Open the Zarr dataset
        ds = xr.open_zarr(input_path, consolidated=True)

        # Use the sum_valid_layers function to get the summed_data
        summed_data = ClassificationDownscaler.sum_valid_layers(ds)

        # Create a mask where summed_data is not zero and not NaN
        total_mask = (summed_data != 0) & (~np.isnan(summed_data))
        # Compute the mask to ensure it's not a lazy Dask array
        total_mask = total_mask.compute()

        # Get the bounding box where values exist (i.e., non-zero, non-NaN)
        non_zero_y, non_zero_x = np.where(total_mask.sum(dim='time').values)

        # Get the min/max bounds of x and y where non-zero, non-NaN data exists
        y_min, y_max = non_zero_y.min(), non_zero_y.max()
        x_min, x_max = non_zero_x.min(), non_zero_x.max()

        # Crop the dataset to this bounding box but retain zeros and NaNs inside
        zoomed_ds = ds.isel(y=slice(y_min, y_max + 1), x=slice(x_min, x_max + 1))

        # Ensure the zoomed dataset is computed (execute Dask computations)
        zoomed_ds = zoomed_ds.compute()
        logger.info("Cropped the dataset to the bounding box, now proceeding to save.")

        # Step 9: Set the output path
        if output_path is not None:
            output_path_ = pathlib.Path(output_path)
        else:
            input_path_ = pathlib.Path(input_path)
            output_path_ = input_path_.parent.joinpath(input_path_.stem + '_cropped.zarr')

        # the issue described here https://github.com/pydata/xarray/issues/3476
        for v in list(zoomed_ds.coords.keys()):
            if zoomed_ds.coords[v].dtype == object:
                zoomed_ds[v].encoding.clear()

        for v in list(zoomed_ds.variables.keys()):
            if zoomed_ds[v].dtype == object:
                zoomed_ds[v].encoding.clear()

        # Set CRS to the dataset if provided
        if crs is not None:
            try:
                # Construct the WKT string representation from the EPSG code
                crs_wkt = rasterio.crs.CRS.from_epsg(crs).to_wkt()

                # Set CRS using rioxarray
                zoomed_ds = zoomed_ds.rio.write_crs(crs_wkt)

                logger.info(f"CRS set to: {crs_wkt}")
            except rasterio.errors.CRSError as e:
                logger.error(f"Error setting CRS: {e}")
        else:
            logger.warning("No CRS provided, proceeding without setting a CRS.")



        # Step 10: Save the cropped dataset to Zarr
        zoomed_ds.to_zarr(output_path_.as_posix(), mode='a')
        #zoomed_ds.close()
        logger.info(f"Dataset cropped to the extent of non-NaN and non-zero summed values and saved to {output_path_}")
        return zoomed_ds

    @staticmethod
    def sum_valid_layers(dataset):
        """
        Sums the valid (non-NaN, non-QC) layers in the dataset, treating NaNs as missing but preserving zeros.

        Parameters:
        - dataset (xarray.Dataset): The input dataset.

        Returns:
        - xarray.DataArray: The summed data across all valid layers.
        """
        summed_data = None

        # Loop through all variables, excluding those that start with 'QC_'
        for var in dataset.data_vars:
            if var.startswith('QC_'):
                continue  # Skip variables that start with 'QC_'

            # Create a valid layer but treat NaNs as missing, preserve zeros
            valid_layer = xr.where(~np.isnan(dataset[var]), dataset[var], 0)  # Keep zeros, exclude NaN

            # Accumulate into summed_data
            if summed_data is None:
                summed_data = valid_layer
            else:
                summed_data += valid_layer

        return summed_data

    @staticmethod
    def set_zeros_to_nan(dataset, output_path=None):
        """
        Sets values in the dataset to NaN where the summed data equals 0, while preserving the original dimensions.
        This will result in QC_mask as float64 with nan values and continuous values for some masks

        Parameters:
        - dataset (xarray.Dataset): The dataset to be updated.
        - output_path (str, optional): Path where the updated dataset should be saved (if provided).

        Returns:
        - xarray.Dataset: The updated dataset where values in summed_data == 0 are set to NaN.
        """
        # Calculate the summed data across valid layers
        summed_data = ClassificationDownscaler.sum_valid_layers(dataset)

        # Loop through each variable in the dataset
        for var in dataset.data_vars:
            # Check if the variable is 2D (x, y) or 3D (x, y, time)
            if 'time' in dataset[var].dims:
                # Apply NaN for the 3D variables (x, y, time)
                dataset[var] = dataset[var].where(summed_data != 0, np.nan)
            else:
                # Apply NaN for the 2D variables (x, y)
                dataset[var] = dataset[var].where(summed_data.isel(time=0) != 0, np.nan)

        # Special case for variables starting with 'QC_sum_consist'
        if var.startswith('QC_sum_consistency'):
            # Also set negative values to NaN
            dataset[var] = dataset[var].where(dataset[var] >= 0, np.nan)

        # Save the dataset to Zarr if an output path is provided
        if output_path:
            dataset.to_zarr(output_path, mode='a')
            logger.info(f"NaN values set where the sum of layers equals zero, and the dataset saved to {output_path}")

        # Close the dataset
        dataset.close()

        return dataset

    @staticmethod
    def convert_qc_masks_to_int64(dataset, output_path=None):
        """
        Converts QC masks to int64. Non-zero values are set to 1, and zero values remain as 0.

        Parameters:
        ----------
        - dataset (xarray.Dataset): The dataset to be updated.
        - output_path (str, optional): Path where the updated dataset should be saved (if provided).

        Returns:
        -------
        - xarray.Dataset: The updated dataset with QC masks as int64, where non-zero values are set to 1 and zero values remain 0.
        """
        # Loop through each variable in the dataset
        for var in dataset.data_vars:
            # Only apply to variables that start with 'QC_' (assumed to be the QC masks)
            if var.startswith('QC_'):
                # Convert non-zero values to 1 and zero values remain as 0, then convert to int64
                dataset[var] = xr.where(dataset[var] != 0, 1, 0).astype('int64')

        # Save the dataset to Zarr if an output path is provided
        if output_path:
            dataset.to_zarr(output_path, mode='a')
            logger.info(f"QC masks converted to int64, and the dataset saved to {output_path}")

        # Close the dataset
        dataset.close()

        return dataset


if __name__ == "__main__":
    # Example usage
    classification_file = "path/to/geopackage.gpkg"
    zarr_file = "path/to/output.zarr"
    ref_dataset_file = "path/to/mask.zarr"
    crs = rasterio.crs.CRS.from_epsg(32632)  # UTM Zone 32N
    study_area = "FRIEN"

    patterns_dict = {
        'bare_soil': ['bare_soil'],
        'undamaged_crop': ['vital_crop', 'ripening_crop', 'flowering_crop', 'dry_crop'],
        'lodged_crop': ['vital_lodged_crop', 'dry_lodged_crop'],
        'weed_infestation': ['weed_infestation']
    }

    patterns = ['bare_soil', 'vital_crop', 'ripening_crop', 'flowering_crop', 'dry_crop',
                'vital_lodged_crop', 'dry_lodged_crop', 'weed_infestation']
    for pattern_of_interest, PATTERN_list in patterns_dict.items():
        # Initialize the converter
        converter = ClassificationDownscaler(
            classification_file=classification_file,
            zarr_file=zarr_file,
            ref_dataset_file=ref_dataset_file,
            crs=crs,
            patterns=PATTERN_list,  # 'undamage_crop_mean' mask will be calculated here using median
            pattern_of_interest=pattern_of_interest,  # for each pattern of second level teh mask will be generated
            time_dimension_labels=['20210503'],
            study_area=study_area
        )

        # Process the dataset
        converter.convert()

