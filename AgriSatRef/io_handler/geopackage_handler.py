import sqlite3
import pathlib
import pandas as pd
import timeit
import fiona
import geopandas as gpd
from geocube.api.core import make_geocube
from geoprocessing.zonal_statistics import ZonalStatistics
import dask.array as da
import gc
import xarray
import rasterio
from rasterio.transform import from_origin


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeoPackageHandler:
    def __init__(self, geopackage_file=None):
        """
        Initialize with an optional GeoPackage file.
        Parameters:
            geopackage_file (str, optional): Path to a GeoPackage file.
        """
        self.geopackage_file = geopackage_file

    def set_geopackage_file(self, geopackage_file):
        """
        Sets the GeoPackage file for individual file operations.
        """
        self.geopackage_file = geopackage_file

    def layer_exists(self, layer_name):
        """
        Check if a specific layer exists in the GeoPackage file.

        This function verifies if the GeoPackage file exists, and then checks if a
        specified layer is present in the file.

        Parameters:
            layer_name (str): The name of the layer to check for existence.

        Returns:
            bool: True if the layer exists, False otherwise.

        Raises:
            FileNotFoundError: If the GeoPackage file does not exist.
        """
        if not pathlib.Path(self.geopackage_file).exists():
            raise FileNotFoundError(f"GeoPackage file does not exist: {self.geopackage_file}")
        return layer_name in fiona.listlayers(self.geopackage_file)

    def get_layer_information(self, patterns, separator="_"):
        """
        Retrieve layer information from a GeoPackage file, allowing for flexible layer name formats.

        This function connects to a GeoPackage file, queries the `gpkg_contents` table to find layers
        matching a list of patterns, and extracts layer details such as name components, date, BBCH,
        and Point of Interest (POI). It splits the layer names based on a given separator and structures
        the information in a DataFrame.

        Parameters:
            patterns (list): A list of string patterns to search for in the layer names.
            separator (str): The character used to split the layer names into components. Default is "_".

        Returns:
            DataFrame: A DataFrame containing the layer name, components, BBCH, date, and POI.
        """
        conn = sqlite3.connect(self.geopackage_file)
        layer_info = []
        for pattern in patterns:
            pattern_query = f"SELECT table_name FROM gpkg_contents WHERE table_name LIKE '%{pattern}%'"
            matching_layers = pd.read_sql_query(pattern_query, conn)
            for _, row in matching_layers.iterrows():
                layer_name = row['table_name']
                components = layer_name.split(separator)
                layer_info.append({
                    'layer_name': layer_name,
                    'components': components,
                    'bbch': components[0] if len(components) > 0 else None,
                    'date': components[1] if len(components) > 1 else None,
                    'POI': separator.join(components[2:]) if len(components) > 2 else None
                })
        conn.close()
        return pd.DataFrame(layer_info)


    def get_available_layers(self):
        """
        Retrieve a list of available layers in the GeoPackage file.

        This function checks if the GeoPackage file exists, and if so, it returns a list of
        all the available layers within the file.

        Returns:
            list: A list of strings representing the names of the layers available in the GeoPackage file.

        Raises:
            FileNotFoundError: If the GeoPackage file does not exist.
        """
        if not pathlib.Path(self.geopackage_file).exists():
            raise FileNotFoundError(f"GeoPackage file does not exist: {self.geopackage_file}")
        return fiona.listlayers(self.geopackage_file)

    def read_layer(self, layer_name):
        """
        Reads a layer from the GeoPackage.

        Parameters:
            layer_name (str): Name of the layer to read.

        Returns:
            GeoDataFrame: The loaded layer as a GeoDataFrame.
        """
        if not self.layer_exists(layer_name):
            raise ValueError(f"Layer '{layer_name}' does not exist in the GeoPackage.")
        return gpd.read_file(self.geopackage_file, layer=layer_name)

    # Function to collect unique values from the specified column across all layers and GeoPackages in a folder
    def get_layer_information_from_folder(self, folder_path, patterns, separator="_"):
        """
        Retrieve layer information from all GeoPackages in a folder.

        Parameters:
            folder_path (str): The path to the folder containing GeoPackage files.
            patterns (list): A list of patterns to match layer names.
            separator (str): The separator used in layer names to extract components.

        Returns:
            DataFrame: A DataFrame with all the collected layer information from the GeoPackages.
        """
        folder = pathlib.Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder_path}")

        # Initialize an empty list to store all the DataFrames
        all_layer_info = []

        # Iterate through each GeoPackage file in the folder
        for geopackage_file in folder.rglob('*.gpkg'):
            print(f"Processing GeoPackage: {geopackage_file}")

            # Set the current GeoPackage file
            self.set_geopackage_file(geopackage_file)

            # Retrieve the layer information for this GeoPackage and append it to the list
            layer_info_df = self.get_layer_information(patterns, separator)
            all_layer_info.append(layer_info_df)

        # Concatenate all the DataFrames into one
        if all_layer_info:
            combined_layer_info = pd.concat(all_layer_info, ignore_index=True)
        else:
            combined_layer_info = pd.DataFrame()  # Return an empty DataFrame if no data found

        return combined_layer_info

    @staticmethod
    def write_raster_with_dask(data_array, output_path, transform, crs, dtype=rasterio.uint8):
        """
        Writes a Dask-backed DataArray to a raster file using rasterio.

        Args:
        data_array (xarray.DataArray): The DataArray to write.
        output_path (str or pathlib.Path): The path to the output raster file.
        transform (affine.Affine): Transformation coefficients for the raster.
        crs (str): Coordinate reference system.
        dtype (rasterio dtype, optional): Data type of the output raster.
        """
        # Ensure the data is a Dask array
        data_array = data_array.chunk({'x': -1, 'y': -1})

        # Open a new raster file with write access
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=data_array.shape[0],
                width=data_array.shape[1],
                count=1,
                dtype=dtype,
                crs=crs,
                transform=transform,
        ) as dst:
            # Write the data using Dask to handle the computation
            dst.write(data_array.data.astype(dtype), 1)

    def calculate_poi_coverage_for_layer(self, ifile=None, indir=None, outdir=None, z_vector=None, lyr_name=None,
                                         input_crs='32633', fformat='gpkg', ofile=None, pattern=None,
                                         reproject_to_crs=None, tempfile=False, code_dict=None,
                                         resolution=(-0.05, 0.05)):
        """
        Generates a dataset by rasterizing vector data, applying zonal statistics, and optionally reprojecting the results.

        Parameters:
            ifile (str or pathlib.Path): Input vector file path. If None, it will use the GeoPackage layers.
            indir (str or pathlib.Path): Directory of the input file.
            outdir (str or pathlib.Path): Output directory for results.
            z_vector (str or pathlib.Path): Path to the zonal statistics vector grid.
            lyr_name (str or None): Layer name for output files. If None, the first layer of the GeoPackage is used.
            input_crs (str): Input coordinate reference system in EPSG code (default: '32633').
            fformat (str): Output file format ('gpkg' or 'shp').
            ofile (str or None): Output file name.
            pattern (list of str or None): Specific patterns to extract (default: None, all patterns).
            reproject_to_crs (str or None): CRS to reproject the output (default: None, no reprojection).
            tempfile (bool): Whether to keep temporary files (default: False).
            code_dict (dict): Dictionary mapping class names to codes.
            resolution (tuple): Resolution for rasterization, given as (width, height).

        Returns:
            str: Path to the output file generated.
        """
        # Supported formats and their respective GeoPandas drivers
        gpd_driver = {
            'gpkg': 'GPKG',
            'shp': 'ESRI Shapefile',
            'geojson': 'GeoJSON',
        }

        # Validate the output format
        if fformat not in gpd_driver:
            raise ValueError(f"Unsupported file format: {fformat}. Supported formats are: {list(gpd_driver.keys())}")

        # Default code dictionary if not provided
        code_dict = code_dict or {
            'bare_soil': 1, 'vital_crop': 2, 'vital_lodged_crop': 3,
            'flowering_crop': 4, 'dry_crop': 5, 'dry_lodged_crop': 6,
            'weed_infestation': 7, 'ripening_crop': 8, 'intercrop': 9
        }

        # Path setup and layer retrieval
        if ifile:
            # If an input vector file is provided, use that
            input_dir = pathlib.Path(indir)
            print(input_dir)
            field = gpd.read_file(input_dir / ifile, layer=lyr_name)
        else:
            # If no input file is provided, use the GeoPackage layers
            available_layers = self.get_available_layers()
            if not lyr_name:
                # If no specific layer is provided, default to the first layer
                lyr_name = available_layers[0] if available_layers else None
            if not lyr_name:
                raise ValueError("No layer available in the GeoPackage.")
            field = self.read_layer(lyr_name)

        # Reproject if necessary
        print(field.crs.to_string())
        if field.crs.to_string() != f'EPSG:{input_crs}':
            field = field.to_crs(input_crs)
            print(f'File was reprojected to {input_crs}')

        field['Code'] = field['Class_name'].apply(lambda x: code_dict.get(x, 'Unknown'))

        # Rasterize the vector data
        start_time = timeit.default_timer()
        field_raster = make_geocube(vector_data=field, measurements=['Code'], resolution=resolution)
        elapsed_time = timeit.default_timer() - start_time
        print(f'Processing time of geocube(): {elapsed_time / 60:.2f} minutes')

        selection = pattern or field['Class_name'].unique()
        output_filename = pathlib.Path(outdir) / (ofile or f"{ifile.stem}_extracted.gpkg")

        for pat in selection:
            pattern_code = code_dict.get(pat)
            print(' -' * 20)
            print(f'Extracting pattern: {pat} ...')

            # Process raster data
            bs = field_raster.where(field_raster['Code'] == pattern_code)
            bs_divided = bs.Code / bs.Code
            pattern_file = pathlib.Path(outdir) / f"{lyr_name}_{pat}.tif"

            print(f"Attempting to write to file {pattern_file}")
            #bs_divided.rio.to_raster(pattern_file)



            #----------------------------------------------------------------------------

            # Example usage
            transform = bs_divided.rio.transform()
            GeoPackageHandler.write_raster_with_dask(bs_divided, pattern_file, transform, bs_divided.rio.crs)

            #----------------------------------------------------------------------------

            # Convert raster array to a dask array for chunked processing
            # Clear up resources
            del bs_divided
            gc.collect()
            print(f"Finished writing {pattern_file}")


            # Step 1: Initialize the ZonalStatistics class
            zonal_stats_processor = ZonalStatistics(
                vector=z_vector, raster=pattern_file, resolution=resolution, reproject_to_crs=reproject_to_crs
            )

            # Step 2: Get the final zonal statistics DataFrame with polygon properties and normalized stats
            zonal_stats_df = zonal_stats_processor.get_final_dataframe()

            # Step 3: Reprojection logic (if needed)
            if reproject_to_crs:
                zonal_stats_df = zonal_stats_processor.reproject_vector_data(zonal_stats_df, reproject_to_crs)
            elif reproject_to_crs is None:
                zonal_stats_df.crs = {'init': f'epsg:{input_crs}'}
            else:
                raise ValueError("parameter 'reproject_to_crs' must be of type str or None!")

            # Step 4: Handle output
            zonal_stats_df.fillna(0, inplace=True)
            zonal_stats_df['sum'] = zonal_stats_df['sum'].astype(int)


            # Remove temporary files if requested
            #if not tempfile:
            #    pattern_file.unlink()

            # Close raster dataset to release resources
            zonal_stats_processor.close()

            print(f"Attempting to delete {pattern_file}")
            if pattern_file.exists():
                pattern_file.unlink()
                print(f"File {pattern_file} deleted.")
            else:
                print(f"File {pattern_file} not found.")


            # Step 5: Save the output file
            output_format = fformat.lower()
            output_file = pathlib.Path(outdir) / f"{output_filename.stem}.{output_format}"
            if output_format in gpd_driver:
                driver_name = gpd_driver[output_format]
                print(output_file)
                zonal_stats_df.to_file(output_file, driver=driver_name, layer=f"{lyr_name}_{pat}")
                print(f"Output saved in {output_format} format.")
            else:
                raise ValueError(
                    f"Unsupported format: {output_format}. Supported formats are: {list(gpd_driver.keys())}")

        return output_filename.as_posix()