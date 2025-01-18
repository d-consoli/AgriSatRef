import pathlib
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZonalStatistics:
    """
    Calculate zonal statistics for given vector and raster datasets.

    Attributes:
        vector_path (pathlib.Path): Path to the vector file.
        raster_path (pathlib.Path): Path to the raster file.
        resolution (tuple): Resolution for rasterization.
        reproject_to_crs (str, optional): EPSG code to reproject vector data if different from current.
    """

    def __init__(self, vector, raster, resolution=(-0.05, 0.05), reproject_to_crs=None):
        """
        Initializes ZonalStatistics with vector and raster paths, resolution, and optional CRS reprojection.

        Parameters:
            vector (str or pathlib.Path): Path to the vector file.
            raster (str or pathlib.Path): Path to the raster file.
            resolution (tuple): Resolution for rasterization.
            reproject_to_crs (str, optional): EPSG code to reproject vector data if necessary.
        """
        self.vector_path = pathlib.Path(vector)
        self.raster_path = pathlib.Path(raster)
        self.resolution = resolution
        self.reproject_to_crs = reproject_to_crs
        self.vector_data = self.load_vector_data()  # Load and possibly reproject vector data
        self.raster_data = self.load_raster_data()  # Load the raster data

    def load_vector_data(self):
        """
        Loads the vector data from the file path.

        Returns:
            GeoDataFrame: The loaded vector data.
        """
        if not self.vector_path.exists():
            raise FileNotFoundError(f"Vector file not found: {self.vector_path}")
        vector_data = gpd.read_file(self.vector_path)

        # Reproject the vector data if CRS is different
        if self.reproject_to_crs:
            vector_data = self.reproject_vector_data(vector_data, self.reproject_to_crs)
        return vector_data

    def load_raster_data(self):
        """
        Loads the raster data from the file path.

        Returns:
            DatasetReader: The loaded raster data.
        """
        if not self.raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.raster_path}")
        return rasterio.open(self.raster_path)

    def reproject_vector_data(self, vector_data, target_crs):
        """
        Reprojects vector data to a specified CRS.

        Parameters:
            vector_data (GeoDataFrame): The GeoDataFrame to reproject.
            target_crs (str): The target CRS in EPSG code.

        Returns:
            GeoDataFrame: The reprojected GeoDataFrame.
        """
        if vector_data.crs is None:
            raise ValueError("Input GeoDataFrame does not have a defined CRS.")
        return vector_data.to_crs(target_crs)

    def calculate_zonal_stats(self, stats='sum'):
        """
        Calculates zonal statistics.

        Parameters:
            stats (str or list): Statistics to calculate, e.g., 'sum', 'mean', 'min', 'max'.

        Returns:
            DataFrame: DataFrame containing zonal statistics for each polygon.
        """
        zonal_stats_results = zonal_stats(self.vector_data, self.raster_path.as_posix(), stats=stats, all_touched=False)
        stats_df = pd.DataFrame(zonal_stats_results)
        return stats_df

    def compute_polygon_properties(self):
        """
        Computes polygon properties such as area and number of pixels covered by the polygon.

        Returns:
            GeoDataFrame: Updated GeoDataFrame with polygon area and number of pixels.
        """
        self.vector_data['polygon_area'] = self.vector_data.geometry.area
        self.vector_data['number_of_pixels'] = self.vector_data['polygon_area'] / (
            abs(self.resolution[0]) * abs(self.resolution[1])
        )
        return self.vector_data

    def normalize_statistics(self, stats_df):
        """
        Normalizes zonal statistics based on the number of pixels in each polygon.

        Parameters:
            stats_df (DataFrame): DataFrame containing raw zonal statistics.

        Returns:
            GeoDataFrame: Updated GeoDataFrame with normalized statistics.
        """
        if 'number_of_pixels' not in self.vector_data.columns:
            raise ValueError("Polygon properties not computed. 'number_of_pixels' column is missing.")

        # Normalize each statistic column by dividing by the number of pixels in the polygon
        for col in stats_df.columns:
            self.vector_data[col] = (stats_df[col] / self.vector_data['number_of_pixels']) * 100

        # Fill any NaN values with 0
        self.vector_data.fillna(0, inplace=True)

        return self.vector_data

    def get_final_dataframe(self):
        """
        Combines zonal statistics with polygon properties and normalizes the results.

        Returns:
            GeoDataFrame: Final GeoDataFrame with normalized statistics and polygon properties.
        """
        # Step 1: Calculate raw zonal statistics
        stats_df = self.calculate_zonal_stats()
        logger.info("Zonal Stats DataFrame Columns:", stats_df.columns)
        # Step 2: Compute polygon properties (adds polygon_area and number_of_pixels to self.vector_data)
        self.compute_polygon_properties()

        # Step 3: Normalize statistics based on polygon properties
        return self.normalize_statistics(stats_df)

    def close(self):
        """
        Closes the raster dataset to free up resources. No explicit closing is needed for vector data.
        """
        if self.raster_data:
            self.raster_data.close()
            logger.info("Raster data closed.")
        else:
            logger.warning("No raster data to close.")