import numpy as np
import pathlib
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RasterProcessing:
    """
    Class for handling raster data and related operations, such as converting to vector and retrieving pixel size.
    """

    def __init__(self, raster_path):
        """
        Initializes the RasterProcessing class.

        Parameters:
            raster_path (str or pathlib.Path): Path to the raster file.
        """
        self.raster_path = pathlib.Path(raster_path)
        if not self.raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {self.raster_path}")
        self.raster_data = self.load_raster()

    def load_raster(self):
        """
        Loads the raster file.

        Returns:
            DatasetReader: Raster dataset.
        """
        return rasterio.open(self.raster_path)

    def get_pixel_size(self):
        """
        Retrieves the pixel size of the raster.

        Returns:
            tuple: Pixel width and height.
        """
        # The pixel width and height are stored in the transform (src.transform)
        # src.transform[0] is pixel width, and src.transform[4] is pixel height
        # Note: pixel height is often negative due to the way the Y axis is typically represented
        with self.raster_data as src:
            pixel_size = (src.transform[0], abs(src.transform[4]))
        logger.info(f"Pixel Width: {pixel_size[0]} units, Pixel Height: {pixel_size[1]} units")
        return pixel_size

    def raster_to_vector(self, vector_filename=None, epsg=None, write_to_file=True):
        """
        Converts the raster grid into a vector grid of polygons.

        Parameters:
            vector_filename (str): Filename for the output vector file.
            epsg (int): EPSG code for the output coordinate reference system.
            write_to_file (bool): Whether to write the vector to a file.

        Returns:
            GeoDataFrame: The vector grid as a GeoDataFrame.
        """
        bounds = self.raster_data.bounds
        resolution = self.raster_data.res
        cols = list(np.arange(bounds.left, bounds.right, resolution[0]))
        rows = list(np.arange(bounds.top, bounds.bottom, resolution[1]))
        rows.reverse()

        logger.info('bounds: {}'.format(bounds))
        logger.info('crs: {}'.format(self.raster_data.crs))
        logger.info('length: {}'.format(cols))
        logger.info('width: {}'.format(rows))

        # Create polygons for each cell in the raster grid
        polygons = [
            Polygon([(x, y), (x + resolution[0], y), (x + resolution[0], y - resolution[1]), (x, y - resolution[1])])
            for x in cols for y in rows]

        grid = gpd.GeoDataFrame({'geometry': polygons}, crs=self.raster_data.crs if not epsg else f"epsg:{epsg}")

        # Optionally write the vector to a file
        if write_to_file:
            grid.to_file(vector_filename)
            logger.info('vector writen to a file  {}:'.format(vector_filename))
        return grid


