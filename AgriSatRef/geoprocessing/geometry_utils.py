import shapely
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeometryUtils:
    @staticmethod
    def safe_loads(wkt):
        """
        Safely load a Well-Known Text (WKT) into a shapely geometry object, attempting to repair if necessary.

        Parameters:
        wkt (str): Well-Known Text representation of a geometry.

        Returns:
        shapely.geometry.base.BaseGeometry or None: A Shapely geometry object if successful, None otherwise.

        This function tries to create a geometry from a WKT string. If the geometry is found to be invalid,
        it attempts to "clean" it by buffering by zero, which can resolve some common geometry issues.
        If an exception occurs during this process, it catches the exception, logs an error message, and
        returns None.
        """
        try:
            geom = shapely.wkt.loads(wkt) if wkt is not None else None
            if geom and not geom.is_valid:
                geom = geom.buffer(0)
            return geom
        except Exception as e:
            print(f"Error processing geometry: {e}")
            return None


    # redundunt to safe_loads(wkt) function - consider to delete this !
    @staticmethod
    def handle_geometry(geom):
        """
        Handle and potentially repair a given geometry object.

        Parameters:
        geom (shapely.geometry.Polygon or None): A geometry object to be validated and repaired if necessary.

        Returns:
        shapely.geometry.Polygon or None: The repaired geometry if successful, None if repair is not possible.

        This function performs several checks and operations on the input geometry:
        1. It checks if the input is a valid `shapely.geometry.Polygon` object. If not, it returns None.
        2. If the polygon is not valid, it tries to repair it by buffering by zero.
        3. If the polygon is still invalid, it attempts to further simplify the geometry with a tolerance of 0.01.
        4. It returns the polygon only if it is valid; otherwise, it returns None.
        """
        if geom is None or not isinstance(geom, shapely.geometry.Polygon):
            return None
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_valid:
            geom = geom.simplify(0.01)
        return geom if geom.is_valid else None