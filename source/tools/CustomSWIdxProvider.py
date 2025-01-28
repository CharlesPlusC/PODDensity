from org.orekit.models.earth.atmosphere.data import PythonAbstractSolarActivityData
from org.orekit.time import AbsoluteDate

class CustomSolarActivityData(PythonAbstractSolarActivityData):
    def __init__(self, custom_data):
        """
        Initialize with custom solar activity data.
        
        :param custom_data: Dictionary mapping AbsoluteDate to solar activity parameters
        """
        super().__init__()
        self.custom_data = custom_data

    def getMinDate(self):
        """Return the minimum date available in the custom data."""
        return min(self.custom_data.keys())

    def getMaxDate(self):
        """Return the maximum date available in the custom data."""
        return max(self.custom_data.keys())

    def getInstantFlux(self, date):
        """Retrieve the instantaneous solar flux (F10.7) for the given date."""
        return self.custom_data.get(date, {}).get('instant_flux', 0.0)

    def getMeanFlux(self, date):
        """Retrieve the mean solar flux (F10.7) for the given date."""
        return self.custom_data.get(date, {}).get('mean_flux', 0.0)

    def getThreeHourlyKP(self, date):
        """Retrieve the 3-hourly Kp index for the given date."""
        return self.custom_data.get(date, {}).get('kp_3hr', 0.0)

    def get24HoursKp(self, date):
        """Retrieve the 24-hour mean Kp index for the given date."""
        return self.custom_data.get(date, {}).get('kp_24hr', 0.0)
    
from org.orekit.models.earth.atmosphere import DTM2000
from org.orekit.bodies import CelestialBodyFactory
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory
from org.hipparchus.geometry.euclidean.threed import Vector3D

def query_dtm2000_with_custom_space_weather(position, datetime, custom_data):
    # Initialize Orekit
    utc = TimeScalesFactory.getUTC()
    frame = FramesFactory.getEME2000()
    wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
    sun = CelestialBodyFactory.getSun()

    # Create the custom solar activity data object
    custom_space_weather = CustomSolarActivityData(custom_data)

    # Initialize the DTM2000 atmosphere model
    atmosphere = DTM2000(custom_space_weather, sun, wgs84Ellipsoid, utc)

    # Convert position and date
    absolute_date = datetime_to_absolutedate(datetime)
    position_vector = Vector3D(float(position[0]), float(position[1]), float(position[2]))

    # Query the atmospheric density
    density = atmosphere.getDensity(absolute_date, position_vector, frame)
    return density

from org.orekit.time import AbsoluteDate

utc = TimeScalesFactory.getUTC()

custom_data = {
    AbsoluteDate(2025, 1, 1, 0, 0, 0, utc): {
        'instant_flux': 150.0,
        'mean_flux': 140.0,
        'kp_3hr': 4.5,
        'kp_24hr': 3.8,
    },
    AbsoluteDate(2025, 1, 2, 0, 0, 0, utc): {
        'instant_flux': 155.0,
        'mean_flux': 142.0,
        'kp_3hr': 5.0,
        'kp_24hr': 4.0,
    },
    # Add more dates and data as needed
}

density = query_dtm2000_with_custom_space_weather(
    position=[7000e3, 7000e3, 7000e3],
    datetime=some_datetime_object,
    custom_data=custom_data
)
print("Density:", density)