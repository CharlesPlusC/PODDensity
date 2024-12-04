import orekit
from orekit.pyhelpers import setup_orekit_curdir, datetime_to_absolutedate,download_orekit_data_curdir

download_orekit_data_curdir()
setup_orekit_curdir()
vm = orekit.initVM()

from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.utils import IERSConventions
from org.orekit.models.earth import ReferenceEllipsoid
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel, SolidTides, OceanTides, ThirdBodyAttraction, Relativity, NewtonianAttraction
from org.orekit.forces.radiation import SolarRadiationPressure, IsotropicRadiationSingleCoefficient, KnockeRediffusedForceModel
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.utils import Constants
from org.orekit.models.earth.atmosphere.data import JB2008SpaceEnvironmentData, CssiSpaceWeatherData
from org.orekit.models.earth.atmosphere import JB2008, DTM2000, NRLMSISE00
from org.orekit.data import DataSource
from org.orekit.time import TimeScalesFactory   
from ..tools.utilities import extract_acceleration, download_file_url

import numpy as np

# Download SOLFSMY and DTCFILE files for JB2008 model
solfsmy_file = download_file_url("https://sol.spacenvironment.net/JB2008/indices/SOLFSMY.TXT", "external/jb08_inputs/SOLFSMY.TXT")
dtcfile_file = download_file_url("https://sol.spacenvironment.net/JB2008/indices/DTCFILE.TXT", "external/jb08_inputs/DTCFILE.TXT")

# Create DataSource instances
solfsmy_data_source = DataSource(solfsmy_file)
dtcfile_data_source = DataSource(dtcfile_file)

def query_jb08(position, datetime):
    frame = FramesFactory.getEME2000()
    wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
    jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                        dtcfile_data_source)

    utc = TimeScalesFactory.getUTC()
    sun = CelestialBodyFactory.getSun()
    atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
    absolute_date = datetime_to_absolutedate(datetime)
    position_vector = Vector3D(float(position[0]), float(position[1]), float(position[2]))
    density = atmosphere.getDensity(absolute_date, position_vector, frame)
    return density

def query_dtm2000(position, datetime):
    frame = FramesFactory.getEME2000()
    wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
    cssi_sw_data = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
    sun = CelestialBodyFactory.getSun()
    atmosphere = DTM2000(cssi_sw_data, sun, wgs84Ellipsoid)
    absolute_date = datetime_to_absolutedate(datetime)
    position_vector = Vector3D(float(position[0]), float(position[1]), float(position[2]))
    density = atmosphere.getDensity(absolute_date, position_vector, frame)
    return density

def query_nrlmsise00(position, datetime):
    frame = FramesFactory.getEME2000()
    wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
    cssi_sw_data = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
    sun = CelestialBodyFactory.getSun()
    atmosphere = NRLMSISE00(cssi_sw_data, sun, wgs84Ellipsoid)
    absolute_date = datetime_to_absolutedate(datetime)
    position_vector = Vector3D(float(position[0]), float(position[1]), float(position[2]))
    density = atmosphere.getDensity(absolute_date, position_vector, frame)
    return density

def state2acceleration(state_vector, t0, cr, cd, cross_section, mass, **force_model_config):

    # given a state vector, and a force model configuration, return the acceleration at that state vector
    # and the individual accelerations due to each force model component

    assert state_vector.shape == (6,), "State vector must be of shape (6,)"

    epochDate = datetime_to_absolutedate(t0)
    accelerations_dict = {}

    if force_model_config.get('36x36gravity', False):
        grav_3636_acc = 0
        MU = Constants.WGS84_EARTH_MU
        monopolegrav = NewtonianAttraction(MU)
        monopole_gravity_eci_t0 = extract_acceleration(state_vector, epochDate, mass, monopolegrav)
        monopole_gravity_eci_t0 = np.array([monopole_gravity_eci_t0[0].getX(), monopole_gravity_eci_t0[0].getY(), monopole_gravity_eci_t0[0].getZ()])
        grav_3636_acc+=monopole_gravity_eci_t0

        gravityProvider = GravityFieldFactory.getNormalizedProvider(36,36)
        gravityfield = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        gravityfield_eci_t0 = extract_acceleration(state_vector, epochDate, mass, gravityfield)
        gravityfield_eci_t0 = np.array([gravityfield_eci_t0[0].getX(), gravityfield_eci_t0[0].getY(), gravityfield_eci_t0[0].getZ()])
        grav_3636_acc+=gravityfield_eci_t0
        accelerations_dict['36x36gravity'] = grav_3636_acc

    if force_model_config.get('90x90gravity', False):
        grav9090_acc = 0
        MU = Constants.WGS84_EARTH_MU
        monopolegrav = NewtonianAttraction(MU)
        monopole_gravity_eci_t0 = extract_acceleration(state_vector, epochDate, mass, monopolegrav)
        monopole_gravity_eci_t0 = np.array([monopole_gravity_eci_t0[0].getX(), monopole_gravity_eci_t0[0].getY(), monopole_gravity_eci_t0[0].getZ()])
        grav9090_acc+=monopole_gravity_eci_t0

        gravityProvider = GravityFieldFactory.getNormalizedProvider(90,90)
        gravityfield = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, False), gravityProvider)
        gravityfield_eci_t0 = extract_acceleration(state_vector, epochDate, mass, gravityfield)
        gravityfield_eci_t0 = np.array([gravityfield_eci_t0[0].getX(), gravityfield_eci_t0[0].getY(), gravityfield_eci_t0[0].getZ()])
        grav9090_acc+=gravityfield_eci_t0
        accelerations_dict['90x90gravity'] = grav9090_acc

    if force_model_config.get('3BP', False):
        luni_solar_acc = 0
        moon = CelestialBodyFactory.getMoon()
        sun = CelestialBodyFactory.getSun()
        moon_3dbodyattraction = ThirdBodyAttraction(moon)
        sun_3dbodyattraction = ThirdBodyAttraction(sun)

        moon_eci_t0 = extract_acceleration(state_vector, epochDate, mass, moon_3dbodyattraction)
        moon_eci_t0 = np.array([moon_eci_t0[0].getX(), moon_eci_t0[0].getY(), moon_eci_t0[0].getZ()])
        luni_solar_acc+=moon_eci_t0

        sun_eci_t0 = extract_acceleration(state_vector, epochDate, mass, sun_3dbodyattraction)
        sun_eci_t0 = np.array([sun_eci_t0[0].getX(), sun_eci_t0[0].getY(), sun_eci_t0[0].getZ()])
        luni_solar_acc+=sun_eci_t0
        accelerations_dict['3BP'] = luni_solar_acc

    if force_model_config.get('solid_tides', False):
        solid_tides_acc = 0
        central_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        mu = Constants.WGS84_EARTH_MU
        tidesystem = TideSystem.ZERO_TIDE
        iersConv = IERSConventions.IERS_2010
        ut1scale = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, False)
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()
        solid_tides_moon = SolidTides(central_frame, ae, mu, tidesystem, iersConv, ut1scale, moon)
        solid_tides_sun = SolidTides(central_frame, ae, mu, tidesystem, iersConv, ut1scale, sun)
        solid_tides_moon_eci_t0 = extract_acceleration(state_vector, epochDate, mass, solid_tides_moon)
        solid_tides_moon_eci_t0 = np.array([solid_tides_moon_eci_t0[0].getX(), solid_tides_moon_eci_t0[0].getY(), solid_tides_moon_eci_t0[0].getZ()])
        solid_tides_acc+=solid_tides_moon_eci_t0
        solid_tides_sun_eci_t0 = extract_acceleration(state_vector, epochDate, mass, solid_tides_sun)
        solid_tides_sun_eci_t0 = np.array([solid_tides_sun_eci_t0[0].getX(), solid_tides_sun_eci_t0[0].getY(), solid_tides_sun_eci_t0[0].getZ()])
        solid_tides_acc+=solid_tides_sun_eci_t0
        accelerations_dict['solid_tides'] = solid_tides_acc

    if force_model_config.get('ocean_tides', False):
        central_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, False)
        ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS
        mu = Constants.WGS84_EARTH_MU
        ocean_tides = OceanTides(central_frame, ae, mu, 4, 4, IERSConventions.IERS_2010, TimeScalesFactory.getUT1(IERSConventions.IERS_2010, False))
        ocean_tides_eci_t0 = extract_acceleration(state_vector, epochDate, mass, ocean_tides)
        ocean_tides_eci_t0 = np.array([ocean_tides_eci_t0[0].getX(), ocean_tides_eci_t0[0].getY(), ocean_tides_eci_t0[0].getZ()])
        accelerations_dict['ocean_tides'] = ocean_tides_eci_t0

    if force_model_config.get('SRP', False):
        radiation_sensitive = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        earth_ellipsoid =  OneAxisEllipsoid(Constants.IERS2010_EARTH_EQUATORIAL_RADIUS, Constants.IERS2010_EARTH_FLATTENING, FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        solarRadiationPressure = SolarRadiationPressure(CelestialBodyFactory.getSun(), earth_ellipsoid, radiation_sensitive)
        solarRadiationPressure.addOccultingBody(CelestialBodyFactory.getMoon(), Constants.MOON_EQUATORIAL_RADIUS)
        solar_radiation_eci_t0 = extract_acceleration(state_vector, epochDate, mass, solarRadiationPressure)
        solar_radiation_eci_t0 = np.array([solar_radiation_eci_t0[0].getX(), solar_radiation_eci_t0[0].getY(), solar_radiation_eci_t0[0].getZ()])
        accelerations_dict['SRP'] = solar_radiation_eci_t0

    if force_model_config.get('knocke_erp', False):
        sun = CelestialBodyFactory.getSun()
        spacecraft = IsotropicRadiationSingleCoefficient(float(cross_section), float(cr))
        onedeg_in_rad = np.radians(1.0)
        angularResolution = float(onedeg_in_rad)  # Angular resolution in radians
        knockeModel = KnockeRediffusedForceModel(sun, spacecraft, Constants.WGS84_EARTH_EQUATORIAL_RADIUS, angularResolution)
        knocke_eci_t0 = extract_acceleration(state_vector, epochDate, mass, knockeModel)
        knocke_eci_t0 = np.array([knocke_eci_t0[0].getX(), knocke_eci_t0[0].getY(), knocke_eci_t0[0].getZ()])
        accelerations_dict['knocke_erp'] = knocke_eci_t0

    if force_model_config.get('relativity', False):
        relativity = Relativity(Constants.WGS84_EARTH_MU)
        relativity_eci_t0 = extract_acceleration(state_vector, epochDate, mass, relativity)
        relativity_eci_t0 = np.array([relativity_eci_t0[0].getX(), relativity_eci_t0[0].getY(), relativity_eci_t0[0].getZ()])
        accelerations_dict['relativity'] = relativity_eci_t0

    if force_model_config.get('jb08drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, False))
        jb08_data = JB2008SpaceEnvironmentData(solfsmy_data_source,
                                            dtcfile_data_source)
        utc = TimeScalesFactory.getUTC()
        sun = CelestialBodyFactory.getSun()
        atmosphere = JB2008(jb08_data, sun, wgs84Ellipsoid, utc)
        drag_sensitive = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, drag_sensitive)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector, epochDate, mass, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_dict['jb08drag'] = atmospheric_drag_eci_t0

    elif force_model_config.get('dtm2000drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cssi_sw_data = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
        sun = CelestialBodyFactory.getSun()
        atmosphere = DTM2000(cssi_sw_data, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector, epochDate, mass, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_dict['dtm2000drag'] = atmospheric_drag_eci_t0

    elif force_model_config.get('nrlmsise00drag', False):
        wgs84Ellipsoid = ReferenceEllipsoid.getWgs84(FramesFactory.getITRF(IERSConventions.IERS_2010, True))
        cssi_sw_data = CssiSpaceWeatherData(CssiSpaceWeatherData.DEFAULT_SUPPORTED_NAMES)
        sun = CelestialBodyFactory.getSun()
        atmosphere = NRLMSISE00(cssi_sw_data, sun, wgs84Ellipsoid)
        isotropicDrag = IsotropicDrag(float(cross_section), float(cd))
        dragForce = DragForce(atmosphere, isotropicDrag)
        atmospheric_drag_eci_t0 = extract_acceleration(state_vector, epochDate, mass, dragForce)
        atmospheric_drag_eci_t0 = np.array([atmospheric_drag_eci_t0[0].getX(), atmospheric_drag_eci_t0[0].getY(), atmospheric_drag_eci_t0[0].getZ()])
        accelerations_dict['nrlmsise00drag'] = atmospheric_drag_eci_t0

    return accelerations_dict

def get_gravity_potential(position_vector, date, degree, order):
    position = Vector3D(float(position_vector[0]), float(position_vector[1]), float(position_vector[2]))
    Epoch_date = datetime_to_absolutedate(date)
    gravityProvider = GravityFieldFactory.getNormalizedProvider(int(degree), int(order))
    gravityfield = HolmesFeatherstoneAttractionModel(FramesFactory.getITRF(IERSConventions.IERS_2010, True), gravityProvider)
    grav_pot = gravityfield.value(Epoch_date, position, Constants.WGS84_EARTH_MU)
    return grav_pot