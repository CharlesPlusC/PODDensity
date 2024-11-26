import os
import pandas as pd
import numpy as np 
import orekit
import matplotlib.pyplot as plt
from orekit.pyhelpers import setup_orekit_curdir
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import Constants, IERSConventions, PVCoordinates
from org.orekit.frames import FramesFactory
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.hipparchus.geometry.euclidean.threed import Vector3D

orekit.initVM()
setup_orekit_curdir()

def transform_to_lat_lon_alt(csv_path):
    df = pd.read_csv(csv_path)
    
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                             Constants.WGS84_EARTH_FLATTENING, itrf)
    inertial_frame = FramesFactory.getEME2000()
    utc = TimeScalesFactory.getUTC()
    
    latitudes, longitudes, altitudes = [], [], []

    for _, row in df.iterrows():
        x, y, z = float(row['x']), float(row['y']), float(row['z'])
        
        date_time = pd.to_datetime(row['UTC'])
        timestamp = AbsoluteDate(
            date_time.year, date_time.month, date_time.day,
            date_time.hour, date_time.minute, float(date_time.second), utc
        )

        position = Vector3D(x, y, z)
        pv_coordinates = PVCoordinates(position, Vector3D.ZERO)
        
        # Transform to geodetic point for latitude, longitude, and altitude
        geodetic_point = earth.transform(pv_coordinates.getPosition(), inertial_frame, timestamp)
        
        latitudes.append(np.degrees(geodetic_point.getLatitude()))
        longitudes.append(np.degrees(geodetic_point.getLongitude()))
        altitudes.append(geodetic_point.getAltitude())

    df['latitude'] = latitudes
    df['longitude'] = longitudes
    df['altitude'] = altitudes
    df.to_csv(csv_path, index=False)

def process_storm_data(storm_folder, missions):
    for mission in missions:
        mission_folder = os.path.join(storm_folder, mission)
        if not os.path.isdir(mission_folder):
            print(f"Directory {mission_folder} not found. Skipping.")
            continue

        for storm_file in os.listdir(mission_folder):
            print(f"Processing file: {storm_file}")
            if storm_file.endswith(".csv"):
                storm_csv_path = os.path.join(mission_folder, storm_file)
                transform_to_lat_lon_alt(storm_csv_path)

storm_folder = "output/PODDensityInversion/Data/StormAnalysis/"
missions = ["CHAMP"]#"CHAMP", ,"TerraSAR-X"

process_storm_data(storm_folder, missions)