import numpy as np
import datetime
from datetime import timedelta
import pandas as pd

class SatTrack:
    def __init__(self, sat_name, RAAN, inclination, eccentricity, argp, mean_anomaly, mean_motion, observer_latitude, observer_longitude, observer_altitude):
        self.sat_name = sat_name
        # All given in degrees/meters
        self.RAAN = RAAN 
        self.inclination = inclination
        self.eccentricity = eccentricity
        self.argp = argp
        self.mean_anomaly = mean_anomaly
        self.mean_motion = mean_motion
        self.observer_latitude= observer_latitude
        self.observer_longitude = observer_longitude
        self.observer_altitude = observer_altitude
        self.gmst_rad = None
        self.datetime = None
        self.semi_major_orbit = None
        self.E = None
        self.M = None
        self.true_anomaly_rads = None
        self.target_pqw = None
        self.sat_coverage_geojson = None
    
    def geodetic_to_ecef(self, latitude, longitude, altitude=0.0):
        """Convert latitude/longitude/altitude(HAE) to ECEF Coordinate Frame"""
        latitude = np.radians(latitude)
        longitude = np.radians(longitude)
        a = 6378.137 * 1000
        e_squared = 0.00669437999013
        N = a / np.sqrt(1 - (e_squared * np.sin(latitude) ** 2))
        x = (N + altitude) * np.cos(latitude) * np.cos(longitude)
        y = (N + altitude) * np.cos(latitude) * np.sin(longitude)
        z = (N * (1 - e_squared) + altitude) * np.sin(latitude)
        return np.array([x, y, z], dtype=float)
  
    def ecef_to_geodetic(self, x, y, z):
        """Convert ECEF Coorindates to Geodetic (Lat, Lon, Altitude (HAE))"""
        epsilon_1 = 1e-6
        epsilon_2 = 1e-6 
        a = 6378.137 * 1000
        b = 6356.752 * 1000
        e_squared = 0.00669437999013
        N = a

        H = np.linalg.norm([x, y, z]) - np.sqrt(a * b)
        B = np.arctan2(z, np.linalg.norm([x, y]) * (1 - ((e_squared * N) / (N + H))))
        while True:
            N_i = a / np.sqrt(1 - e_squared * np.sin(B) ** 2)
            H_i = (np.linalg.norm([x, y]) / np.cos(B)) - N_i
            B_i = np.arctan2(z, np.linalg.norm([x, y]) * (1 - ((e_squared * N_i) / (N_i + H_i))))
            if (np.abs(H_i - H) < epsilon_1) and (np.abs(B_i - B) < epsilon_2):
                break 
            N = N_i
            H = H_i
            B = B_i

        lon = np.rad2deg(np.arctan2(y, x))
        lat = np.rad2deg(B)
        alt = H
        return lat, lon, alt # Reutrns in degrees
  
    def ecef_to_topocentric(self, target_ecef, observer_ecef):
        """Given an observer location convert target coordinates from ECEF to Topocentric (Observer perspective coords)"""
        lat_rads = np.radians(self.observer_latitude)
        lon_rads = np.radians(self.observer_longitude)
        R = np.array([
            [-np.sin(lon_rads),                     np.cos(lon_rads),                      0],
            [-np.sin(lat_rads)*np.cos(lon_rads),    -np.sin(lat_rads)*np.sin(lon_rads),    np.cos(lat_rads)],
            [ np.cos(lat_rads)*np.cos(lon_rads),     np.cos(lat_rads)*np.sin(lon_rads),     np.sin(lat_rads)]
        ])
        observer_to_target = target_ecef - observer_ecef
        target_topo = R @ observer_to_target
        return target_topo

    def topocentric_to_ecef(self, target_topo, observer_ecef):
        """Given Topocentric Coordinates of the target from the perspective of the observer convert back to ECEF coordinates"""
        lat_rads = np.radians(self.observer_latitude)
        lon_rads = np.radians(self.observer_longitude)
        R = np.array([
            [-np.sin(lon_rads),                     np.cos(lon_rads),                      0],
            [-np.sin(lat_rads)*np.cos(lon_rads),    -np.sin(lat_rads)*np.sin(lon_rads),    np.cos(lat_rads)],
            [ np.cos(lat_rads)*np.cos(lon_rads),     np.cos(lat_rads)*np.sin(lon_rads),     np.sin(lat_rads)]
        ]).T
        return (R @ target_topo) + observer_ecef

    def eci_to_ecef(self, eci_vector, theta):
        """Convert ECI to ECEF given some Theta (GMST Radians) and a ECI Vector"""
        # Rotation by +theta
        if not self.gmst_rad:
            return print('Must first calculate GMST (Radians) from gmst_from_julian_date func')
        R = np.array([
            [ np.cos(theta),  np.sin(theta), 0],
            [-np.sin(theta),  np.cos(theta), 0],
            [ 0,              0,             1]
        ])
        return R @ eci_vector

    def ecef_to_eci(self, ecef_vector):
        """Convert ECEF to ECI given an ECEF vector and theta (GMST Radians)"""
        # Rotation by -theta
        if not self.gmst_rad:
            return print('Must first calculate GMST (Radians) from gmst_from_julian_date func')
        R = np.array([
            [ np.cos(self.gmst_rad), -np.sin(self.gmst_rad), 0],
            [ np.sin(self.gmst_rad),  np.cos(self.gmst_rad), 0],
            [ 0,              0,             1]
        ])
        return R @ ecef_vector

    def solve_kepler(self, M, tol=1e-6):
        E = M
        while True:
            dE = (M - (E - self.eccentricity * np.sin(E))) / (1 - self.eccentricity * np.cos(E))
            E += dE
            if abs(dE) < tol:
                break
        self.E = E
        sqrt_e_ratio = np.sqrt((1 + self.eccentricity) / (1 - self.eccentricity))
        self.true_anomaly_rads = 2 * np.arctan(sqrt_e_ratio * np.tan(E / 2))
        print('Loaded E and processing true anomaly radians from perigee')


    def semi_major(self):
        mu = 3.986e5  # km^3/s^2, Earth's gravitational parameter
        n_rad_s = (2 * np.pi * self.mean_motion) / 86400.0  # rev/day -> rad/s
        self.semi_major_orbit = (mu / (n_rad_s**2))**(1/3)*1000  # in kilometers
        print('Loaded semi major orbit')


    def true_target_vector(self, a_m, true_anomaly):
        """Given a true anomaly (target) radius from point of perigee, semi_major, and orbit eccentricity return target PQW coordinate frame (sat body)"""
        r = (a_m * (1 - self.eccentricity**2)) / (1 + self.eccentricity * np.cos(true_anomaly))
        pqw = np.array([
            r * np.cos(true_anomaly),
            r * np.sin(true_anomaly),
            0.0
        ])
        self.target_pqw = pqw
        return print('True Target Vector loaded')

    def parse_tle_epoch(self, tle_epoch_str):
        """Parse TLE EPOCH Str EX. 25037.96095632 and return datetime UTC"""
        year_two_digits = int(tle_epoch_str[0:2])
        day_of_year = float(tle_epoch_str[2:])

        if year_two_digits < 57:
            full_year = 2000 + year_two_digits
        else:
            full_year = 1900 + year_two_digits

        day_int = int(day_of_year)  # 320
        day_frac = day_of_year - day_int  # 0.90946019

        total_seconds = day_frac * 24 * 3600
        base_date = datetime(full_year, 1, 1) + timedelta(days=day_int - 1)
        self.datetime = base_date + timedelta(seconds=total_seconds)

        return "datetime loaded as UTC datetime obj"

    def gmst_from_julian_date(self, datetime):
        """Using datetime object get julian date and convert to GMST radians."""
        jd = pd.Timestamp(datetime).to_julian_date()
        frac = jd - np.floor(jd)
        JD0 = np.where(frac >= 0.5, np.floor(jd) + 0.5, np.floor(jd) - 0.5)
        H = (jd - JD0) * 24.0
        D  = jd  - 2451545.0
        D0 = JD0 - 2451545.0
        T = D / 36525.0
        GMST_hours = 6.697374558 \
                    + (0.06570982441908 * D0) \
                    + (1.00273790935   * H ) \
                    + (0.000026        * (T**2))
                    
        GMST_hours = np.mod(GMST_hours, 24.0)
        GMST_deg = GMST_hours * 15.0
        gmst_rad = np.radians(GMST_deg)
        self.gmst_rad = gmst_rad
        return print("GMST Radians loaded... Call obj_name.gmst_rad")

    def compute_footprint(self, sat_ecef):
        # Use Earth radius in meters
        R_EARTH_m = 6378.137 * 1000  # in meters
        sub_lat, sub_lon, sat_alt = self.ecef_to_geodetic(*sat_ecef)
        max_angle = np.arccos(R_EARTH_m / (R_EARTH_m + sat_alt))
        footprint_radius = R_EARTH_m * np.tan(max_angle)
        angular_distance = footprint_radius / R_EARTH_m  # This is in radians
        angles = np.linspace(0, 2 * np.pi, 100)
        footprint_points = []
        
        for theta in angles:
            d_lat = angular_distance * np.cos(theta)
            d_lon = angular_distance * np.sin(theta) / np.cos(np.radians(sub_lat))
            lat = sub_lat + np.degrees(d_lat)
            lon = sub_lon + np.degrees(d_lon)
            footprint_points.append((lon, lat))
        
        return footprint_points



    def export_geojson(self, footprint_points, satellite_position,timestamp):
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [footprint_points + [footprint_points[0]]] 
                    },
                    "properties": {
                        "name": self.sat_name + " Coverage Area",
                        "satellite_position": satellite_position,
                        "timestamp": timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
            ]
        }
        self.sat_coverage_geojson = geojson_data
        print(f"GeoJSON file saved: {self.sat_name}_coverage.geojson")
        return


    
    def pqw_to_eci(self, target_pqw):
        """Convert Target PQW coordinates to ECI. Used for then converting to ECEF and finally to Geodetic"""
        R_pqw_to_eci = np.array([
            [np.cos(np.radians(self.RAAN))*np.cos(np.radians(self.argp)) - np.sin(np.radians(self.RAAN))*np.sin(np.radians(self.argp))*np.cos(np.radians(self.inclination)),
            -np.cos(np.radians(self.RAAN))*np.sin(np.radians(self.argp)) - np.sin(np.radians(self.RAAN))*np.cos(np.radians(self.argp))*np.cos(np.radians(self.inclination)),
            np.sin(np.radians(self.RAAN))*np.sin(np.radians(self.inclination))],

            [np.sin(np.radians(self.RAAN))*np.cos(np.radians(self.argp)) + np.cos(np.radians(self.RAAN))*np.sin(np.radians(self.argp))*np.cos(np.radians(self.inclination)),
            -np.sin(np.radians(self.RAAN))*np.sin(np.radians(self.argp)) + np.cos(np.radians(self.RAAN))*np.cos(np.radians(self.argp))*np.cos(np.radians(self.inclination)),
            -np.cos(np.radians(self.RAAN))*np.sin(np.radians(self.inclination))],

            [np.sin(np.radians(self.argp))*np.sin(np.radians(self.inclination)),
            np.cos(np.radians(self.argp))*np.sin(np.radians(self.inclination)),
            np.cos(np.radians(self.inclination))]
        ])

        target_eci = R_pqw_to_eci @ target_pqw
        return target_eci