import math

def convert_distance_to_gps(self, x, y, lat_current, lon_current):
    EARTH_RADIUS = 6371000.0

    new_latitude  = lat_current  + (y / EARTH_RADIUS) * (180 / math.pi)
    new_longitude = lon_current + (x / EARTH_RADIUS) * (180 / math.pi) / math.cos(lat_current * math.pi/180)

    return new_latitude, new_longitude # lets return the new positions 