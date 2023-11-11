import util
from gps import Boat

# this class is used as a tile, it says the tile location, whether it is an obstacle or not etc
class Tile(Boat):
    def __init__(self, lat, lon, isObstacle=False):
        self.latitude = lat
        self.longitude = lon
        self.isObstacle = isObstacle
    
    def get_coordinates(self):
        # returns the bounding box of the square i.e. 4 (lat,long) points
        # topLeft ----------- topRight
        # |                          |
        # |                          |
        # bottomLeft ----- bottomRight

        topLeft = convert_distance_to_gps(-1, 1, self.latitude, self.longitude)
        bottomLeft = convert_distance_to_gps(-1, -1, self.latitude, self.longitude)
        topRight = convert_distance_to_gps(1, 1, self.latitude, self.longitude)
        bottomRight = convert_distance_to_gps(1, -1, self.latitude, self.longitude)

        return topLeft, topRight, bottomRight, bottomleft # Clockwise direction starting from top lef t




