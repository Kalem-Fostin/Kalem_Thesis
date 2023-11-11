# This is the script for interfacing with the gps 

from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import util # lets import util .. contains handy shared functions

class Boat:
    def __init__(self):
        self.boat = connect('/dev/ttyACM0', wait_ready=True, baud=57600, heartbeat_timeout=200, timeout=200) 
        self.latitude = 0
        self.longitude = 0
        self.headingLatitude = 0
        self.headingLongitude = 0
        self.boat.mode = VehicleMode("GUIDED") # setup the boat mode
        self.boat.armed = True # arm it 
        self.boatWidth = 60 # 60 cm
        self.boatLength = 100 # 100 cm

    def get_gps(self):
        self.latitude = self.boat.location.global_relative_frame.lat # latitude
        self.longitude = self.boat.location.global_relative_frame.lon # longitude

        return self.latitude, self.longitude

    def move_boat_to_coords(self, lat, lon, alt=0):
        waypoint = LocationGlobalRelative(lat, lon, alt) # create a waypoint

        self.boat.simple_goto(waypoint) # tell the boat to travel to the waypoint
    
    def shutdown_boat(self):
        self.boat.close()
    
    def head_home(self):
        self.boat.mode = VehicleMode("RTL") # head back where we started !

    def set_boat_speed(self, speed):
        self.boat.airspeed = speed