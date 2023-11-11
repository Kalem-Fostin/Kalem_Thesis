import util
from obj_detection import Camera, FrameData
from gps import Boat
from tile import Tile
import signal
import io
import sys
import numpy as np
import cv2
import os 

class WorldMap:
    def __init__(self):
        self.worldMap = [[0 for _ in range(100)] for _ in range(100)] # array of tiles
    
    def set_value(self, x: int, y: int, tile: Tile):
        self.worldMap[x][y] = tile
    
    def get_value(self, x, y):
        return self.worldMap[x][y]

signalCaught = False

def signal_handler(sig, frame):
    signalCaught = True 

CAMERA = 0
SVO = 1


def main():
    signal.signal(signal.SIGINT, signal_handler)
    #rogue = Boat() # lets define our boat class - this is used for controlling the boat and getting statistics 
    camera = Camera() # lets define our camera class - this is used for image processing etc
    #worldMap = WorldMap() # point to a world map
    #currentLat, currentLon = 100, 100#rogue.get_gps() # get the current gps coordinates and plonk a tile here
    #worldMap.set_value(50, 50, Tile(currentLat, currentLon, isObstacle=False)) # just start it at the middle of the array i.e.e 50,50

    crashImmenent = 0 # used to flag whether the boat is gonna crash or not 
    prevState = 0

    # Control Loop - Lets indefinitely loop
    while not signalCaught:
        # lets detect objects etc and return their positions
        # If we are safe continue on to the current direction 
        # If we are not safe lets choose a direction to move to

        frameData = camera.grab_image() # get the image, objects and their locations 
        
        if frameData is None:
            continue

        objectPositions = frameData.get_object_positions()
        segmentData = frameData.get_seg_data() # get the segment bounding boxes !   
        origImg = frameData.get_original_image() # get the original image 
        boundingBoxes = frameData.get_bounding_boxes() # get the bounding boxes of every detected object !!
        locations = frameData.get_locations() # get the locations extracted from the segmentation masks 
        objectsClash = []

        #print("##################### OBJECTS POSITIONS #######################")
        count = 0
        maska = 0
        maskArray = []

        objectsClash = []
        for obj in objectPositions:
            count += 1
        
        crashImmenent = 0 # reset the flag 

        for loc in locations:

            if len(loc) > 0:
                x = loc[0][0]
                z = loc[1][2]
                Z_THRESHOLD = 70.0 # distance object is away from the boat
                X_THRESHOLD = 80.0 # distance away to the left or right of the boat
                if abs(x) <= X_THRESHOLD and abs(z) <= Z_THRESHOLD:
                    crashImmenent = 1
                    objectsClash.append(boundingBoxes[maska])
            
            maska += 1
    
        if crashImmenent:
            camera.remove_bounding_boxes()
            print("We are going to crash !")
        else:
            #camera.remove_bounding_boxes()
            print("We are now safe :)")
        
        prevState = crashImmenent
                
    camera.close_camera()
        # can we continue forwards? 
        # if so we add a tile after we have moved off the current one

        # 1. Grab the image and detect/extract objects and their positions/dimensions
        # 2. If a collision immenent wait until 2.5m away then choose a direction to turn to avoid the collision 
        #    we want to move away so the length of the object is missed 
        # 3. Repeat ?


################ Main ################
if __name__ == "__main__":
    main()