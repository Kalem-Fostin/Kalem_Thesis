# This file contains logic for BA* Online Complete Coverage Path Planning of unknown environments.
# This Code was written by Kalem Fostin

import time
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import Astar # Shortest Path Algorithm

# 1's are denoted as the shoreline and obstacles
# 0's are free space that the robot can freely travel to
map1 = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1],
       [1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1],
       [1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
       [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]


map2 = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
        [1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

map3 = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,1],
        [1,1,1,0,1,0,1,1,0,0,1,1,1,0,0,0,0,0,0,1],
        [1,0,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1],
        [1,0,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1],
        [1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
        [1,0,0,1,1,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1],
        [1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,0,0,1,1],
        [1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

map4 = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

map5 = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

map6 = [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]

map = map6

# Create the figure and axis
fig, ax = plt.subplots()
fig.set_facecolor('#bababa')
ax.set_title("BA* Complete Coverage Algorithm", fontsize=16)
ax.set_xlabel('X position (m)', fontsize=14)
ax.set_ylabel('Y position (m)', fontsize=14)
ax.tick_params(axis='both', labelsize=10)
legend = ax.legend([], bbox_to_anchor=(1.05, 1), loc='upper right')
legend.set_title("Legend")

robot_positions = [(-1, -1), (-1, 0), (-1, 1), 
                   (0, -1),           (0, 1),
                   (1, -1),  (1, 0),  (1, 1)]

stack = [] # define our stack for storing neighbouring cells

q = {'x': 0, 'y': 0, 'theta': 0} # this is whats used for describing the robots position on a 2d cartesian plane i.e. from top down view

qSim = {'x': 0, 'y': 0} # this is used for tracking the robot positon but this is a simulation using array. this is the current position

s = {'x': 0, 'y': 0, '2r': 0} # this is what is used for creating tiles which describes where the robot has been

sSim = {'x': 0, 'y': 0} # this is s but for this simulation that uses an array map 

colourPallete = ['#bf812d', '#dfc27d', '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30', '#000080', '#808000', '#008080', '#00FFFF', '#00FF00', '#FF00FF', '#FFA500', '#FFFF00', '#FFFFF0', '#FF0000', '#FF9333']
colourSwapper = 0 # usded for swapping colour pallete

movementPriority = []
legend_names = []
counter = 0
positionHistory = [] # this list is used for tracking the history of the robot each square successfully travelled too which go here 
obstacleHistory = []

backTrackPoints = []
start = {'x': 0, 'y': 0}

OBSTACLE = 2
SHORELINE = 1
FREESPACE = 0

NORTH = 0
SOUTH = 1
EAST = 2
WEST = 3
totalTilesCovered = 0
retraced = 0

def create_graphic():
    # Create rectangles for each tile in the array
    for x in range(20):
        for y in range(20):
            tile_value = map[y][x]
            colour = '#543005' if tile_value == FREESPACE else '#8c510a'
            ax.add_patch(patches.Rectangle((x, y), 1, 1, color=colour))

    # Set the aspect ratio to equal, so the tiles are square
    ax.set_aspect('equal')

# returns true if there is an obstacle and false if not 
def is_obstacle(x, y):

    # check out of bounds
    if (x < 0 or x > 19 or y < 0 or y > 19):
        return True

    # If the desired new position is a blocked zone return true
    if (map[y][x] == OBSTACLE or map[y][x] == SHORELINE):
        return True
    
    return False

# reuturns true if the robot has been here already and false if not 
def has_been_already(x, y):

    # Check out of bounds
    if (x < 0 or x > 19 or y < 0 or y > 19):
        return True
    
    # have we already travelled here before ?
    if ({'x': x, 'y': y} in positionHistory):
        return True
    
    return False

def check_for_obstacles(x, y):
    for dx, dy in robot_positions:
        new_x, new_y = x + dx, y + dy
        if new_x >= 0 and new_x <= 19 and new_y <= 19 and new_y >= 0:
            if map[new_y][new_x] == OBSTACLE or  map[new_y][new_x] == SHORELINE:
                obstacleHistory.append({'x': new_x, 'y': new_y})

def get_next_direction_for_line():
    newDirectionGo = NORTH
    if ((not is_obstacle(qSim['x'], qSim['y'] + 1)) and (not has_been_already(qSim['x'], qSim['y'] + 1))): # CHECK NORTH
        newDirectionGo = NORTH
    elif ((not is_obstacle(qSim['x'], qSim['y'] - 1)) and (not has_been_already(qSim['x'], qSim['y'] - 1))): # CHECK SOUTH
        newDirectionGo = SOUTH
    elif ((not is_obstacle(qSim['x'] + 1, qSim['y'])) and (not has_been_already(qSim['x'] + 1, qSim['y']))): # CHECK EAST
        newDirectionGo = EAST
    elif ((not is_obstacle(qSim['x'] - 1, qSim['y'])) and (not has_been_already(qSim['x'] - 1, qSim['y']))): # CHECK WEST
        newDirectionGo = WEST

    return newDirectionGo

# This function is used for executing boustrophedon motion
# Motion will be performed on the priority of N-S-E-W
def Boustrophedon():
    global counter, start, colourSwapper, legend, totalTilesCovered
    # Update the legend dynamically
    legend.remove()

    legend_item = (patches.Rectangle((qSim['x'], qSim['y']), 1, 1, color=colourPallete[colourSwapper]), "Motion" + str(colourSwapper + 1))
    legend_names.append(legend_item)
    legend = ax.legend(*zip(*legend_names), bbox_to_anchor=(1.33, 1), loc='upper right', title="Legend")
    notComplete = True # used for terminating the boustrophedon motion since we have reached a critical point
    letsLabel = 0
    lineDraw = 0
    prev_x = qSim['x']
    lastGo = 0
    prev_y = qSim['y']
    prevDirection = NORTH

    while notComplete:
        notComplete = False # force to false every iteration 
        direction = NORTH
        qSimCopy = qSim.copy()
        positionHistory.append(qSimCopy) # since we are still going lets add this as a position we have been to
        check_for_obstacles(qSim['x'], qSim['y']) # appends an obstacle to the list 

        if ((not is_obstacle(qSim['x'], qSim['y'] + 1)) and (not has_been_already(qSim['x'], qSim['y'] + 1))): # CHECK NORTH
            # cool this is free space !!!! so lets head in that direction babyyyy
            qSim['y'] += 1
            notComplete = True
        elif ((not is_obstacle(qSim['x'], qSim['y'] - 1)) and (not has_been_already(qSim['x'], qSim['y'] - 1))): # CHECK SOUTH
            qSim['y'] -= 1
            direction = SOUTH
            notComplete = True
        elif ((not is_obstacle(qSim['x'] + 1, qSim['y'])) and (not has_been_already(qSim['x'] + 1, qSim['y']))): # CHECK EAST
            qSim['x'] += 1
            direction = EAST
            notComplete = True
        elif ((not is_obstacle(qSim['x'] - 1, qSim['y'])) and (not has_been_already(qSim['x'] - 1, qSim['y']))): # CHECK WEST
            direction = WEST
            qSim['x'] -= 1
            notComplete = True
        counter += 1
        if notComplete:
            totalTilesCovered += 1 # increment counter 

        nextDirection = get_next_direction_for_line()
        if lineDraw:
            yControl = prev_y + 0.5
            yEnd = qSim['y'] + 0.5
            if lastGo:
                yControl += 0.5
                lastGo = 0
            if not notComplete:
                yEnd += 0.5
            ax.plot([prev_x + 0.5, qSim['x'] + 0.5], [yControl, qSim['y'] + 0.5], color='white')
        lineDraw = 1

        ax.add_patch(patches.Rectangle((qSim['x'], qSim['y']), 1, 1, color=colourPallete[colourSwapper]))
        if not letsLabel and notComplete:
            ax.text(qSim['x'] + 0.5, qSim['y'] + 0.5, 'S' + str(colourSwapper + 1), color='white', fontsize=12, ha='center', va='center')
            lastGo = 1
        letsLabel = 1
        prev_x = qSim['x']
        prev_y = qSim['y']
        # Update the bots position #
        plt.draw()
        plt.pause(0.1) # pause for half a second so we can view
        prevDirection = direction
    start = qSimCopy.copy()
    ax.text(qSim['x'] + 0.5, qSim['y'] + 0.5, 'E' + str(colourSwapper + 1), color='white', fontsize=12, ha='center', va='center')
    colourSwapper += 1 # increment so next colour is new !!!

############################### START OF BACK TRACK RELATED FUNCTIONS #####################################

# Return true if S can be used a a start point for the next boustrophedon 
# else return false 
def can_s_be_used(s):
    sum_function = 0

    x, y = s['x'], s['y']

    # Check S1 and S8 and S1 and S2
    if x + 1 <= 19:
        compare = {'x': x + 1, 'y': y}
        if compare not in obstacleHistory and compare not in positionHistory: # This means is a free zone !
            if y + 1 <= 19: # check S1 against S8
                compare = {'x': x + 1, 'y': y + 1}
                if compare in obstacleHistory or compare in positionHistory:
                    sum_function += 1
                    return True
            if y - 1 >= 0: # Check S1 against S2
                compare = {'x': x + 1, 'y': y - 1}
                if compare in obstacleHistory or compare in positionHistory:
                    sum_function += 1
                    return True
    
    # check S5 and S6, and S5 and S4
    if x - 1 >= 0:
        compare = {'x': x - 1, 'y': y}
        if compare not in obstacleHistory and compare not in positionHistory: # This means is a free zone !
            if y + 1 <= 19: # check S5 and S4
                compare = {'x': x - 1, 'y': y + 1}
                if compare in obstacleHistory or compare in positionHistory:
                    sum_function += 1
                    return True
            if y - 1 >= 0: # Check S5 and S6 
                compare = {'x': x - 1, 'y': y - 1}
                if compare in obstacleHistory or compare in positionHistory:
                    sum_function += 1
                    return True
    
    # check S7 and S6 and S7 and S8
    if y - 1 >= 0:
        compare = {'x': x, 'y': y - 1}
        if compare not in obstacleHistory and compare not in positionHistory: # This means is a free zone !
            if x - 1 >= 0: # check S7 and S6
                compare = {'x': x - 1, 'y': y - 1}
                if compare in obstacleHistory or compare in positionHistory:
                    return True
            if x + 1 <= 19: # Check S5 and S6 
                compare = {'x': x + 1, 'y': y - 1}
                if compare in obstacleHistory or compare in positionHistory:
                    sum_function += 1
                    return True
    
    return False




def create_backtrack_list():

    ## ----------------
    ## | S4 | S3 | S2 |
    ## ----------------
    ## | S5 | S  | S1 |
    ## ----------------
    ## | S6 | S7 | S8 |
    ## ----------------
    backTrackPoints.clear() # clear the list and recalculate
    for i in positionHistory:
        if can_s_be_used(i): # ensures this point is at a corner !
            backTrackPoints.append(i) # lets add it here baby cakes


############################### END OF BACK TRACK RELATED FUNCTIONS #####################################

#### START OF A* #########
def run_astar():
    global start, totalTilesCovered, retraced
    # lets first create a map that we can perform this on 
    # Find the dictionary with the lowest x position
    min_x_dict = min(positionHistory, key=lambda d: d['x'])

    # Find the dictionary with the highest x position
    max_x_dict = max(positionHistory, key=lambda d: d['x'])

    # Find the dictionary with the lowest y position
    min_y_dict = min(positionHistory, key=lambda d: d['y'])

    # Find the dictionary with the highest y position
    max_y_dict = max(positionHistory, key=lambda d: d['y'])

    maze = [[1 for _ in range(max_x_dict['x'] + 2)] for _ in range(max_y_dict['y'] + 2)]
    for i in positionHistory:
        maze[i['y']][i['x']] = 0
    
    print("start !!!", start, "ends", backTrackPoints)

    path = 100000000
    lowestPath = []
    for i in backTrackPoints:
        pathTaken = Astar.astar(maze, (start['x'], start['y']), (i['x'], i['y']))
        if len(pathTaken) < path:
            path = len(pathTaken)
            lowestPath = pathTaken
    
    if path != 100000000:

        totalTilesCovered += path
        retraced += path

    return lowestPath

##### END OF A* ########

##### START OF RUN TO NEXT BOUSTROPHEDON #####
def run_to_next_boustrophedon(path):
    ax.add_patch(patches.Rectangle((qSim['x'], qSim['y']), 1, 1, color='#3B0102'))
    plt.draw()
    plt.pause(0.1) # pause for half a second so we can view
    for i in path:

        qSim['x'] = i[0]
        qSim['y'] = i[1]
        ax.add_patch(patches.Rectangle((qSim['x'], qSim['y']), 1, 1, color='#3B0102'))
        plt.draw()
        plt.pause(0.1) # pause for half a second so we can view
##### END OF RUN TO NEXT BOUSTROPHEDON #####

def main():
    global totalTilesCovered, retraced
    qSim['x'], qSim['y'] = 1, 1 # 10, 0 #6, 5 #1, 1 #10, 0#6, 5
    # Place the robot on the plot

    create_graphic() # Create the graphic to show the algorithm !!
    ax.add_patch(patches.Rectangle((1, 1), 1, 1, color='#bf812d'))
    # Set the aspect ratio to equal, so the tiles are square
    ax.set_aspect('equal')

    # Set the axis limits based on the array size
    ax.set_xlim(0, 20, 1)
    ax.set_ylim(0, 20, 1)
    # Set the tick positions and labels for the x-axis only
    ax.set_xticks(np.arange(0, 20, 1))
    ax.set_xticklabels(range(0, 20))
    ax.set_yticks(np.arange(0, 20, 1))
    ax.set_yticklabels(range(0, 20))
    
    t0 = time.time()
    # Main Control Loop for complete coverage !! #
    while True:
        Boustrophedon()
        print("Finished Boustrophedon !")
        create_backtrack_list()
        print("Finished backtrack lsit creation !")
        if (not backTrackPoints):
            break
        path = run_astar()
        print("Finished A* !")
        run_to_next_boustrophedon(path)
        print("Finished traversal !")

    t1 = time.time()
    plt.savefig("island.jpg", dpi=300)
    print("\n\n#############################################")
    print("Time:", t1 - t0, "seconds")
    print("Total Tiles Covered:", totalTilesCovered)
    print("Retraced:", retraced)

if __name__ == '__main__':
    main()