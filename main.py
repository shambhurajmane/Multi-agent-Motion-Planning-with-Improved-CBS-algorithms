import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd
# import ipdb
from Astar import run , DiffBot
from cbs import CBSSolver


class MapParameters:    
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY):
        self.mapMinX = mapMinX               # Minimum x-coordinate of the map
        self.mapMinY = mapMinY               # Minimum y-coordinate of the map
        self.mapMaxX = mapMaxX               # Maximum x-coordinate of the map
        self.mapMaxY = mapMaxY               # Maximum y-coordinate of the map
        self.xyResolution = xyResolution     # Resolution of the grid
        self.yawResolution = yawResolution   # Resolution of yaw
        self.ObstacleKDTree = ObstacleKDTree # KDTree of obstacles
        self.obstacleX = obstacleX           # X-coordinates of obstacles
        self.obstacleY = obstacleY           # Y-coordinates of obstacles

def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):   
        # Calculate map boundaries based on obstacles
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # Create KDTree for obstacles
        ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])

        # Return map parameters
        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  



def main():
    # Initialize lists to store start and goal positions for each agent
    starts=[]
    goals = []
    
    # Define start and goal positions for agent 1
    starts.append([5, 15, np.deg2rad(0)])  # Start at (5, 15) with 0 degree orientation
    goals.append([25, 15, np.deg2rad(180)])  # Goal at (25, 15) with 180 degree orientation
    
    # Define start and goal positions for agent 2
    starts.append([25, 15, np.deg2rad(180)])  # Start at (25, 15) with 180 degree orientation
    goals.append([5, 15, np.deg2rad(180)])  # Goal at (5, 15) with 180 degree orientation
    
    # Define start and goal positions for agent 3
    starts.append([20, 5, np.deg2rad(-45)])  # Start at (20, 5) with -45 degree orientation
    goals.append([25, 15, np.deg2rad(180)])  # Goal at (25, 15) with 180 degree orientation
    
    # Define start and goal positions for agent 4
    starts.append([5, 20, np.deg2rad(-90)])  # Start at (5, 20) with -90 degree orientation
    goals.append([20, 5, np.deg2rad(-90)])  # Goal at (20, 5) with -90 degree orientation

    # Get obstacle map
    obstacleX, obstacleY = map()  # Call the map function to get the obstacle map

    # Calculate map parameters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 1, np.deg2rad(15.0))  # Calculate map parameters based on obstacle map and resolution

    # Initialize CBS (Conflict-Based Search) solver with map parameters and start/goal positions
    cbs = CBSSolver(mapParameters, starts, goals)

    # Find solution using CBS solver
    solution = cbs.find_solution()  # The solution includes paths for each agent and any collisions

    # Extract collisions and paths from the solution
    collisions = solution.collisions  # List of collisions
    paths = solution.paths  # List of paths for each agent

    # Visualize the solution
    visualize(obstacleX, obstacleY, paths, collisions)  # Call the visualize function to display the paths and collisions



def visualize(obstacleX, obstacleY, paths, collisions):
    # This function is used to visualize the paths and collisions of the robots on a 2D plot.

    # Loop for animation frames
    for k in range(200):
        plt.cla()  # Clear the current figure
        plt.xlim(min(obstacleX), max(obstacleX))  # Set the x-limits of the current axes
        plt.ylim(min(obstacleY), max(obstacleY))  # Set the y-limits of the current axes
        plt.plot(obstacleX, obstacleY, "sk")  # Plot the obstacles as black squares

        # Loop over all paths
        for path in paths:
            plt.plot(path[0], path[1], linewidth=1.5, color='r', zorder=0)  # Plot the path as a red line

            # Draw the differential drive robot and its bounding box at each point along the path
            if k < len(path[0]):  
                drawDiffBot(path[0][k], path[1][k], path[2][k])  # Draw the robot
                draw_boundingbox(path[0][k], path[1][k], path[2][k])  # Draw the bounding box
                plt.arrow(path[0][k], path[1][k], 1*math.cos(path[2][k]), 1*math.sin(path[2][k]), width=.1)  # Draw an arrow representing the robot's orientation
            else:
                # If we've reached the end of the path, draw the robot and bounding box at the final position
                drawDiffBot(path[0][len(path[0])-1], path[1][len(path[0])-1], path[2][len(path[0])-1])
                draw_boundingbox(path[0][len(path[0])-1], path[1][len(path[0])-1], path[2][len(path[0])-1])
                plt.arrow(path[0][len(path[0])-1], path[1][len(path[0])-1], 1*math.cos(path[2][len(path[0])-1]), 1*math.sin(path[2][len(path[0])-1]), width=.1)

        plt.title("Hybrid A*")  # Set the title of the current axes
        plt.pause(0.1)  # Pause for a short period to create an animation effect
        
    plt.show()  # Display the figure


def drawDiffBot(x, y, yaw, color='black'):
    # This function is used to draw a differential drive robot on a 2D plot.

    theta_B = math.pi + yaw  # Calculate the angle of the robot's back side

    # Calculate the position of the robot's back side
    xB = x + DiffBot.botDiameter / 3 * np.cos(theta_B)
    yB = y + DiffBot.botDiameter / 3 * np.sin(theta_B)

    # Calculate the angles of the robot's bottom-left and bottom-right vertices
    theta_BL = theta_B + math.pi / 2
    theta_BR = theta_B - math.pi / 2

    # Calculate the positions of the robot's bottom-left and bottom-right vertices
    x_BL = xB + DiffBot.botDiameter / 3 * np.cos(theta_BL)
    y_BL = yB + DiffBot.botDiameter / 3 * np.sin(theta_BL)
    x_BR = xB + DiffBot.botDiameter / 3 * np.cos(theta_BR)
    y_BR = yB + DiffBot.botDiameter / 3 * np.sin(theta_BR)

    # Calculate the positions of the robot's front-left and front-right vertices
    x_FL = x_BL + DiffBot.botDiameter * np.cos(yaw)
    y_FL = y_BL + DiffBot.botDiameter * np.sin(yaw)
    x_FR = x_BR + DiffBot.botDiameter * np.cos(yaw)
    y_FR = y_BR + DiffBot.botDiameter * np.sin(yaw)

    # Plot the robot as a rectangle
    plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                [y_BL, y_BR, y_FR, y_FL, y_BL],
                linewidth=1, color='black')


def draw_boundingbox(x, y, color='red'):
    # This function is used to draw a bounding box around a point (x, y) on a 2D plot.

    # Calculate the coordinates of the vertices of the bounding box
    x_BL, y_BL = x - 1 , y - 1  # Bottom-left vertex
    x_B , y_B  = x , y - 1  # Bottom vertex
    x_BR, y_BR = x + 1 , y - 1  # Bottom-right vertex
    x_FR, y_FR = x + 1 , y + 1  # Top-right vertex
    x_F , y_F  = x , y + 1  # Top vertex
    x_FL, y_FL = x - 1 , y + 1  # Top-left vertex
    x_L , y_L  = x - 1 , y  # Left vertex
    x_R , y_R  = x + 1 , y  # Right vertex
   
    # Plot the bounding box as a red rectangle
    plt.plot([x_BL, x_B,  x_BR, x_R, x_FR, x_F, x_FL, x_L ,x_BL ], [y_BL, y_B,  y_BR,y_R,  y_FR, y_F, y_FL, y_L ,y_BL ],linewidth=0.5, color='red')

def map():
    # This function is used to build a 2D map with obstacles.

    # Initialize the lists of obstacle coordinates
    obstacleX, obstacleY = [], []

    # Add the border obstacles
    for i in range(31):
        obstacleX.append(i)
        obstacleY.append(0)
    for i in range(31):
        obstacleX.append(0)
        obstacleY.append(i)
    for i in range(31):
        obstacleX.append(i)
        obstacleY.append(30)
    for i in range(31):
        obstacleX.append(30)
        obstacleY.append(i)

    # Add the internal obstacles
    for i in range(8):
        obstacleX.append(i)
        obstacleY.append(18)
    for i in range(22,31):
        obstacleX.append(i)
        obstacleY.append(18)
    for i in range(8):
        obstacleX.append(i)
        obstacleY.append(12) 
    for i in range(22,31):
        obstacleX.append(i)
        obstacleY.append(12) 

    # Return the lists of obstacle coordinates
    return obstacleX, obstacleY

if __name__ == '__main__':
    # If this script is run directly, call the main function
    main()