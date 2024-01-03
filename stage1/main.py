import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd
import ipdb
from Astar import run , DiffBot


class MapParameters:    
    def __init__(self, mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY):
        self.mapMinX = mapMinX               # map min x coordinate(0)
        self.mapMinY = mapMinY               # map min y coordinate(0)
        self.mapMaxX = mapMaxX               # map max x coordinate
        self.mapMaxY = mapMaxY               # map max y coordinate
        self.xyResolution = xyResolution     # grid block length
        self.yawResolution = yawResolution   # grid block possible yaws
        self.ObstacleKDTree = ObstacleKDTree # KDTree representating obstacles
        self.obstacleX = obstacleX           # Obstacle x coordinate list
        self.obstacleY = obstacleY           # Obstacle y coordinate list


def calculateMapParameters(obstacleX, obstacleY, xyResolution, yawResolution):   
        print("map")
        # calculate min max map grid index based on obstacles in map
        mapMinX = round(min(obstacleX) / xyResolution)
        mapMinY = round(min(obstacleY) / xyResolution)
        mapMaxX = round(max(obstacleX) / xyResolution)
        mapMaxY = round(max(obstacleY) / xyResolution)

        # create a KDTree to represent obstacles
        ObstacleKDTree = kd.KDTree([[x, y] for x, y in zip(obstacleX, obstacleY)])
        print(len(obstacleX))

        return MapParameters(mapMinX, mapMinY, mapMaxX, mapMaxY, xyResolution, yawResolution, ObstacleKDTree, obstacleX, obstacleY)  


def main():

    # Set Start, Goal x, y, theta
    starts=[]
    s1 = [20, 10, np.deg2rad(45)]
    g1 = [10, 20, np.deg2rad(45)]
    
    s2 = [20, 20, np.deg2rad(-45)]
    g2 = [10, 10, np.deg2rad(-45)]


    starts=[s1, s2]
    goals=[g1, g2]
    # Get Obstacle Map
    obstacleX, obstacleY = map()

    # Calculate map Paramaters
    mapParameters = calculateMapParameters(obstacleX, obstacleY, 1, np.deg2rad(15.0))

    # Run Hybrid A*
    paths = []
    for i in range(len(starts)):
        x, y, yaw = run(starts[i], goals[i], mapParameters, plt)
        paths.append([x, y, yaw])
    
    visualize(obstacleX, obstacleY, paths)

def visualize(obstacleX, obstacleY, paths):
    # Draw Animated Differential drive robot
     
    for k in range(100):
        plt.cla()
        plt.xlim(min(obstacleX), max(obstacleX)) 
        plt.ylim(min(obstacleY), max(obstacleY))
        plt.plot(obstacleX, obstacleY, "sk")
        
        for path in paths:
            plt.plot(path[0], path[1], linewidth=1.5, color='r', zorder=0)
            if k <len(path[0]):  
                drawDiffBot(path[0][k], path[1][k], path[2][k])
                plt.arrow(path[0][k], path[1][k], 1*math.cos(path[2][k]), 1*math.sin(path[2][k]), width=.1)
            else:
                drawDiffBot(path[0][len(path[0])-1], path[1][len(path[0])-1], path[2][len(path[0])-1])
                plt.arrow(path[0][len(path[0])-1], path[1][len(path[0])-1], 1*math.cos(path[2][len(path[0])-1]), 1*math.sin(path[2][len(path[0])-1]), width=.1)
   
        plt.title("Hybrid A*")
        plt.pause(0.1)    
        
    plt.show()


def drawDiffBot(x, y, yaw, color='black'):
    theta_B = math.pi + yaw

    xB = x + DiffBot.botDiameter / 4 * np.cos(theta_B)
    yB = y + DiffBot.botDiameter / 4 * np.sin(theta_B)

    theta_BL = theta_B + math.pi / 2
    theta_BR = theta_B - math.pi / 2

    x_BL = xB + DiffBot.botDiameter / 2 * np.cos(theta_BL)        # Bottom-Left vertex
    y_BL = yB + DiffBot.botDiameter / 2 * np.sin(theta_BL)
    x_BR = xB + DiffBot.botDiameter / 2 * np.cos(theta_BR)        # Bottom-Right vertex
    y_BR = yB + DiffBot.botDiameter / 2 * np.sin(theta_BR)

    x_FL = x_BL + DiffBot.botDiameter * np.cos(yaw)               # Front-Left vertex
    y_FL = y_BL + DiffBot.botDiameter * np.sin(yaw)
    x_FR = x_BR + DiffBot.botDiameter * np.cos(yaw)               # Front-Right vertex
    y_FR = y_BR + DiffBot.botDiameter * np.sin(yaw)

    plt.plot([x_BL, x_BR, x_FR, x_FL, x_BL],
                [y_BL, y_BR, y_FR, y_FL, y_BL],
                linewidth=1, color='black')
    
def map():
    # Build Map
    obstacleX, obstacleY = [], []

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
    
    # for i in range(10,20):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(30,51):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(0,31):
    #     obstacleX.append(20)
    #     obstacleY.append(i) 

    # for i in range(0,31):
    #     obstacleX.append(30)
    #     obstacleY.append(i) 

    # for i in range(40,50):
    #     obstacleX.append(15)
    #     obstacleY.append(i)

    # for i in range(25,40):
    #     obstacleX.append(i)
    #     obstacleY.append(35)

    # Parking Map
    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(0)

    # for i in range(51):
    #     obstacleX.append(0)
    #     obstacleY.append(i)

    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(50)

    # for i in range(51):
    #     obstacleX.append(50)
    #     obstacleY.append(i)

    # for i in range(51):
    #     obstacleX.append(i)
    #     obstacleY.append(40)

    # for i in range(0,20):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(29,51):
    #     obstacleX.append(i)
    #     obstacleY.append(30) 

    # for i in range(24,30):
    #     obstacleX.append(19)
    #     obstacleY.append(i) 

    # for i in range(24,30):
    #     obstacleX.append(29)
    #     obstacleY.append(i) 

    # for i in range(20,29):
    #     obstacleX.append(i)
    #     obstacleY.append(24)

    return obstacleX, obstacleY




if __name__ == '__main__':
    main()