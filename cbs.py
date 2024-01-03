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
   

class Root:
    def __init__(self, cost, success, paths, collisions, traj_path):
        # This class represents the root of a tree in the Conflict-Based Search (CBS) algorithm.
        self.cost = cost  # The total cost of the paths
        self.success = success  # Whether the search was successful
        self.paths = paths  # The paths for each agent
        self.collisions = collisions  # The collisions between agents
        self.trajectory = traj_path  # The trajectory of the paths

class Collision:
    def __init__(self, agent1, agent2, location, timestamp):
        # This class represents a collision between two agents.
        self.agent1 = agent1  # The first agent involved in the collision
        self.agent2 = agent2  # The second agent involved in the collision
        self.location = location  # The location of the collision
        self.timestamp = timestamp  # The time of the collision

def get_sum_of_cost(paths):
    # This function calculates the total cost of the paths.
    rst = 0
    for path in paths:
        rst += len(path[0]) - 1  # The cost of a path is its length minus one
        if(len(path[0])>1):
            assert path[0][-1] != path[0][-2]  # Ensure that the last two points in the path are not the same
    return rst

def collsion_check(paths):
    # This function checks for collisions between the paths.
    for i in range(len(paths)-1):
        for j in range(i+1,len(paths)):
            if detect_collision(paths[i],paths[j]) !=None:
                position,t = detect_collision(paths[i],paths[j])
                return Collision(i,j,position,t)  # If a collision is detected, return a Collision object
    return None  # If no collisions are detected, return None

def get_boundingbox(x, y, yaw):
    # This function calculates the bounding box of a point (x, y) with orientation yaw.
    x_BL, y_BL = x - 1 , y - 1  # Bottom-left vertex
    x_B , y_B  = x , y - 1  # Bottom vertex
    x_BR, y_BR = x + 1 , y - 1  # Bottom-right vertex
    x_FR, y_FR = x + 1 , y + 1  # Top-right vertex
    x_F , y_F  = x , y + 1  # Top vertex
    x_FL, y_FL = x - 1 , y + 1  # Top-left vertex
    x_L , y_L  = x - 1 , y  # Left vertex
    x_R , y_R  = x + 1 , y  # Right vertex

    # Return the coordinates of the vertices of the bounding box
    return [x, x_BL, x_B,  x_BR, x_FR, x_F, x_FL, x_L , x_R], [y , y_BL, y_B,  y_BR, y_FR, y_F, y_FL, y_L , y_R]


def detect_collision(path1, path2):
    # This function checks for a collision between two paths.
    t_range = min(len(path1[0]),len(path2[0]))  # The range of time steps to check
    timestamp1 = path1[3]  # The timestamps for path1
    timestamp2 = path2[3]  # The timestamps for path2
    
    for t in range(t_range-1):  # For each time step in the range
        # Get the bounding box for the current position in each path
        vertex_x1, vertex_y1 = get_boundingbox(path1[0][t], path1[1][t], path1[2][t])
        vertex_x2, vertex_y2 = get_boundingbox(path2[0][t], path2[1][t], path2[2][t])
        bb1= []  # List to store the bounding box for path1
        bb2= []  # List to store the bounding box for path2
        
        for i in range(len(vertex_x1)):  # For each vertex in the bounding box
            # Add the vertex to the bounding box list, rounding the coordinates and adding the timestamp
            bb1.append((round(vertex_x1[i]), round(vertex_y1[i]), path1[3][t]))
            bb2.append((round(vertex_x2[i]), round(vertex_y2[i]), path2[3][t]))
        
        for loc in bb1:  # For each location in the bounding box for path1
            if loc in bb2:  # If the location is also in the bounding box for path2
                return [path2[0][t], path2[1][t], path2[3][t]],t  # Return the location and time of the collision
       
    return None  # If no collision is detected, return None

class CBSSolver(object):
    """The high-level search of CBS."""

    def __init__(self, mapParameters, starts, goals):
        # This class represents the high-level search in the Conflict-Based Search (CBS) algorithm.
        self.mapParameters = mapParameters  # The parameters of the map
        self.starts = starts  # The start positions for each agent
        self.goals = goals  # The goal positions for each agent
        self.num_of_agents = len(goals)  # The number of agents

        self.num_of_generated = 0  # The number of nodes generated
        self.num_of_expanded = 0  # The number of nodes expanded
        self.CPU_time = 0  # The CPU time used

    def push_node(self, node):
        # This function pushes a node onto the open list.
        heapq.heappush(self.open_list, (node.cost, len(node.collisions), self.num_of_generated, node))  # Push the node onto the open list, using the cost, number of collisions, and number of generated nodes as the priority
        self.num_of_generated += 1  # Increment the number of generated nodes

    def pop_node(self):
        # This function pops a node from the open list.
        _, _, id, node = heapq.heappop(self.open_list)  # Pop the node with the lowest priority from the open list
        print("Expand node {}".format(id))  # Print the id of the expanded node
        self.num_of_expanded += 1  # Increment the number of expanded nodes
        return node  # Return the expanded node

    def find_solution(self):
        # This method finds a solution to the multi-agent pathfinding problem.
        root = Root(0,False,[],[],[])  # Create a root node with initial cost of 0, no collisions, and empty paths.

        # Run Hybrid A* algorithm for the first agent (self.starts[0] to self.goals[0]).
        x, y, yaw, timestamp = run(self.starts[0], self.goals[0], self.mapParameters, plt, root.collisions)
        root.paths.append([x, y, yaw, timestamp])  # Append the resulting path to the root node's paths.

        print("paaaaa",root.cost)  # Print the cost of the root node.

        # For each of the remaining agents, do the following:
        for i in range(1,len(self.starts)):
            # Run Hybrid A* algorithm for the agent.
            x, y, yaw, timestamp = run(self.starts[i], self.goals[i], self.mapParameters, plt, root.collisions)
            root.paths.append([x, y, yaw, timestamp])  # Append the resulting path to the root node's paths.

            criterion = collsion_check(root.paths)  # Check for collisions in the root node's paths.

            # If there are no collisions, set the root node's success attribute to True.
            if criterion ==None:
                print(f"path found for agent {i}")
                root.success =True
            else:
                # If there are collisions, add the collision to the root node's collisions.
                root.collisions.append(criterion)
                # Enter a loop that continues until a collision-free path is found for the agent.
                while not root.success:
                    del root.paths[i]  # Delete the agent's current path.
                    # Run Hybrid A* algorithm again with the updated collisions.
                    x, y, yaw, timestamp = run(self.starts[i], self.goals[i], self.mapParameters, plt, root.collisions)
                    root.paths.append([x, y, yaw, timestamp])  # Append the resulting path to the root node's paths.
                    criterion = collsion_check(root.paths)  # Check for collisions in the new path.
                    
                    # If there are no collisions, set the root node's success attribute to True.
                    if criterion ==None:
                        print(f"path found for agent {i}")
                        root.success =True
                    root.collisions.append(criterion)  # Add the collision to the root node's collisions.

        print(f"path total agent {len(root.paths)}")  # Print the total number of agents.
        
        return root  # Return the root node.

def print_results(self, node):
    # This method prints the results of the CBS algorithm.
    print("\n Found a solution! \n")
    CPU_time = timer.time() - self.start_time  # Calculate the CPU time used.
    print("CPU time (s):    {:.2f}".format(CPU_time))  # Print the CPU time used.
    print("Sum of costs:    {}".format(get_sum_of_cost(node.paths)))  # Print the sum of the costs of the paths.
    print("Expanded nodes:  {}".format(self.num_of_expanded))  # Print the number of expanded nodes.
    print("Generated nodes: {}".format(self.num_of_generated))  # Print the number of generated nodes.

    print("Solution:")  # Print the solution.
    for i in range(len(node.paths)):  # For each agent, print its path.
        print("agent", i, ": ", node.paths[i])


