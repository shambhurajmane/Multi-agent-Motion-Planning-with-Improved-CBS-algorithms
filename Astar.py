import math
import sys
import os
import heapq
import numpy as np
import matplotlib.pyplot as plt
from heapdict import heapdict
import scipy.spatial.kdtree as kd
# import ipdb



# This is a class representing a differential robot (DiffBot)
class  DiffBot:       
    speedPrecision = 4  # This is the precision of the robot's speed
    wheelSeperation=0.287 * 5  # This is the distance between the robot's wheels, taken as grid dimension
    botDiameter=0.32 * 5  # This is the diameter of the robot
    wheelDiameter=0.066 *5  # This is the diameter of the robot's wheels
    maxVelocity=2.0  # This is the maximum velocity of the robot
    minVelocity=1.0  # This is the minimum velocity of the robot
    step_size=0.5  # This is the step size for the robot's movement

# This is a class representing the cost of different actions for the robot
class Cost:      
    reverse = 30  # Cost of reversing
    directionChange = 300  # Cost of changing direction
    steerAngle = 10  # Cost of steering at an angle
    hybridCost = 20  # Cost of hybrid action
    lowSpeedCost=0  # Cost of moving at low speed
    steerAngleChange = 10  # Cost of changing the steering angle
    
# This is a class representing a node in the path planning algorithm
class Node:       
    def __init__(self, gridIndex, traj, steeringAngle, direction, cost, parentIndex, timestamp ):
        self.gridIndex = gridIndex  # The grid block x, y, yaw index of the node
        self.traj = traj  # The trajectory x, y of a simulated node
        self.steeringAngle = steeringAngle  # The steering angle throughout the trajectory
        self.direction = direction  # The direction throughout the trajectory
        self.cost = cost  # The cost of reaching this node
        self.parentIndex = parentIndex  # The index of the parent node
        self.timestamp = timestamp  # The timestamp of when this node was created

# This class represents a node for a holonomic robot
class HolonomicNode:   
    def __init__(self, gridIndex, cost, parentIndex):
        self.gridIndex = gridIndex  # The grid index of the node
        self.cost = cost  # The cost of reaching this node
        self.parentIndex = parentIndex  # The index of the parent node

# This function returns the grid index of a node as a tuple
def index(Node):       
    return tuple([Node.gridIndex[0], Node.gridIndex[1], Node.gridIndex[2]])

# This function calculates and returns the motion commands for a differential drive robot
def motionCommands():     
    angle=20
    step_length = DiffBot.maxVelocity / 2
    vr= step_length /(DiffBot.step_size* math.cos(math.radians(angle))) + ( DiffBot.wheelSeperation * math.tan(math.radians(angle)))/2
    vl= vr - (DiffBot.wheelSeperation * math.tan(math.radians(angle)))
    motionCommand = [[DiffBot.maxVelocity, DiffBot.maxVelocity],[vr,vl],[-vr,-vl]]
    return motionCommand

# This function normalizes an angle to the range [-pi, pi]
def pi_2_pi(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta

# This function returns the action set for a holonomic robot
def holonomicMotionCommands():     
    holonomicMotionCommand = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    return holonomicMotionCommand

# This function simulates a node using given current Node and Motion Commands
def kinematicSimulationNode(currentNode, motionCommand, mapParameters, collission, simulationLength=1, step = 0.2 ):      

    # Initialize trajectory and timestamp lists
    traj = []
    timestamp = []

    # Calculate the angle for the new node
    angle = pi_2_pi(currentNode.traj[-1][2] + (motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation)
    
    # Append the new position and angle to the trajectory
    traj.append([round(currentNode.traj[-1][0] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.cos(angle),2),
                round(currentNode.traj[-1][1] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.sin(angle),2),
                round(pi_2_pi(angle + (motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation),2)])

    # Continue to append new positions and angles to the trajectory for the length of the simulation
    for i in range(int((simulationLength/step))-1):
        traj.append([round(traj[i][0] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.cos(traj[i][2]),2),
                    round(traj[i][1] + ((motionCommand[0] + motionCommand[1]) /2) * step * math.sin(traj[i][2]),2),
                    round(pi_2_pi(traj[i][2] + (motionCommand[0]-motionCommand[1]) * step /  DiffBot.wheelSeperation),2)])

    # Get the last timestamp from the current node
    time=currentNode.timestamp[-1]

    # Append new timestamps for the length of the simulation
    for i in range(int((simulationLength/step))):
        time = time+0.2
        timestamp.append(round(time,2)) 
        
    # Calculate the grid index for the new node
    gridIndex = [round(traj[-1][0]/mapParameters.xyResolution), \
                 round(traj[-1][1]/mapParameters.xyResolution), \
                 round(traj[-1][2]/mapParameters.yawResolution)]
 
    # Check if the new node is valid (within map bounds and not colliding with an obstacle)
    if not isValid(traj, gridIndex, mapParameters):
        return None
    
    # Check if the new node is in the constraint set (colliding with an obstacle)
    if collission !=[]:
        if inConstraint(traj, timestamp, collission):
            return None

    # Calculate the cost of the new node
    cost = simulatedPathCost(currentNode, motionCommand, simulationLength)
    
    # Return the new node
    return Node(gridIndex, traj, motionCommand[0], motionCommand[1], cost, index(currentNode), timestamp)

# This function checks if the trajectory is in the constraint set (colliding with an obstacle)
def inConstraint(traj, timestamps, collission):
    traj_set = []
    # Create a set of tuples with the x, y coordinates and timestamp for each point in the trajectory
    for i in range(len(traj)):
        traj_set.append((traj[i][0], traj[i][1], timestamps[i]))
    # Check each constraint in the collision set
    for constraint in collission :
        if constraint is not None:
            # If the constraint's location is in the trajectory set, return True
            if (constraint.location[0], constraint.location[1], constraint.location[2]) in traj_set:
                return True
    # If no constraints were found in the trajectory set, return False
    return False

# This function checks if the node is valid (within map bounds and not colliding with an obstacle)
def isValid(traj, gridIndex, mapParameters):    
    # Check if Node is out of map bounds
    if gridIndex[0]<=mapParameters.mapMinX or gridIndex[0]>=mapParameters.mapMaxX or \
       gridIndex[1]<=mapParameters.mapMinY or gridIndex[1]>=mapParameters.mapMaxY:
        return False
    # Check if Node is colliding with an obstacle
    if collision(traj, mapParameters):
        return False
    # If the node is within map bounds and not colliding with an obstacle, return True
    return True

# This function checks if the trajectory is colliding with an obstacle
def collision(traj, mapParameters):       
    # Calculate the radius of the robot
    diffBotRadius = ( DiffBot.botDiameter)/4 + 1
    # Check each point in the trajectory
    for i in traj:
        cx = i[0] 
        cy = i[1] 
        # Find all points in the obstacle set within the robot's radius of the current point
        pointsInObstacle = mapParameters.ObstacleKDTree.query_ball_point([cx, cy], diffBotRadius)
        # If no points were found, continue to the next point in the trajectory
        if not pointsInObstacle:
            continue
        # Check each point in the obstacle set
        for p in pointsInObstacle:
            # Calculate the x and y distances from the obstacle point to the current point
            xo = mapParameters.obstacleX[p] - cx
            yo = mapParameters.obstacleY[p] - cy
            # Rotate the obstacle point around the current point by the current angle
            dx = xo * math.cos(i[2]) + yo * math.sin(i[2])
            dy = -xo * math.sin(i[2]) + yo * math.cos(i[2])
            # If the rotated point is within the robot's radius, return True
            if abs(dx) < diffBotRadius and abs(dy) <  diffBotRadius:
                return True
    # If no collisions were found, return False
    return False



# This function calculates the cost of a simulated path
def simulatedPathCost(currentNode, motionCommand, simulationLength):
    # Start with the cost of the current node
    cost = currentNode.cost
    # Add the distance cost (simulation length times the cost of moving forward or backward)
    if motionCommand[0] > 0:
        cost += simulationLength
    elif motionCommand[0] < 0 :
        cost += simulationLength * Cost.reverse
    # Add the cost of changing direction
    if currentNode != motionCommand[0] :
        cost += Cost.directionChange
    # Add the cost of steering (the steering angle times the cost per unit of steering angle)
    if motionCommand[0] != DiffBot.maxVelocity:
        cost += motionCommand[0] * Cost.steerAngle
    # Add the cost of changing the steering angle (the absolute difference between the current and new steering angles times the cost per unit of steering angle change)
    cost += abs(motionCommand[0] - currentNode.steeringAngle) * Cost.steerAngleChange
    # Return the total cost
    return cost



# This function calculates the Euclidean distance between two nodes
def eucledianCost(holonomicMotionCommand):        
    return math.hypot(holonomicMotionCommand[0], holonomicMotionCommand[1])

# This function returns the grid index of a node as a tuple
def holonomicNodeIndex(HolonomicNode):    
    return tuple([HolonomicNode.gridIndex[0], HolonomicNode.gridIndex[1]])

# This function creates a map of obstacles
def obstaclesMap(obstacleX, obstacleY, xyResolution):
    # Convert the x and y coordinates of the obstacles to grid indices
    obstacleX = [round(x / xyResolution) for x in obstacleX]
    obstacleY = [round(y / xyResolution) for y in obstacleY]
    # Create a 2D list of booleans representing the grid, with all values initially set to False (no obstacle)
    obstacles =[[False for i in range(max(obstacleY))]for i in range(max(obstacleX))]
    # Set the grid locations with obstacles to True
    for i, j in zip(obstacleX, obstacleY): 
        obstacles[i][j] = True
        break
    # Return the map of obstacles
    return obstacles

# This function checks if a node is valid (i.e., within the map bounds and not on an obstacle)
def holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):       
    # Check if the node is out of the map bounds
    if neighbourNode.gridIndex[0]<= mapParameters.mapMinX or \
       neighbourNode.gridIndex[0]>= mapParameters.mapMaxX or \
       neighbourNode.gridIndex[1]<= mapParameters.mapMinY or \
       neighbourNode.gridIndex[1]>= mapParameters.mapMaxY:
        return False
    # Check if the node is on an obstacle
    if obstacles[neighbourNode.gridIndex[0]][neighbourNode.gridIndex[1]]:
        return False
    # If the node is within the map bounds and not on an obstacle, it is valid
    return True

# This function calculates the costs of moving to all nodes from the goal node, taking into account obstacles
def holonomicCostsWithObstacles(goalNode, mapParameters):      
    # Calculate the grid index of the goal node
    gridIndex = [round(goalNode.traj[-1][0]/mapParameters.xyResolution), round(goalNode.traj[-1][1]/mapParameters.xyResolution)]
    # Create a new node at the grid index of the goal node
    gNode =HolonomicNode(gridIndex, 0, tuple(gridIndex))
    # Create a map of obstacles
    obstacles = obstaclesMap(mapParameters.obstacleX, mapParameters.obstacleY, mapParameters.xyResolution)
    # Get the set of possible motion commands
    holonomicMotionCommand = holonomicMotionCommands()
    # Initialize the open set with the goal node
    openSet = {holonomicNodeIndex(gNode): gNode}
    # Initialize the closed set as an empty dictionary
    closedSet = {}
    # Initialize the priority queue with the goal node
    priorityQueue =[]
    heapq.heappush(priorityQueue, (gNode.cost, holonomicNodeIndex(gNode)))
    # While there are nodes in the open set
    while True:
        if not openSet:
            break
        # Pop the node with the lowest cost from the priority queue
        _, currentNodeIndex = heapq.heappop(priorityQueue)
        currentNode = openSet[currentNodeIndex]
        # Remove the current node from the open set and add it to the closed set
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode
        # For each possible motion command
        for i in range(len(holonomicMotionCommand)):
            # Calculate the grid index and cost of the neighbour node
            neighbourNode = HolonomicNode([currentNode.gridIndex[0] + holonomicMotionCommand[i][0],\
                                      currentNode.gridIndex[1] + holonomicMotionCommand[i][1]],\
                                      currentNode.cost + eucledianCost(holonomicMotionCommand[i]), currentNodeIndex)
            # If the neighbour node is not valid, skip it
            if not holonomicNodeIsValid(neighbourNode, obstacles, mapParameters):
                continue
            # Get the index of the neighbour node
            neighbourNodeIndex = holonomicNodeIndex(neighbourNode)
            # If the neighbour node is not in the closed set
            if neighbourNodeIndex not in closedSet:            
                # If the neighbour node is in the open set and its cost is lower than the current cost
                if neighbourNodeIndex in openSet:
                    if neighbourNode.cost < openSet[neighbourNodeIndex].cost:
                        # Update the cost and parent index of the neighbour node in the open set
                        openSet[neighbourNodeIndex].cost = neighbourNode.cost
                        openSet[neighbourNodeIndex].parentIndex = neighbourNode.parentIndex
                else:
                    # If the neighbour node is not in the open set, add it to the open set and the priority queue
                    openSet[neighbourNodeIndex] = neighbourNode
                    heapq.heappush(priorityQueue, (neighbourNode.cost, neighbourNodeIndex))
    # Initialize the cost of all nodes as infinity
    holonomicCost = [[np.inf for i in range(max(mapParameters.obstacleY))]for i in range(max(mapParameters.obstacleX))]
    # For each node in the closed set, update its cost in the cost matrix
    for nodes in closedSet.values():
        holonomicCost[nodes.gridIndex[0]][nodes.gridIndex[1]]=nodes.cost
    # Return the cost matrix
    return holonomicCost


def backtrack(startNode, goalNode, closedSet, plt):        
    # Get the index of the start node
    startNodeIndex= index(startNode)
    # Get the index of the last node in the closed set
    currentNodeIndex = list(closedSet)[-1]
    # Get the current node from the closed set
    currentNode = closedSet[currentNodeIndex]
    # Initialize lists to store x, y, yaw (orientation), and timestamp values
    x=[]
    y=[]
    yaw=[]
    timestamp=[]

    # Loop until we reach the start node from the current node
    while currentNodeIndex != startNodeIndex:
        # Unpack the trajectory of the current node into a, b, c
        a, b, c = zip(*currentNode.traj)
        # Add the values of a, b, c to x, y, yaw in reverse order
        x += a[::-1] 
        y += b[::-1] 
        yaw += c[::-1]
        # Add the timestamp of the current node in reverse order
        timestamp += currentNode.timestamp[::-1]
        # Update the current node index to the parent index of the current node
        currentNodeIndex = currentNode.parentIndex
        # Update the current node to the parent node
        currentNode = closedSet[currentNodeIndex]

    # Return the x, y, yaw, and timestamp lists in reverse order
    return x[::-1], y[::-1], yaw[::-1], timestamp[::-1]

def run(s, g, mapParameters, plt, collission):   
    # Compute grid index for start and goal node
    sGridIndex = [round(s[0] / mapParameters.xyResolution), round(s[1] / mapParameters.xyResolution), round(s[2]/mapParameters.yawResolution)]
    gGridIndex = [round(g[0] / mapParameters.xyResolution), round(g[1] / mapParameters.xyResolution), round(g[2]/mapParameters.yawResolution)]

    # Generate all possible motion commands for the robot
    motionCommand = motionCommands()

    # Create start and end node
    startNode = Node(sGridIndex, [s], 0, 1, 0 , tuple(sGridIndex),[0])
    goalNode = Node(gGridIndex, [g], 0, 1, 0, tuple(gGridIndex),[0])

    # Compute holonomic heuristic
    holonomicHeuristics = holonomicCostsWithObstacles(goalNode, mapParameters)

    # Add start node to open set
    openSet = {index(startNode):startNode}
    closedSet = {}

    # Create a priority queue for nodes based on their costs
    costQueue = heapdict()

    # Add start node into priority queue
    costQueue[index(startNode)] = max(startNode.cost , Cost.hybridCost * holonomicHeuristics[startNode.gridIndex[0]][startNode.gridIndex[1]])

    # Run loop until path is found or open set is empty
    while True:
        # If openSet is empty, no solution available       
        if not openSet:
            print("path not found")
            break

        # Get first node in the priority queue
        currentNodeIndex = costQueue.popitem()[0]
        currentNode = openSet[currentNodeIndex]

        # Move currentNode from openSet to closedSet
        openSet.pop(currentNodeIndex)
        closedSet[currentNodeIndex] = currentNode

        # Check if current node is the goal
        if currentNodeIndex[0] == index(goalNode)[0] and currentNodeIndex[1] == index(goalNode)[1] :
            print("Path Found")
            print(currentNode.traj[-1])
            break
        
        # Get all simulated nodes from current node
        for i in range(len(motionCommand)):
            simulatedNode = kinematicSimulationNode(currentNode, motionCommand[i], mapParameters, collission)
            
            # Check if path is within map bounds and is collision free
            if not simulatedNode:
                continue

            # Draw simulated node
            x,y,z =zip(*simulatedNode.traj)
            plt.plot(x, y, linewidth=0.3, color='g')

            # Check if simulated node is already in closed set
            simulatedNodeIndex = index(simulatedNode)
            
            if simulatedNodeIndex not in closedSet: 
                # Check if simulated node is already in open set
                if simulatedNodeIndex not in openSet:
                    openSet[simulatedNodeIndex] = simulatedNode
                    costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])
                else:
                    if simulatedNode.cost < openSet[simulatedNodeIndex].cost:
                        openSet[simulatedNodeIndex] = simulatedNode
                        costQueue[simulatedNodeIndex] = max(simulatedNode.cost , Cost.hybridCost * holonomicHeuristics[simulatedNode.gridIndex[0]][simulatedNode.gridIndex[1]])

    # Backtrack to find the path
    print("no.of nodes explore ", len(closedSet))
    x, y, yaw, timestamp = backtrack(startNode, goalNode, closedSet, plt)
    return x, y, yaw, timestamp

