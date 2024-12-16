# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import*
class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


import time
import util  # Ensure util.Stack is available or replace with a Python stack implementation



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first and display timing and path information.
    """
    # Track the start time
    start_time = time.time()

    frontier = util.Stack()  # Use a Stack for DFS
    explored = set()         # Set to track visited states

    # Push the start state with an empty path
    start_state = problem.getStartState()
    print(f"Starting State: {start_state}")
    frontier.push((start_state, []))

    while not frontier.isEmpty():
        currState, currPath = frontier.pop()

        # Check if it's a goal state
        if problem.isGoalState(currState):
            # Track the end time
            end_time = time.time()

            # Calculate and print elapsed time
            elapsed_time = end_time - start_time
            print(f"Goal Found: {currState}")
            print(f"Path to Goal: {currPath}")
            print(f"Time Taken: {elapsed_time:.4f} seconds")
            return currPath

        # Process the state only if it hasn't been explored
        if currState not in explored:
            explored.add(currState)

            # Add all successors to the frontier
            for nextState, action, cost in problem.getSuccessors(currState):
                if nextState not in explored:
                    frontier.push((nextState, currPath + [action]))

    # If no solution is found
    print("No solution found.")
    return []  # Return an empty path if no solution is found





import time
from queue import Queue

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Capture the start time of the search
    start_time = time.time()
    
    currPath = []            # The path that is popped from the frontier in each loop
    currState = problem.getStartState()    # The state(position) that is popped for the frontier in each loop
    print(f"Starting State: {currState}")
    
    # Check if the start state is already the goal
    if problem.isGoalState(currState):     
        return currPath

    # Initialize the frontier queue and the explored set
    frontier = Queue()
    frontier.put((currState, currPath))     # Insert just the start state, in order to pop it first
    explored = set()

    while not frontier.empty():
        currState, currPath = frontier.get()    # Pop a state and the corresponding path
        
        # Check if we've reached the goal state
        if problem.isGoalState(currState):
            # Capture the end time of the search
            end_time = time.time()
            # Calculate the time taken
            elapsed_time = end_time - start_time
            print(f"Goal found: {currState}")
            print(f"Time taken: {elapsed_time:.4f} seconds")
            print(f"Path to goal: {currPath}")
            return currPath
        
        explored.add(currState)
        
        # Get the states reachable from the current state (successors)
        frontierStates = [t[0] for t in frontier.queue]
        
        for s in problem.getSuccessors(currState):
            # If the state has not been explored and is not in the frontier
            if s[0] not in explored and s[0] not in frontierStates:
                frontier.put((s[0], currPath + [s[1]]))      # Add successor and its path to the frontier

    # If no solution is found
    return []       # Return an empty list if no solution is found



# def breadthFirstSearch(problem):
#     """Search the shallowest nodes in the search tree first."""
#     "*** YOUR CODE HERE ***"
#     #util.raiseNotDefined()
#     """ Search the shallowest nodes in the search tree first. """
#     currPath = []           # The path that is popped from the frontier in each loop
#     currState =  problem.getStartState()    # The state(position) that is popped for the frontier in each loop
#     print(f"currState: {currState}")
#     if problem.isGoalState(currState):     # Checking if the start state is also a goal state
#         return currPath

#     frontier = Queue()
#     frontier.push( (currState, currPath) )     # Insert just the start state, in order to pop it first
#     explored = set()
#     while not frontier.isEmpty():
#         currState, currPath = frontier.pop()    # Popping a state and the corresponding path
#         # To pass autograder.py question2:
#         if problem.isGoalState(currState):
#             return currPath
#         explored.add(currState)
#         frontierStates = [ t[0] for t in frontier.list ]
#         for s in problem.getSuccessors(currState):
#             if s[0] not in explored and s[0] not in frontierStates:
#                 # Lecture code:
#                 # if problem.isGoalState(s[0]):
#                 #     return currPath + [s[1]]
#                 frontier.push( (s[0], currPath + [s[1]]) )      # Adding the successor and its path to the frontier

#     return []       # If this point is reached, a solution could not be found.

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
   
from util import PriorityQueue



import time
from util import PriorityQueue  # Assuming util provides the PriorityQueue

def uniformCostSearch(problem):
    """Search the node of least total cost first, optimized for minimal re-expansion."""
    
    # Track the start time
    start_time = time.time()

    frontier = PriorityQueue()  # Priority queue to manage states by cost
    visited = {}  # Dictionary to track visited states and their lowest known costs

    # Push the start state with an empty path and cost 0
    start_state = problem.getStartState()
    print(f"Starting State: {start_state}")
    frontier.push((start_state, []), 0)
    visited[start_state] = 0  # Record the cost of reaching the start state

    while not frontier.isEmpty():
        currState, path = frontier.pop()

        # If this is a goal state, return the path
        if problem.isGoalState(currState):
            # Track the end time
            end_time = time.time()

            # Calculate and print elapsed time
            elapsed_time = end_time - start_time
            print(f"Goal Found: {currState}")
            print(f"Path to Goal: {path}")
            print(f"Time Taken: {elapsed_time:.4f} seconds")
            return path

        # Expand the state only if it hasn't been expanded at a lower cost
        currCost = problem.getCostOfActions(path)
        if currState not in visited or currCost <= visited[currState]:
            visited[currState] = currCost  # Update the cost for this state

            # Expand successors
            for nextState, action, cost in problem.getSuccessors(currState):
                newPath = path + [action]
                newCost = currCost + cost

                # Add to the frontier if this is the first visit or a cheaper path is found
                if nextState not in visited or newCost < visited[nextState]:
                    frontier.push((nextState, newPath), newCost)
                    visited[nextState] = newCost  # Update the cost for this state

    # If no solution is found
    print("No solution found.")
    return []  # Return an empty path if no solution is found





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    from util import Queue,PriorityQueue
    fringe = PriorityQueue()                    # Fringe to manage which states to expand
    fringe.push(problem.getStartState(),0)
    currState = fringe.pop()
    visited = []                                # List to check whether state has already been visited
    tempPath=[]                                 # Temp variable to get intermediate paths
    path=[]                                     # List to store final sequence of directions 
    pathToCurrent=PriorityQueue()               # Queue to store direction to children (currState and pathToCurrent go hand in hand)
    while not problem.isGoalState(currState):
        if currState not in visited:
            visited.append(currState)
            successors = problem.getSuccessors(currState)
            for child,direction,cost in successors:
                tempPath = path + [direction]
                costToGo = problem.getCostOfActions(tempPath) + heuristic(child,problem)
                if child not in visited:
                    fringe.push(child,costToGo)
                    pathToCurrent.push(tempPath,costToGo)
        currState = fringe.pop()
        path = pathToCurrent.pop()    
    return path
    
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
