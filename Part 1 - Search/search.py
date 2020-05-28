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

#####################################################
#####################################################
# Please enter the number of hours you spent on this
# assignment here
"""
num_hours_i_spent_on_this_assignment = 22.5
Initial Understanding: 3
Q1.1: 2.5
Q1.2: 2
Q1.3: 2
Q1.4: 4

Q2.1: 3
Q2.2: 6

I am new to Python (in the Engineering program we do not learn Python), so the only knowledge that I
have in python is self-taught; this is partly why it took me so long to gain an initial understanding
of all of the files and functions. Q1.1, Q1.2, and Q1.3 were relatively straightforward as we were
given adequate pseudo-code in lectures/textbook. I struggled with Q1.4 when implementing the Priority Queue
in place of a stack for DFS. 

Q2.1 was relatively straight forward as we could base a lot of code and functionality to code for the SearchProblem. 
Q2.2 was more difficult and I struggled implementing an admissable heuristic that expanded few nodes. This was a lot of 
tinkering before I found and algorithm that worked well. 
"""
#
#####################################################
#####################################################

#####################################################
#####################################################
# Give one short piece of feedback about the course so far. What
# have you found most interesting? Is there a topic that you had trouble
# understanding? Are there any changes that could improve the value of the
# course to you? (We will anonymize these before reading them.)
"""
I have thoroughly enjoyed this course so far. Now that we are beginning more interesting topics outside of 
initial search problems, it is even more interesting and I am now able to see the beginnings of AI. 
One thing that I find incredibly helpful is the inclusion of videos in our PPTs as that helps me visualize
how an algorithm should be performing. 

"""
#####################################################
#####################################################

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""




import util
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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Q1.1
    Search the deepest nodes in the search tree first.
    print ( problem.getStartState() )
    You will get (5,5) -> use for root node

    print (problem.isGoalState(problem.getStartState()) ) -> check if were at a goal state
    You will get False

    print ( problem.getSuccessors(problem.getStartState()) )
    You will get [((x1,y1),'South',1),((x2,y2),'West',1)] -> use to expand nodes
    """
    # Graph Search Algorithm from textbook: R&N 3ed Section 3.3, Figure 3.7
    # return solution or failure
    # solution is a path to a goal node in a list of actions [south, south, west]
    # failure is an empty list []

    ###################################
    ### BEGINNING OF IMPLEMENTATION ###
    ###################################

    # data structures
    exploredNodes = list()  # use dictionary to have: node : path to reach node
    fringe = util.Stack()  # LIFO

    # start state is the first node in our graph
    rootNode = problem.getStartState()

    # print("Root node:", rootNode)
    # print("Root node type:", type(rootNode))

    # [] because no actions needed to go to start state
    # always push start state to the fringe
    rootTuple = (rootNode, [])
    fringe.push(rootTuple)

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []  # failure

        poppedNode = fringe.pop()  # remove a node from the fringe
        # print("Leaf Node:", leafNode)
        node = poppedNode[0]
        path2node = poppedNode[1]

        # print("Path to leaf:", leafPath)

        # we only do the following if we havent explored this node from the fringe already
        # if node not in exploredNodes:

        # print("Explored nodes before expansion:", exploredNodes)

        # print("At goal state?", goalStateTF)
        if problem.isGoalState(node):  # always check if at goal state
            print("We found a goal state!")
            return path2node

        if node not in exploredNodes:
            exploredNodes.append(node)
            # expand available nodes
            expandedNodes = problem.getSuccessors(node)
            # (location, moves, distance)
            # print("Expanded nodes: ", expandedNodes)
            for nodes in expandedNodes:
                if nodes not in exploredNodes:
                    child = nodes[0]
                    path = nodes[1]
                    childPath = [path]
                    # check if expanded nodes are in explored nodes
                    # print("Node:", location)
                    # print("Path to Node:", pathToExpanded)
                    fringeTuple = (child, path2node + childPath)
                    # print("Fringe Tuple:", fringeTuple)
                    fringe.push(fringeTuple)

    return []  # failure


def breadthFirstSearch(problem):
    """
    Q1.2
    Search the shallowest nodes in the search tree first."""
    # Based on pseudo-code from R&N 3ed Section 3.4, Figure 3.11
    # Function returns failure or a solution
    # function is called by Q2.1

    ###################################
    ### BEGINNING OF IMPLEMENTATION ###
    ###################################

    # data structures
    # exploredNodes = dict()
    exploredNodes = list()
    fringe = util.Queue()  # FIFO
    # print("USING MY BFS IN Q2.1!")

    # always start at root node
    rootNode = problem.getStartState()
    # print("Start State: ", rootNode)

    if problem.isGoalState(rootNode):
        print("The root node is a solution!")
        return []

    # instantiate fringe by pushing root nodes
    rootTuple = (rootNode, [])
    # print("Root Tuple:", rootTuple)
    fringe.push(rootTuple)

    while not fringe.isEmpty():

        if fringe.isEmpty():
            return []  # failure

        # We look at the first node in the fringe (FIFO)
        poppedNode = fringe.pop()
        node = poppedNode[0]  # (x, y)
        path2node = poppedNode[1]  # ex. ['South', 'West, 'North']
        # print("Node", path2node)

        # did we just pop off the solution?
        if problem.isGoalState(node):
            # print("We found our goal path:", path2node)
            return path2node

        # begin exploring the node
        if node not in exploredNodes:
            exploredNodes.append(node)  # add to list
            # exploredNodes[node] = path2node
            # expand child nodes
            successorNodes = problem.getSuccessors(node)
            for childNode in successorNodes:
                if childNode not in exploredNodes:
                    child = childNode[0]  # location (x, y)
                    path = childNode[1]  # how to get there

                    # add the child to the fringe
                    childPath = [path]
                    path2child = path2node + childPath
                    # print("Path 2 child:", path2child)
                    fringeTuple = (child, path2child)
                    fringe.push(fringeTuple)

    if fringe.isEmpty():
        return []  # failure


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """
    Q1.3
    Search the node that has the lowest combined cost and heuristic first."""
    """Call heuristic(s,problem) to get h(s) value."""
    # Based on pseudo-code from R&N 3ed Section 3.4, Figure 3.14
    # Adapted from UCS to A* Search

    ###################################
    ### BEGINNING OF IMPLEMENTATION ###
    ###################################

    # data structures
    # priority queue where the priority is the cost
    # pops the lowest cost node first
    fringe = util.PriorityQueue()
    exploredNodes = list()

    # Code to understand priority Queue
    # firstTuple = ((1,1), [], 0)
    # secondTuple = ((2,2), [], 0)
    # thirdTuple = ((3,3), [], 0)

    # fringe.push(firstTuple, 2)
    # fringe.push(secondTuple, 1)
    # fringe.push(thirdTuple, 0)

    # print(fringe.pop())
    # print(fringe.pop())
    # print(fringe.pop())
    # prints: thirdTuple, secondTuple, firstTuple

    # now we need to keep track of cost to reach a node, so we add this to our tuple
    rootNode = problem.getStartState()
    rootTuple = (rootNode, [], 0)  # location, moves2reach, cost2reach
    fringe.push(rootTuple, 0)  # assign zero priority to begin
    # print("Fringe: ", fringe)

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []  # failure

        popped = fringe.pop()
        # print("Popped:", popped)
        node = popped[0]  # location (x, y)
        path2node = popped[1]  # how to reach
        bwCost = popped[2]  # cost to reach
        # print("Path:", path2node)
        # print("Cost:", bwCost)

        if problem.isGoalState(node):  # we've found a goal state
            return path2node

        if node not in exploredNodes:
            exploredNodes.append(node)
            childNodes = problem.getSuccessors(node)  # expand nodes
            for children in childNodes:  # for each child
                if children not in exploredNodes:
                    child = children[0]  # location
                    path2child = path2node + \
                        [children[1]]  # path to reach child
                    # cost to the child node = cost to parent + cost to child from parent
                    bwcost2child = bwCost + children[2]
                    # heauristic of child
                    forwardCost = heuristic(child, problem)
                    # f(n) = g(h) + h(n)
                    totalCost = bwcost2child + forwardCost
                    fringeTuple = (child, path2child, bwcost2child)
                    fringe.update(fringeTuple, totalCost)

    return []  # failure


def priorityQueueDepthFirstSearch(problem):
    """
    Q1.4a.
    Reimplement DFS using a priority queue.
    """
    # DFS with a Queue is just UCS!
    # Implemented by making the queue behave as a stack
    # Based on pseudo-code from R&N 3ed Section 3.3, Figure 3.14
    ###################################
    ### BEGINNING OF IMPLEMENTATION ###
    ###################################

    # data structures
    exploredNodes = list()
    fringe = util.PriorityQueue()  # LIFO
    tempStack = util.Stack()

    # start state is the first node in our graph
    rootNode = problem.getStartState()

    # print("Root node:", rootNode)
    # print("Root node type:", type(rootNode))
    # states are described by a tuple (x, y)

    # [] because no actions needed to go to start state
    # always push start state to the fringe
    rootTuple = (rootNode, [], 0)
    fringe.push(rootTuple, 0)

    while not fringe.isEmpty():
        poppedNode = fringe.pop()

        # print("Leaf Node:", leafNode)
        leafNode = poppedNode[0]
        path2node = poppedNode[1]
        cost2node = poppedNode[2]
        len2node = len(path2node)
        # print("Cost to node: ", cost2node)

        # print("Path to leaf:", leafPath)

        # we only do the following if we havent explored this node from the fringe already
        if leafNode not in exploredNodes:
            exploredNodes.append(leafNode)
            # print("Explored nodes before expansion:", exploredNodes)

            goalStateTF = problem.isGoalState(leafNode)
            # print("At goal state?", goalStateTF)
            if goalStateTF:  # always check if at goal state
                print("We found a goal state!")
                return path2node

            # expand available nodes
            # expandedNodes = reversed(problem.getSuccessors(leafNode))
            expandedNodes = problem.getSuccessors(leafNode)
            #print("Expended Nodes:", expandedNodes)
            # (location, moves, distance)
            # print("Expanded nodes: ", expandedNodes)

            for children in reversed(expandedNodes):
                if children not in exploredNodes:
                    child = children[0]  # (x, y)
                    path2child = path2node + [children[1]]  # []
                    cost2child = cost2node + children[2]

                    # total cost to get to child
                    bwCost2Child = len(path2child)
                    fringeTuple = (child, path2child, bwCost2Child)
                    fringe.update(fringeTuple, bwCost2Child*-1)

    return []  # failure


def priorityQueueBreadthFirstSearch(problem):
    """
    Q1.4b.
    Reimplement BFS using a priority queue.
    """
    # Copied BFS code from above and made changes to use the priority queue
    ###################################
    ### BEGINNING OF IMPLEMENTATION ###
    ###################################

    # data structures
    fringe = util.PriorityQueue()
    exploredNodes = dict()

    # start at root node
    rootNode = problem.getStartState()
    rootTuple = (rootNode, [], 0)  # (x,y), [], cost
    fringe.push(rootTuple, 0)

    while not fringe.isEmpty():
        if fringe.isEmpty():
            return []  # failure

        popped = fringe.pop()
        # print(popped)
        node = popped[0]  # (x, y)
        path2node = popped[1]  # []
        cost2node = popped[2]

        if problem.isGoalState(node):
            return path2node

        if node not in exploredNodes:
            exploredNodes[node] = path2node  # node : path2node

            childNodes = problem.getSuccessors(node)  # expand nodes

            for children in childNodes:
                if children not in exploredNodes:
                    child = children[0]  # (x, y)
                    path2child = path2node + [children[1]]  # []
                    cost2child = cost2node + children[2]

                    bwCost2Child = cost2node + cost2child  # total cost to get to child
                    fringeTuple = (child, path2child, bwCost2Child)
                    fringe.update(fringeTuple, bwCost2Child)

    return []  # failure


#####################################################
#####################################################
# Discuss the results of comparing the priority-queue
# based implementations of BFS and DFS with your original
# implementations.

"""
DFS Statistics------------------------

Medium Maze:
Total Cost: 130 in 0.0 seconds
Nodes Expanded: 146
Score: 380.0

Medium Maze w/ Priority Queue:
Total Cost: 130 in 0.1 seconds
Nodes Expanded: 146
Score: 346.0

Big Maze:
Total Cost: 210 in 0.0 seconds
Nodes Expanded: 390
Score: 300.0

Big Maze w/ Priority Queue:
Total Cost: 210 in 0.0 seconds
Nodes Expanded: 390
Score: 300.0

BFS Statistics------------------------

Medium Maze:
Total Cost: 68 in 0.0 seconds
Nodes Expanded: 269 nodes
Score: 442.0

Medium Maze w/ Priority Queue:
Total Cost: 68 in 0.0 seconds
Nodes Expanded: 269 nodes
Score: 442.0

Big Maze:
Total Cost: 210 in 0.0 seconds
Nodes Expanded: 620 nodes
Score: 300.0

Big Maze w/ Priority Queue:
Total Cost: 210 in 0.0 seconds
Nodes Expanded: 620 seconds
Score: 300.0

Comments:------------------------------
We are able to see that the results are exactly the same when using a priority queue in place of 
a queue or a stack. It should be noted that for this to work correctly, we change our DFS algorithm
to use a queue in the same way it would use a stack. 

We know that when using a stack or a queue we avoid the log(n) overhead associated with a priority
queue; this is one reason to continue to use a stack or a queue. 

Consequently, we can see that a priority queue is a fine idea to use with BFS but to use with DFS
changes must be made to the algorithm.

"""


#####################################################
#####################################################


# Abbreviations (please DO NOT change these.)
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
bfs2 = priorityQueueBreadthFirstSearch
dfs2 = priorityQueueDepthFirstSearch
