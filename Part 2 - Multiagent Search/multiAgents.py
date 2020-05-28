# multiAgents.py
# --------------
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

# DONE! READY FOR SUBMISSION! :)

from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        # Scores each of the
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # we are checking the successfor for each legal move available to PacMan

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(
            action)  # new game state for the given action

        pacPos = successorGameState.getPacmanPosition()  # where we will move (x,y)
        newFood = successorGameState.getFood().asList()  # list of (x, y) of food
        newGhostLocations = successorGameState.getGhostPositions()  # where ghost will move
        newScore = successorGameState.getScore()  # use the score as a baseline

        # basic principle: closer the food the better, the closer the ghost the worst
        moveScore = 0

        foodDistances = []
        ghostDistances = []

        # Penalise Pacman for Stopping
        if action == 'Stop':
            moveScore -= 2000

        # Always avoid the losing game state so score it as low as possible
        if successorGameState.isLose():
            return float("-Inf")

        # Food locations
        for foodLoc in newFood:
            foodDistances.append(util.manhattanDistance(pacPos, foodLoc))

        # ghostLocations
        for ghostLoc in newGhostLocations:
            ghostDistances.append(util.manhattanDistance(pacPos, ghostLoc))

        # This is good! We may have just won
        if len(foodDistances) == 0:
            return float("+Inf")

        # Ghost is eaten-> GOOD!
        if len(ghostDistances) == 0:
            return float("+Inf")

        # Minimum Distances to Ghost and Food
        closestGhostDist = min(ghostDistances)
        closestFoodDist = min(foodDistances)

        # Ghost Closer -> Lower Score
        # Food Closer -> Higher Score
        if closestGhostDist < closestFoodDist:  # move away from Ghost
            return newScore + moveScore + (closestGhostDist/50)
        elif closestGhostDist > closestFoodDist:  # move towards Food
            return newScore + moveScore + (closestGhostDist/(10*closestFoodDist))


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        # Based on Pseudo Code from Lecture 10 - Adversarial Search 1
        # pacman is agent 0
        # start at depth 0
        # initial call to miniMax
        retVal = self.miniMax(0, 0, gameState)
        action = retVal[1]

        # Return the action from result
        return action

    # def miniMax(self, gameState, index, depth):
    def miniMax(self, agent, depth, gameState):
        # Terminal states:
        if len(gameState.getLegalActions(agent)) == 0 or depth == self.depth:
            # no action needed to further
            return (self.evaluationFunction(gameState), None)
        if agent == 0:  # Pacman Maximizes
            return self.max(agent, depth, gameState)
        else:  # Ghosts Minimize
            return self.min(agent, depth, gameState)

    def max(self, agent, depth, gameState):
        maxV = float("-Inf")  # super small number
        maxAction = None  # no action
        allAgents = gameState.getNumAgents()

        for action in gameState.getLegalActions(agent):
            # Check if we go to next depth
            whatAgent = agent + 1
            if not (whatAgent % allAgents):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agent + 1
                nextDepth = depth

            ret = self.miniMax(
                nextAgent, nextDepth, gameState.generateSuccessor(agent, action))

            if ret[0] > maxV:
                maxV = ret[0]
                maxAction = action

        return (maxV, maxAction)

    def min(self, agent, depth, gameState):
        val = float("+Inf")
        act = None
        allAgents = gameState.getNumAgents()

        for actions in gameState.getLegalActions(agent):
            # Check if we go to next depth
            whatAgent = agent + 1  # what would the next agent be?
            if not (whatAgent % allAgents):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agent + 1
                nextDepth = depth

            ret = self.miniMax(
                nextAgent, nextDepth, gameState.generateSuccessor(agent, actions))

            if ret[0] < val:
                val = ret[0]
                act = actions

        return (val, act)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Based on PseudoCode from:
        # Adversarial Search Lecture

        # return action

        # util.raiseNotDefined()
        # Start with some default alpha beta values
        alpha = float("-Inf")
        beta = float("+Inf")

        return self.alphaBeta(0, 0, alpha, beta, gameState)[1]

    def alphaBeta(self, agent, depth, alpha, beta, gameState):
        # return (value, action)
        # Terminating State
        if len(gameState.getLegalActions(agent)) == 0 or depth == self.depth:
            # (value, action)
            # State weve reached is a leaf, so theres no action to take
            return (self.evaluationFunction(gameState), None)

        # Ghosts Minimize
        # Pacman Maximizes
        if agent == 0:  # Pacman
            return self.max(agent, depth, alpha, beta, gameState)
        else:  # Ghosts
            return self.min(agent, depth, alpha, beta, gameState)

    def max(self, agent, depth, alpha, beta, gameState):
        # return (value, action)
        maxV = float("-Inf")  # Some super small number
        maxAction = None  # Start with nothing
        allAgents = gameState.getNumAgents()

        # For each move pacman can make
        for action in gameState.getLegalActions(agent):
            # Check to see if move to next depth
            whatAgent = agent + 1  # what would the next agent be?
            if not (whatAgent % allAgents):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agent + 1
                nextDepth = depth

            newValues = self.alphaBeta(
                nextAgent, nextDepth, alpha, beta, gameState.generateSuccessor(agent, action))
            newMax = newValues[0]

            # Found new Max
            if newMax > maxV:
                maxV = newMax
                maxAction = action

            # PRUNING
            if maxV > beta:
                return (maxV, maxAction)

            # Update Alpha
            alpha = max(alpha, maxV)

        return (maxV, maxAction)

    def min(self, agent, depth, alpha, beta, gameState):
        # return (value, action)
        minV = float("+Inf")
        minAction = None  # Need to find the action
        allAgents = gameState.getNumAgents()

        # for each move a Ghost can make
        for action in gameState.getLegalActions(agent):
            # Check to see if move to next depth
            whatAgent = agent + 1  # what would the next agent be?
            if not (whatAgent % allAgents):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agent + 1
                nextDepth = depth

            newValues = self.alphaBeta(
                nextAgent, nextDepth, alpha, beta, gameState.generateSuccessor(agent, action))
            newMin = newValues[0]

            # Found New Min
            if newMin < minV:
                minV = newMin
                minAction = action

            # PRUNING
            if minV < alpha:
                return (minV, minAction)

            # Update Beta
            beta = min(beta, minV)

        return (minV, minAction)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        # Based on Pseudocode from Expectimax Lecture
        # returns action
        retVal = self.expectiMax(0, 0, gameState)  # (value, actiom)

        return retVal[1]

    def expectiMax(self, agent, depth, gameState):
        # returns (value, action)
        # Termination States
        if len(gameState.getLegalActions(agent)) == 0 or depth == self.depth:
            # State weve reached is a leaf, so theres no action to take
            return (self.evaluationFunction(gameState), None)
        if agent == 0:  # Pacman
            return self.max(agent, depth, gameState)
        else:  # Ghosts
            return self.exp(agent, depth, gameState)

    # Same MAX function from Minimax

    def max(self, agent, depth, gameState):
        maxV = float("-Inf")  # super small number
        maxAction = None  # no action
        allAgents = gameState.getNumAgents()

        for action in gameState.getLegalActions(agent):
            # Check if we go to next depth
            whatAgent = agent + 1
            if not (whatAgent % allAgents):
                nextAgent = 0  # Pacman
                nextDepth = depth + 1  # Next Depth
            else:
                nextAgent = agent + 1  # next Ghost
                nextDepth = depth  # same depth

            ret = self.expectiMax(
                nextAgent, nextDepth, gameState.generateSuccessor(agent, action))

            if ret[0] > maxV:
                maxV = ret[0]
                maxAction = action

        return (maxV, maxAction)

    def exp(self, agent, depth, gameState):
        # returns (value, action)
        expV = 0
        expAction = None
        legalActions = gameState.getLegalActions(agent)
        allAgents = gameState.getNumAgents()
        # Each step has equal probability
        probability = 1.0 / len(legalActions)

        for action in legalActions:
            whatAgent = agent + 1
            if not (whatAgent % allAgents):
                nextAgent = 0
                nextDepth = depth + 1
            else:
                nextAgent = agent + 1
                nextDepth = depth

            ret = self.expectiMax(nextAgent, nextDepth,
                                  gameState.generateSuccessor(agent, action))

            expV += probability*ret[0]

        return (expV, expAction)


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 

      The basic premise of the below function is to maximize distance between pacman and 
      ghosts while also prioritizing minimizing distance between Pacman and food as well 
      as capsules. Here, we prioritize eating food over eating capsules. We have a 
      dangerzone around Pacman that prevents Ghosts from coming too close. If a ghost comes
      within the dangerzone, we no longer focus on minimizing distance between
      Pacman and food, but rather focus on increasing distance between Pacman and Ghosts.

    """
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostPositions()
    capsules = currentGameState.getCapsules()

    foodCount = len(food)
    ghostCount = len(ghosts)
    capsuleCount = len(capsules)

    ghostDangerZone = 3

    distance2food = []
    distance2ghost = []
    distance2capsules = []
    pacman = currentGameState.getPacmanPosition()

    for fd in food:
        distance2food.append(manhattanDistance(pacman, fd))

    for ghost in ghosts:
        distance2ghost.append(manhattanDistance(pacman, ghost))

    for cap in capsules:
        distance2capsules.append(manhattanDistance(pacman, cap))

    closestFood = 0
    if foodCount:
        closestFood = min(distance2food)

    closestGhost = ghostDangerZone
    if ghostCount:
        closestGhost = min(distance2ghost)

    # decide if we run from ghost or run to food
    if closestGhost < ghostDangerZone:
        return (20*currentGameState.getScore()) - (10*foodCount) - capsuleCount
    else:
        return (20*currentGameState.getScore()) - (20*closestFood) - (10*foodCount) - (capsuleCount)


# Abbreviation
better = betterEvaluationFunction
