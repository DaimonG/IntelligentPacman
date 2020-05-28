# Daimon Gill (daimong@sfu.ca)
# 301305949

# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random
import util
import math


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        # keep track of what states we have visited and QValues
        # (state, action) as keys
        self.states = dict()  # could use util.counter as well?

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) not in self.states:
            return 0.0  # never seen so return 0
        else:
            return self.states[(state, action)]  # lookup in dictionary

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """

        legalActions = self.getLegalActions(state)

        # Return the maximum QValue
        # Based on code written in Q1

        # Terminal States have no legal actions
        if len(legalActions) == 0:
            return 0.0
        else:
          # Get all possible QVals
            qVals = []
            for action in legalActions:
                qVals.append(self.getQValue(state, action))

            # Return the max QVal
            maxQval = max(qVals)
            return maxQval

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        # Based on code from Q1 function
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        if len(legalActions) == 0:
            return action
        else:
            valAndAction = []
            # Find the maximum QVal and return the corresponding action
            for action in legalActions:
                qVal = self.getQValue(state, action)
                newTuple = (qVal, action)
                valAndAction.append(newTuple)

            maxTuple = max(valAndAction)
            return maxTuple[1]

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None

        prob = self.epsilon

        if len(legalActions) == 0:
            return action
        else:
            randomProb = util.flipCoin(prob)
            # Choose random action
            if randomProb:
                action = random.choice(legalActions)
            else:
                action = self.computeActionFromQValues(state)

            return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        # Implement Update Formula:
        # QVal = (1 - alpha) * QVal + Alpha*(Reward + discount*nextQVal)
        A = self.alpha
        D = self.discount
        R = reward

        nextQVal = self.getValue(nextState)
        currentQVal = self.getQValue(state, action)
        updateQVal = (1 - A) * currentQVal + \
            A * (R + D * nextQVal)

        # Update
        self.states[(state, action)] = updateQVal

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05, gamma=0.8, alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        # Q(s,a) = SUM feature * weight
        featuresDict = self.featExtractor.getFeatures(state, action)

        #import numpy
        # return numpy.dot(self.getWeights(), featuresDict)  # dotproduct

        return self.getWeights() * featuresDict  # dot product

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        # Implement Update Formula
        # weight[i] = weight[i] + alpha * difference * feature[i]
        # difference = (reward + discount * nextQVal) - QVal

        nextQVal = self.getValue(nextState)
        currentQVal = self.getQValue(state, action)

        D = self.discount
        A = self.alpha

        difference = (reward + D*nextQVal) - currentQVal

        features = self.featExtractor.getFeatures(state, action)
        weights = self.getWeights()

        # update weights
        for feature in features:
            # print(feature)
            weights[feature] += A * difference * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            # made no changes to this function but tests are being passed
            pass
