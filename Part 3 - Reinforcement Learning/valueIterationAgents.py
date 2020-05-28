# Daimon Gill (daimong@sfu.ca)
# 301305949
# valueIterationAgents.py
# -----------------------
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


import mdp
import util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates() -> States
              mdp.getPossibleActions(state) -> Actions for a given state
              mdp.getTransitionStatesAndProbs(state, action) -> Transition Function (state, probability)
              mdp.getReward(state, action, nextState) -> Reward of starting at state, doing action, ending at nextState
              mdp.isTerminal(state) -> Checks if a state is terminal
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()
        states = mdp.getStates()

        # Val = max QVal
        for iteratate in range(iterations):
            # Use temp values, update actual values at each iteration of the loop
            loopValues = util.Counter()
            for state in states:
                qVals = []

                # For the actions of a possible
                possibleActions = self.mdp.getPossibleActions(state)
                for action in possibleActions:
                    # Find the QVal for each action
                    qVal = self.computeQValueFromValues(state, action)
                    qVals.append(qVal)

                # Need to account for terminal states -> they get a value of 0
                if mdp.isTerminal(state):
                    loopValues[state] = 0
                else:
                    # Only want the max QVal for this state
                    loopValues[state] = max(qVals)

            self.values = loopValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # IMPLEMENT THIS FUNCTION
        # util.raiseNotDefined()

        Qval = 0

        transitionFunc = self.mdp.getTransitionStatesAndProbs(
            state, action)  # (state, probability)

        for transition in transitionFunc:
            # implement Qval = Probability * (Reward + Discount*Value(transitionState))
            Qval += transition[1] * (self.mdp.getReward(state, action,
                                                        transition[0]) + (self.discount * self.getValue(transition[0])))

        return Qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # What is the best action based on QVal?
        # util.raiseNotDefined()

        # Terminal States Have No Action
        retAction = None
        maxVal = float("-Inf")  # start comparing with some tiny number
        maxValandAction = (maxVal, retAction)
        # Possible Actions
        possibleActions = self.mdp.getPossibleActions(state)

        if self.mdp.isTerminal(state):
            return maxValandAction[1]
        else:
            qVals = []

            for action in possibleActions:
                qVal = self.computeQValueFromValues(state, action)
                valnAct = (qVal, action)
                qVals.append(valnAct)

            # Get the action with the highest value
            bestTuple = max(qVals)
            return bestTuple[1]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
