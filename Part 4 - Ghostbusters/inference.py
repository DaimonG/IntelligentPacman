# MADE CHANGES
# Daimon Gill
# 301305949
# daimong@sfu.ca

# inference.py
# ------------
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


import itertools
import util
import random
import busters
import game


class InferenceModule:
    """
    An inference module tracks a belief distribution over a ghost's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self, ghostAgent):
        "Sets the ghost agent for later access"
        self.ghostAgent = ghostAgent
        self.index = ghostAgent.index
        self.obs = []  # most recent observation position

    def getJailPosition(self):
        return (2 * self.ghostAgent.index - 1, 1)

    def getPositionDistribution(self, gameState):
        """
        Returns a distribution over successor positions of the ghost from the
        given gameState.

        You must first place the ghost in the gameState, using setGhostPosition
        below.
        """
        ghostPosition = gameState.getGhostPosition(
            self.index)  # The position you set
        actionDist = self.ghostAgent.getDistribution(gameState)
        dist = util.Counter()
        for action, prob in actionDist.items():
            successorPosition = game.Actions.getSuccessor(
                ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observeState(self, gameState):
        "Collects the relevant noisy distance observation and pass it along."
        distances = gameState.getNoisyGhostDistances()
        if len(distances) >= self.index:  # Check for missing observations
            obs = distances[self.index - 1]
            self.obs = obs
            self.observe(obs, gameState)

    def initialize(self, gameState):
        "Initializes beliefs to a uniform distribution over all positions."
        # The legal positions do not include the ghost prison cells in the bottom left.
        self.legalPositions = [
            p for p in gameState.getWalls().asList(False) if p[1] > 1]
        self.initializeUniformly(gameState)

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initializeUniformly(self, gameState):
        "Sets the belief state to a uniform prior belief over all positions."
        pass

    def observe(self, observation, gameState):
        "Updates beliefs based on the given distance observation and gameState."
        pass

    def elapseTime(self, gameState):
        "Updates beliefs for a time step elapsing from a gameState."
        pass

    def getBeliefDistribution(self):
        """
        Returns the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence so far.
        """
        pass


class ExactInference(InferenceModule):
    """
    The exact dynamic inference module should use forward-algorithm updates to
    compute the exact belief function at each time step.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        """
        Updates beliefs based on the distance observation and Pacman's position.

        The noisyDistance is the estimated Manhattan distance to the ghost you
        are tracking.

        The emissionModel below stores the probability of the noisyDistance for
        any true distance you supply. That is, it stores P(noisyDistance |
        TrueDistance).

        self.legalPositions is a list of the possible ghost positions (you
        should only consider positions that are in self.legalPositions).

        A correct implementation will handle the following special case:
          *  When a ghost is captured by Pacman, all beliefs should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition()

             You can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None (a noisy distance
             of None will be returned if, and only if, the ghost is
             captured).
        """
        # print("Entered Q1 Observe")
        # QUESTION 1
        # Redone
        noisyDistance = observation  # estiamted Man. distance to Ghost
        # P(noisyDistance | TrueDistance), dictionary
        # The likliehood of the noisDistance conditioned upon all possible true distances

        emissionModel = busters.getObservationDistribution(noisyDistance)
        # import pprint
        # print("Emission Model:")
        # pprint.pprint(emissionModel)

        pacmanPosition = gameState.getPacmanPosition()
        ghostPositions = self.legalPositions
        # print("Noisy Distance:", noisyDistance)
        # print("Legal Ghost Positons", ghostPositions)
        # print("Pacman position:", pacmanPosition)

        # Replace this code with a correct observation update
        # Be sure to handle the "jail" edge case where the ghost is eaten
        # and noisyDistance is None
        allPossible = util.Counter()

        # Special Case -> we know the Ghost is in Jail
        if noisyDistance is None:
            jail = self.getJailPosition()
            allPossible[jail] = 1.0
            allPossible.normalize()
            self.beliefs = allPossible
            return

        for position in ghostPositions:
            # Manhattan Distance to Ghost
            truedistance2Ghost = util.manhattanDistance(
                position, pacmanPosition)
            if emissionModel[truedistance2Ghost] > 0:
                currentBeliefs = self.beliefs
                newEvidence = emissionModel[truedistance2Ghost]
                allPossible[position] = newEvidence * \
                    currentBeliefs[position]

        # Update Beliefs
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        """
        Update self.beliefs in response to a time step passing from the current
        state.

        The transition model is not entirely stationary: it may depend on
        Pacman's current position (e.g., for DirectionalGhost).  However, this
        is not a problem, as Pacman's current position is known.

        In order to obtain the distribution over new positions for the ghost,
        given its previous position (oldPos) as well as Pacman's current
        position, use this line of code:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        Note that you may need to replace "oldPos" with the correct name of the
        variable that you have used to refer to the previous ghost position for
        which you are computing this distribution. You will need to compute
        multiple position distributions for a single update.

        newPosDist is a util.Counter object, where for each position p in
        self.legalPositions,

        newPostDist[p] = Pr( ghost is at position p at time t + 1 | ghost is at position oldPos at time t )

        (and also given Pacman's current position).  You may also find it useful
        to loop over key, value pairs in newPosDist, like:

          for newPos, prob in newPosDist.items():
            ...

        *** GORY DETAIL AHEAD ***

        As an implementation detail (with which you need not concern yourself),
        the line of code at the top of this comment block for obtaining
        newPosDist makes use of two helper methods provided in InferenceModule
        above:

          1) self.setGhostPosition(gameState, ghostPosition)
              This method alters the gameState by placing the ghost we're
              tracking in a particular position.  This altered gameState can be
              used to query what the ghost would do in this position.

          2) self.getPositionDistribution(gameState)
              This method uses the ghost agent to determine what positions the
              ghost will move to from the provided gameState.  The ghost must be
              placed in the gameState with a call to self.setGhostPosition
              above.

        It is worthwhile, however, to understand why these two helper methods
        are used and how they combine to give us a belief distribution over new
        positions after a time update from a particular position.
        """
        # QUESTION 2
        # Obtain the distribution over new positons for the ghosts, given its previous positon
        # and Pacmans Positon

        # we need to update beliefs in response to a time step

        # Follow similar structure to Q1

        oldGhostPositions = self.legalPositions

        # use allPossible to update beliefs again
        allPossible = util.Counter()

        # loop through positions of the current time step
        for oldPos in oldGhostPositions:
            newPosDist = self.getPositionDistribution(
                self.setGhostPosition(gameState, oldPos))

            # oldProb of the oldPos
            oldProb = self.beliefs[oldPos]

            # loop through key, value pairs as recommended
            # update the probability for the newPos
            for newPos, newProb in newPosDist.items():
                # HMM Lecture, Slide 26
                # P(newPos | previousEvidence) = Sum( Prob(newPos | oldPos) * Prob(oldPos | previousEvidence) )
                calcProb = newProb * oldProb
                allPossible[newPos] += calcProb

        allPossible.normalize()
        self.beliefs = allPossible

    def getBeliefDistribution(self):
        return self.beliefs


class ParticleFilter(InferenceModule):
    """
    A particle filter for approximately tracking a single ghost.

    Useful helper functions will include random.choice, which chooses an element
    from a list uniformly at random, and util.sample, which samples a key from a
    Counter by treating its values as probabilities.
    """

    def __init__(self, ghostAgent, numParticles=300):
        InferenceModule.__init__(self, ghostAgent)
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initializeUniformly(self, gameState):
        """
        Initializes a list of particles. Use self.numParticles for the number of
        particles. Use self.legalPositions for the legal board positions where a
        particle could be located.  Particles should be evenly (not randomly)
        distributed across positions in order to ensure a uniform prior.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        # QUESTION 4
        # print("Question 4 Initialize")
        # self.numParticles for the number of particles
        # self.legalPositions for legal board positions where particle should be located
        self.particles = list()
        numParticles = self.numParticles
        legalPositions = self.legalPositions
        numofPositions = len(legalPositions)

        # print("Number of particles to add: ", numParticles)

        particleCount = 0
        for particle in range(numParticles):
            # Once we have added 5000 particles, we are done
            if particleCount >= numParticles:
                break
            else:
                if particleCount >= numParticles:
                    break
                else:
                    for position in legalPositions:
                        self.particles.append(position)
                        particleCount += 1

        # print("Particles Added:", len(self.particles))
        # Particles are evenly distributed

    def observe(self, observation, gameState):
        """
        Update beliefs based on the given distance observation. Make sure to
        handle the special case where all particles have weight 0 after
        reweighting based on observation. If this happens, resample particles
        uniformly at random from the set of legal positions
        (self.legalPositions).

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell,
             self.getJailPosition()

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeUniformly. The total
             weight for a belief distribution can be found by calling totalCount
             on a Counter object

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.

        You may also want to use util.manhattanDistance to calculate the
        distance between a particle and Pacman's position.
        """
        # QUESTION 4

        noisyDistance = observation

        # emissionModel is the probability of the Ghost being a distance away
        # maps probability to distance
        emissionModel = busters.getObservationDistribution(noisyDistance)

        pacmanPosition = gameState.getPacmanPosition()
        particlePositions = self.legalPositions
        particleDistribution = self.getBeliefDistribution()
        numParticles = self.numParticles
        jail = self.getJailPosition()

        #print("Jail Position:", jail)

        #print("Jail Position:", self.getJailPosition())

        # Check for special case 1
        if noisyDistance is None:
            # All Particles go to jail since we know the ghost is in jail
            self.particles = []
            for num in range(numParticles):
                self.particles.append(jail)
            return

        # Steps in Observation
        # 1. Calculate the weights of all particle
        # 2. If the sum of all weights across all states is zero, reinitialize the particles,
        #      otherwise, normalize the distribution of total weights, and resample from this distribution

        # Step 1
        dist = util.Counter()
        for position in particlePositions:
            # Calculate weights similarly to Q2
            distance2position = util.manhattanDistance(
                position, pacmanPosition)
            dist[position] = emissionModel[distance2position] * \
                particleDistribution[position]

        # Step 2
        if dist.totalCount() == 0:  # Special Case
            self.initializeUniformly(gameState)
        else:
            # Resample
            # Replace our current particle list, loop through
            newParticleList = self.resampleFunc(self.particles, dist)
            self.particles = newParticleList

    def resampleFunc(self, particleList, myDist):
        # Helper Function to do resampling
        # cleans up code slightly
        newParticleList = list()
        for particle in particleList:
            newSample = util.sample(myDist)
            newParticleList.append(newSample)
        return newParticleList

    def elapseTime(self, gameState):
        """
        Update beliefs for a time step elapsing.

        As in the elapseTime method of ExactInference, you should use:

          newPosDist = self.getPositionDistribution(self.setGhostPosition(gameState, oldPos))

        to obtain the distribution over new positions for the ghost, given its
        previous position (oldPos) as well as Pacman's current position.

        util.sample(Counter object) is a helper method to generate a sample from
        a belief distribution.
        """
        # QUESTION 5
        # Look at Piazza Post @141
        # print("Question 5 In elapse Time!")

        # 1. Loop through particles in our particle list
        # 2. for each particle we use setGhostPosition to get the game state for the particle, and then use getPositionDistribution for that gamestate.
        # 3. Sample that new distribution and add that to our list
        newParticleList = list()
        for particle in self.particles:
            ghostGameState = self.setGhostPosition(gameState, particle)
            posDist = self.getPositionDistribution(ghostGameState)
            newSample = util.sample(posDist)
            newParticleList.append(newSample)

        self.particles = newParticleList

    def getBeliefDistribution(self):
        """
        Return the agent's current belief state, a distribution over ghost
        locations conditioned on all evidence and time passage. This method
        essentially converts a list of particles into a belief distribution (a
        Counter object)
        """

        # print("Question 4 getBelief")
        # QUESTION 4
        # Count our particles and normalize the counter
        # Return the counter
        particleCount = util.Counter()

        for particle in self.particles:
            particleCount[particle] += 1

        particleCount.normalize()
        return particleCount


class MarginalInference(InferenceModule):
    """
    A wrapper around the JointInference module that returns marginal beliefs
    about ghosts.
    """

    def initializeUniformly(self, gameState):
        "Set the belief state to an initial, prior value."
        if self.index == 1:
            jointInference.initialize(gameState, self.legalPositions)
        jointInference.addGhostAgent(self.ghostAgent)

    def observeState(self, gameState):
        "Update beliefs based on the given distance observation and gameState."
        if self.index == 1:
            jointInference.observeState(gameState)

    def elapseTime(self, gameState):
        "Update beliefs for a time step elapsing from a gameState."
        if self.index == 1:
            jointInference.elapseTime(gameState)

    def getBeliefDistribution(self):
        "Returns the marginal belief over a particular ghost by summing out the others."
        jointDistribution = jointInference.getBeliefDistribution()
        dist = util.Counter()
        for t, prob in jointDistribution.items():
            dist[t[self.index - 1]] += prob
        return dist


class JointParticleFilter:
    """
    JointParticleFilter tracks a joint distribution over tuples of all ghost
    positions.
    """

    def __init__(self, numParticles=600):
        self.setNumParticles(numParticles)

    def setNumParticles(self, numParticles):
        self.numParticles = numParticles

    def initialize(self, gameState, legalPositions):
        "Stores information about the game, then initializes particles."
        self.numGhosts = gameState.getNumAgents() - 1
        self.ghostAgents = []
        self.legalPositions = legalPositions
        self.initializeParticles()

    def initializeParticles(self):
        """
        Initialize particles to be consistent with a uniform prior.

        Each particle is a tuple of ghost positions. Use self.numParticles for
        the number of particles. You may find the `itertools` package helpful.
        Specifically, you will need to think about permutations of legal ghost
        positions, with the additional understanding that ghosts may occupy the
        same space. Look at the `itertools.product` function to get an
        implementation of the Cartesian product.

        Note: If you use itertools, keep in mind that permutations are not
        returned in a random order; you must shuffle the list of permutations in
        order to ensure even placement of particles across the board. Use
        self.legalPositions to obtain a list of positions a ghost may occupy.

        Note: the variable you store your particles in must be a list; a list is
        simply a collection of unweighted variables (positions in this case).
        Storing your particles as a Counter (where there could be an associated
        weight with each position) is incorrect and may produce errors.
        """
        # QUESTION 6
        # Base this on Question 4
        #print("QUESTION 6 Initialize")
        self.particles = list()
        numParticles = self.numParticles  # 5000?
        ghostPositions = self.legalPositions  # [(x, y)]
        numGhosts = self.numGhosts
        #print("Positions:", ghostPositions)
        #print("Num of Particles:", numParticles)
        #print("Number of Ghosts", numGhosts)

        # Permutation Calculations
        # To compute the product of an iterable with itself, specify the number of repetitions with
        # the optional repeat keyword argument. For example, product(A, repeat=4) means the same as
        # product(A, A, A, A).
        permutations = list(itertools.product(
            ghostPositions, repeat=numGhosts))

        # Shuffle List so its not in order
        random.shuffle(permutations)
        # for perm in permutations:
        # print(perm)
        #print("Permutations: ", permutations)

        # Each particle is a tuple of ghost positions ((x1, y1), (x2, y2)) for two ghosts
        counter = 0
        for num in range(len(permutations)):
            if counter >= numParticles:
                break
            else:
                for perm in permutations:
                    if counter >= numParticles:
                        break
                    else:
                        self.particles.append(perm)
                        counter += 1

        # print("Particles Added:", len(self.particles))

    def addGhostAgent(self, agent):
        """
        Each ghost agent is registered separately and stored (in case they are
        different).
        """
        self.ghostAgents.append(agent)

    def getJailPosition(self, i):
        return (2 * i + 1, 1)

    def observeState(self, gameState):
        """
        Resamples the set of particles using the likelihood of the noisy
        observations.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        A correct implementation will handle two special cases:
          1) When a ghost is captured by Pacman, all particles should be updated
             so that the ghost appears in its prison cell, position
             self.getJailPosition(i) where `i` is the index of the ghost.

             As before, you can check if a ghost has been captured by Pacman by
             checking if it has a noisyDistance of None.

          2) When all particles receive 0 weight, they should be recreated from
             the prior distribution by calling initializeParticles. After all
             particles are generated randomly, any ghosts that are eaten (have
             noisyDistance of None) must be changed to the jail Position. This
             will involve changing each particle if a ghost has been eaten.

        self.getParticleWithGhostInJail is a helper method to edit a specific
        particle. Since we store particles as tuples, they must be converted to
        a list, edited, and then converted back to a tuple. This is a common
        operation when placing a ghost in jail.
        """
        # QUESTION 6
        #print("QUESTION 6 ObserveState")

        pacmanPosition = gameState.getPacmanPosition()
        noisyDistances = gameState.getNoisyGhostDistances()  # noisy distance for each ghost
        if len(noisyDistances) < self.numGhosts:
            return
        emissionModels = [busters.getObservationDistribution(
            dist) for dist in noisyDistances]

        beliefdist = util.Counter()

        # print("Emission Models")
        # import pprint
        # pprint.pprint(emissionModels)
        # print("Noisy Distances", noisyDistances)

        # Base it on Question 4 Observe Function but loop through ghosts
        # 1. Calculate Weights
        # 2. If the sum of all weights across all states is zero, reinitialize the particles,
        #     otherwise, normalize the distribution of total weights, and resample from this distribution

        # Part 1 Calculations
        for particle in self.particles:
            ghostProd = 1
            for i in range(self.numGhosts):
                # Special Case 1
                if noisyDistances[i] is None:
                    # Look at Piazza Post @ 152 for Help
                    # print("Special Case 1")
                    # print("Particle Before:", particle)
                    ghostProd = 1  # we know where the ghost is, so make this 1
                    particle = self.getParticleWithGhostInJail(
                        particle, i)  # Change the Particle to jail
                    # print("Particle After:", particle)
                else:
                    # Look at Piazza Post @ 121 for Help
                    ghostProd *= emissionModels[i][util.manhattanDistance(
                        particle[i], pacmanPosition)]
            beliefdist[particle] += ghostProd

        # Part 2 Checks
        # Special Case 2
        if beliefdist.totalCount() == 0:
            self.specialCase2Func(noisyDistances)
        else:
            # Resample -> Use same as ParticleFilter Class
            newParticleList = self.resampleFunc(self.particles, beliefdist)
            self.particles = newParticleList

    # Two helper functions to use as code was becoming messy and was hard to debug
    def specialCase2Func(self, noisyDistances):
        # print("Special Case 2 Function")
        self.initializeParticles()
        for i in range(self.numGhosts):
            if noisyDistances[i] is None:
                for particle in self.particles:
                    particle = self.getParticleWithGhostInJail(particle, i)

    def resampleFunc(self, particleList, beliefDist):
        # Helper Function that was created for Q4
        # Same Function used from Question 4 Resample Function
        # print("In Resample Function")
        newParticleList = list()
        for particle in particleList:
            newSample = util.sample(beliefDist)
            newParticleList.append(newSample)
        return newParticleList

    def getParticleWithGhostInJail(self, particle, ghostIndex):
        """
        Takes a particle (as a tuple of ghost positions) and returns a particle
        with the ghostIndex'th ghost in jail.
        """
        particle = list(particle)
        particle[ghostIndex] = self.getJailPosition(ghostIndex)
        return tuple(particle)

    def elapseTime(self, gameState):
        """
        Samples each particle's next state based on its current state and the
        gameState.

        To loop over the ghosts, use:

          for i in range(self.numGhosts):
            ...

        Then, assuming that `i` refers to the index of the ghost, to obtain the
        distributions over new positions for that single ghost, given the list
        (prevGhostPositions) of previous positions of ALL of the ghosts, use
        this line of code:

          newPosDist = getPositionDistributionForGhost(
             setGhostPositions(gameState, prevGhostPositions), i, self.ghostAgents[i]
          )

        Note that you may need to replace `prevGhostPositions` with the correct
        name of the variable that you have used to refer to the list of the
        previous positions of all of the ghosts, and you may need to replace `i`
        with the variable you have used to refer to the index of the ghost for
        which you are computing the new position distribution.

        As an implementation detail (with which you need not concern yourself),
        the line of code above for obtaining newPosDist makes use of two helper
        functions defined below in this file:

          1) setGhostPositions(gameState, ghostPositions)
              This method alters the gameState by placing the ghosts in the
              supplied positions.

          2) getPositionDistributionForGhost(gameState, ghostIndex, agent)
              This method uses the supplied ghost agent to determine what
              positions a ghost (ghostIndex) controlled by a particular agent
              (ghostAgent) will move to in the supplied gameState.  All ghosts
              must first be placed in the gameState using setGhostPositions
              above.

              The ghost agent you are meant to supply is
              self.ghostAgents[ghostIndex-1], but in this project all ghost
              agents are always the same.
        """
        # QUESTION 7
        newParticles = []
        # 1. Loop through particles in our particle list
        # 2. Update the particle; we use setGhostPosition to get the game state for the particle, and then use getPositionDistribution for that gamestate.
        # 3. Sample that new distribution and add that to our list

        # Loop Through Particles
        for oldParticle in self.particles:
            newParticle = list(oldParticle)  # A list of ghost positions
            # now loop through and update each entry in newParticle...

            # Based on ParticleFilter class but
            # extend to multiple Ghosts
            for i in range(self.numGhosts):
                # use the old particle in () form
                ghostGameState = setGhostPositions(gameState, newParticle)

                newPosDist = getPositionDistributionForGhost(
                    ghostGameState, i, self.ghostAgents[i])  # get the newPosDist

                newSample = util.sample(newPosDist)  # return a tuple
                # print("New Sample:", newSample)
                newParticle[i] = newSample  # update one entry in the list

            newParticles.append(tuple(newParticle))
        self.particles = newParticles

    def getBeliefDistribution(self):
        # QUESTION 6
        #print("QUESTION 6 getBeliefDistribution")
        dist = util.Counter()
        for particle in self.particles:
            dist[particle] += 1

        # import pprint
        # print("Dist")
        # pprint.pprint(dist)

        dist.normalize()
        return dist


# One JointInference module is shared globally across instances of MarginalInference
jointInference = JointParticleFilter()


def getPositionDistributionForGhost(gameState, ghostIndex, agent):
    """
    Returns the distribution over positions for a ghost, using the supplied
    gameState.
    """
    # index 0 is pacman, but the students think that index 0 is the first ghost.
    ghostPosition = gameState.getGhostPosition(ghostIndex+1)
    actionDist = agent.getDistribution(gameState)
    dist = util.Counter()
    for action, prob in actionDist.items():
        successorPosition = game.Actions.getSuccessor(ghostPosition, action)
        dist[successorPosition] = prob
    return dist


def setGhostPositions(gameState, ghostPositions):
    "Sets the position of all ghosts to the values in ghostPositionTuple."
    for index, pos in enumerate(ghostPositions):
        conf = game.Configuration(pos, game.Directions.STOP)
        gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
    return gameState
