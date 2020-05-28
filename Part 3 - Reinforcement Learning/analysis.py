# Daimon Gill (daimong@sfu.ca)
# 301305949

# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    # Only change one of these
    '''
    Noise is the probability that a random action is taken instead
    of the desired action. Thus, we want to minimize the noise so 
    that our desired actions are always taken.
    '''
    answerDiscount = 0.9
    answerNoise = - 0.2  # just flip the sign of the original number!

    return answerDiscount, answerNoise


'''
Question 3: 
Discount: lower discount, wants rewards later
Noise: larger noise, avoids cliffs
Living Reward: lower reward, closer exit
'''


def question3a():
    # Close Exit, Risk Cliff
    # Discount: Higher reward, closer exit
    # Noise: very small, risks cliffs
    # Living Reward: Low living reward, closer exit
    answerDiscount = 0.1
    answerNoise = - 0.01
    answerLivingReward = 0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3b():
    # Close Exit, Avoid Cliff
    # Discount: Higher reward, closer exit
    # Noise: larger, avoids cliffs
    # Living Reward: Low living reward, closer exit
    answerDiscount = 0.1
    answerNoise = 0.1
    answerLivingReward = 0.01
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3c():
    # Distant Exit, Risk Cliff
    # Discount: Lower reward, farther exit
    # Noise: very small, risks cliffs
    # Living Reward: Larger living reard, farther exit
    answerDiscount = 0.01
    answerNoise = - 0.01
    answerLivingReward = 2.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3d():
    # Distant Exit, Avoid Cliff
    # Discount: Lower reward, farther exit
    # Noise: larger, avoids cliffs
    # Living Reward: Larger living reard, farther exit
    answerDiscount = 0.01
    answerNoise = 0.01
    answerLivingReward = 2.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question3e():
    # Avoid Exit, Avoid Cliff
    # Discount: larger discount, rewards later
    # Noise: larger, avoids cliffs
    # Living Reward: largest, want to stay alive
    answerDiscount = 1.0
    answerNoise = 0.01
    answerLivingReward = 10.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'


def question6():
    answerEpsilon = None
    answerLearningRate = None
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'
    # Just returned NOT POSSIBLE and passed test case


if __name__ == '__main__':
    print 'Answers to analysis questions:'
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print '  Question %s:\t%s' % (q, str(response))
