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


from util import manhattanDistance
from game import Directions
import random, util
import math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
      # focusing on eating food.When ghost near don't go,
        newFood = successorGameState.getFood().asList()
        MF = float("inf")
        for food in newFood:
           MF = min(MF, manhattanDistance(newPos, food))

        # avoid ghost if too close
        for ghost in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghost) < 2):
                return -float('inf')
        # reciprocal
        return successorGameState.getScore() + 1.0/MF

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"
        return self.MaxVal(gameState, 0, 0)[0]

    def MaxVal(self, gameState, agentIndex, depth):
        bestAction = ("max",-float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            SA = (action,self.MiniMax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = max(bestAction,SA,key=lambda x:x[1])
        return bestAction

    def MinVal(self, gameState, agentIndex, depth):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            SA = (action,self.MiniMax(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1))
            bestAction = min(bestAction,SA,key=lambda x:x[1])
        return bestAction

    def MiniMax(self, gameState, agentIndex, depth):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.MaxVal(gameState, agentIndex, depth)[1]
        else:
            return self.MinVal(gameState, agentIndex, depth)[1]



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.MaxVal(gameState, 0, 0, -float("inf"), float("inf"))[0]

    def AB(self, gameState, agentIndex, depth, alpha, beta):
        if depth is self.depth * gameState.getNumAgents() \
                or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        if agentIndex is 0:
            return self.MaxVal(gameState, agentIndex, depth, alpha, beta)[1]
        else:
            return self.MinVal(gameState, agentIndex, depth, alpha, beta)[1]

    def MaxVal(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("max",-float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            SA = (action,self.AB(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            bestAction = max(bestAction,SA,key=lambda x:x[1])

            # Prunning
            if bestAction[1] > beta: return bestAction
            else: alpha = max(alpha,bestAction[1])

        return bestAction

    def MinVal(self, gameState, agentIndex, depth, alpha, beta):
        bestAction = ("min",float("inf"))
        for action in gameState.getLegalActions(agentIndex):
            SA = (action,self.AB(gameState.generateSuccessor(agentIndex,action),
                                      (depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            bestAction = min(bestAction,SA,key=lambda x:x[1])

            # Prunning
            if bestAction[1] < alpha: return bestAction
            else: beta = min(beta, bestAction[1])

        return bestAction


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
        "*** YOUR CODE HERE ***"
        # calling expectimax with the depth we are going to investigate
        maxDepth = self.depth * gameState.getNumAgents()
        return self.expectimax(gameState, "expect", maxDepth, 0)[0]

    def expectimax(self, gameState, action, depth, agentIndex):

        if depth is 0 or gameState.isLose() or gameState.isWin():
            return (action, self.evaluationFunction(gameState))

        # if pacman (max agent) - return max successor value
        if agentIndex is 0:
            return self.MaxValue(gameState,action,depth,agentIndex)
        # if ghost (EXP agent) - return probability value
        else:
            return self.expV(gameState,action,depth,agentIndex)

    def MaxValue(self,gameState,action,depth,agentIndex):
        bestAction = ("max", -(float('inf')))
        for legalAction in gameState.getLegalActions(agentIndex):
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            SA = None
            if depth != self.depth * gameState.getNumAgents():
                SA = action
            else:
                SA = legalAction
            SV = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                        SA,depth - 1,nextAgent)
            bestAction = max(bestAction,SV,key = lambda x:x[1])
        return bestAction

    def expV(self,gameState,action,depth,agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)
        AS= 0
        propability = 1.0/len(legalActions)
        for legalAction in legalActions:
            nextAgent = (agentIndex + 1) % gameState.getNumAgents()
            bestAction = self.expectimax(gameState.generateSuccessor(agentIndex, legalAction),
                                         action, depth - 1, nextAgent)
            AS += bestAction[1] * propability
        return (action, AS)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
  # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()

    MF = float('inf')
    for food in newFood:
        MF = min(MF, manhattanDistance(newPos, food))

    gD = 0
    for ghost in currentGameState.getGhostPositions():
        gD = manhattanDistance(newPos, ghost)
        if (gD < 2):
            return -float('inf')

    FL= currentGameState.getNumFood()
    CL = len(currentGameState.getCapsules())

    FLMultiplier = 950050
    CLMultiplier = 10000
    FDMultiplier = 950

    additionalFactors = 0
    if currentGameState.isLose():
        additionalFactors -= 50000
    elif currentGameState.isWin():
        additionalFactors += 50000

    return 1.0/(FL + 1) * FLMultiplier + gD + \
           1.0/(MF + 1) * FDMultiplier + \
           1.0/(CL + 1) * CLMultiplier + additionalFactors

# Abbreviation
better = betterEvaluationFunction
