from util import manhattanDistance
from game import Directions
import random, util
import sys
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
        newGhostPositions = successorGameState.getGhostPositions()
        score = 0.0
        successorGameState.data.score = 0.0
        import sys
        minDist = sys.maxsize

        if len(currentGameState.getFood().asList()) > len(newFood.asList()):
            score += 100
        if newPos in newGhostPositions:
            score -= 1000
        for food in newFood.asList():
            minDist = min(minDist, manhattanDistance(newPos, food))
        score += 2/(0.1+minDist)
        for ghost in newGhostPositions:
            score -= 1/(0.1+manhattanDistance(newPos, ghost))
        for i in range(len(newGhostPositions)):
            ghostPos = newGhostPositions[i]
            unScared = True
            if newScaredTimes[i] > 0:
                unScared = False
            d = manhattanDistance(ghostPos, newPos)
            if (not unScared) and d == 0:
                score += 10000
        successorGameState.data.score = score
        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getEvaluation(self,gameState,total_layer, layer):
        
        agentIdx = layer % gameState.getNumAgents()
        
        import sys
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        elif gameState.isLose():
            return self.evaluationFunction(gameState)
        #print(layer, actions, gameState.isWin(),gameState.isLose())
        vals = [] 
        actions = gameState.getLegalActions(agentIdx)
        for a in actions:
            newState = gameState.generateSuccessor(agentIdx, a)  
            if layer == total_layer-1:
                vals.append(self.evaluationFunction(newState))           
            else:
                vals.append(self.getEvaluation(newState, total_layer, layer+1))
                
        #print(layer, actions, vals)
        if agentIdx == 0:         
            return max(vals)
        else:
            return min(vals)
    
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        action = None
        val = None
        for a in actions:
            newState = gameState.generateSuccessor(0,a)
            v = self.getEvaluation(newState, self.depth * gameState.getNumAgents(), 1)
            if action == None or v > val:
                action = a 
                val = v
        return action
        "*** YOUR CODE HERE ***"
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def getPrunedEvaluation(self, gameState, total_layer, layer, alpha, beta):
        agentIdx = layer % gameState.getNumAgents()
        if layer == total_layer or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(agentIdx)
        v = sys.maxsize
        if agentIdx == 0:
            v = -sys.maxsize
        
        for a in actions:
            newState = gameState.generateSuccessor(agentIdx, a)            
            val  = self.getPrunedEvaluation(newState, total_layer, layer+1, alpha, beta)
            
            if agentIdx == 0:
                v = max(v,val)
                if v > beta:
                    return v
                alpha = max(alpha, v)
            else:
                v = min(v, val)
                if v < alpha:
                    return v
                beta = min(beta, v)
        
        return v
    
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys
        actions = gameState.getLegalActions(0)
        action = None
        val = None
        alpha = -sys.maxsize
        beta = sys.maxsize
        for a in actions:
            newState = gameState.generateSuccessor(0, a)
            v = self.getPrunedEvaluation(newState, self.depth * gameState.getNumAgents(), 1, alpha, beta)
            alpha = max(alpha, v)
            if action == None or v > val:
                action = a
                val = v
        return action
        "*** YOUR CODE HERE ***"

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getExp(self, gameState, total_layer, layer):

        agentIdx = layer % gameState.getNumAgents()

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
    
        vals = []
        actions = gameState.getLegalActions(agentIdx)
        for a in actions:
            newState = gameState.generateSuccessor(agentIdx, a)
            if layer == total_layer-1:
                vals.append(self.evaluationFunction(newState))
            else:
                vals.append(self.getExp(newState, total_layer, layer+1))

        #print(layer, actions, vals)
        if agentIdx == 0:
            return max(vals)
        else:
            return sum(vals)/len(vals)
    
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        actions = gameState.getLegalActions(0)
        action = None
        val = None
        for a in actions:
            newState = gameState.generateSuccessor(0, a)
            v = self.getExp(newState, self.depth * gameState.getNumAgents(), 1)
            if action == None or v > val:
                action = a
                val = v
        return action
        "*** YOUR CODE HERE ***"

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    GhostPositions = currentGameState.getGhostPositions()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Capsules = currentGameState.getCapsules()
    score = 0.0
    minDist = -sys.maxsize
    capsuleManhattan = 0.0
    
    #score += len(currentGameState.getLegalActions())*1
    score -= len(Capsules)*20
    score -= len(Food.asList())*10
    
    for i in range(len(GhostPositions)):
        ghostPos = GhostPositions[i]
        unScared = True
        if ScaredTimes[i] > 0:
            unScared = False
        d = manhattanDistance(ghostPos, Pos)
        if unScared and d == 2:
            score -= 500
        if unScared and d == 1:
            score -= 1000
        if unScared and d == 0:
            score -= 10000
        #if (not unScared) and d == 0:
        #    score += 1000

    for food in Food.asList():
        score += 1/(manhattanDistance(Pos, food)+0.1)
        minDist = min(minDist, manhattanDistance(Pos, food))
    score += 5/(minDist+0.1)
    for capsule in Capsules:
        score += 2/(manhattanDistance(Pos, capsule)+0.1)
    
    return score
    "*** YOUR CODE HERE ***"   
    
# Abbreviation
better = betterEvaluationFunction
