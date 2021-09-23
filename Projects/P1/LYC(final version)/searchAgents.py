"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem
      

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        return (self.startingPosition[0], self.startingPosition[1], [0, 0, 0, 0])

    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """
        x, y, c = state
        if (x, y) in self.corners and sum(c) == 4:
            return True
        return False
        

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y, c = state
            nextc = c.copy()
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                if (nextx, nexty) in self.corners:
                    i = self.corners.index((nextx, nexty))
                    nextc[i] = 1
                nextState = (nextx, nexty, nextc)
                cost = 1
                successors.append([nextState, action, cost])
        
        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

def compute_dist_sum(c, corners):
    unvisited_corners = []
    for i in range(4):
        if c[i] == 0:
            unvisited_corners.append(corners[i])
    if len(unvisited_corners) == 1:
        return 0
    elif len(unvisited_corners) == 2:
        return util.manhattanDistance(unvisited_corners[0], unvisited_corners[1])
    elif len(unvisited_corners) == 3:
        return max(corners, key = lambda x: x[1])[1] - 1 +\
            max(corners, key=lambda x: x[0])[0] - 1
    else:
        return 2 * (max(corners, key=lambda x: x[1])[1] - 1) +\
            max(corners, key=lambda x: x[0])[0] - 1

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    x, y, c = state
    if sum(c) == 4:
        return 0
    import sys
    h = sys.maxsize
    for i in range(4):
        xy1 = (x, y)
        xy2 = corners[i]
        if c[i] == 1:
            continue
        manhattan = util.manhattanDistance(xy1, xy2)
        h = min(h, manhattan)
    h += compute_dist_sum(c, corners)
    return h

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem
'''
# can achieve 7583 nodes expansion
# half circumference of the minimum rectangle containing all food 
# + dist to the closest food
def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    position, foodGrid = state
    h = 0
    foodlist = foodGrid.asList()
    endpoints = []
    if len(foodlist) > 0:
        xmax = max(foodlist, key=lambda x: x[0])[0]
        xmin = min(foodlist, key=lambda x: x[0])[0]
        ymax = max(foodlist, key=lambda x: x[1])[1]
        ymin = min(foodlist, key=lambda x: x[1])[1]
        h = (xmax - xmin)+ (ymax - ymin)
        # go to the nearest edge
        endpoints.append((xmin, ymin))
        endpoints.append((xmax, ymax))
        endpoints.append((xmin, ymax))
        endpoints.append((xmax, ymin))
        import sys
        dist1 = sys.maxsize
        for point in foodlist: #set(endpoints):
            xy1 = position
            xy2 = point
            dist1 = min(dist1, util.manhattanDistance(xy1, xy2))  
        dist2 = sys.maxsize
        for point in set(endpoints):
            xy1 = position
            xy2 = point
            dist2 = min(dist2, util.manhattanDistance(xy1, xy2))
        h += max(dist1, dist2)
    return h
'''

from util import manhattanDistance as M
def foodHeuristic(state, problem):
    position, foodGrid = state
    h = 0
    foodlist = foodGrid.asList()  
    if len(foodlist) == 1:
        h = M(position, foodlist[0])
    elif len(foodlist) > 1:
        up = foodlist[0]
        down = foodlist[0]
        left = foodlist[0]
        right = foodlist[0]
        for food in foodlist:
            if food[1] > up[1]:
                up = food
            if food[1] < down[1]:
                down = food
            if food[0] < left[0]:
                left = food
            if food[0] > right[0]:
                right = food
        points = [left, up, right, down]
        edges = [M(up, left), M(right, up), M(down, right), M(left, down)]
        kill_points = get_kill_points(edges, points)
        near_kp = kill_points[0]
        if M(position, kill_points[1]) < M(position, near_kp):
            near_kp = kill_points[1]
        dist_near_killpoint = M(near_kp, position)
        near_kpi = points.index(near_kp)
        sec_p = points[(near_kpi + 1) % 4]
        thi_p = points[(near_kpi + 2) % 4]
        fou_p = points[(near_kpi + 3) % 4]
        if sec_p in kill_points:
            sec_p = points[(near_kpi - 1) % 4]
            thi_p = points[(near_kpi - 2) % 4]
            fou_p = points[(near_kpi - 3) % 4]
        #a1 , fl= getadditional(foodlist, position, near_kp) 
        a2 , fl = getadditional(foodlist, near_kp, sec_p)
        a3, fl = getadditional(fl, sec_p, thi_p) 
        a4, fl = getadditional(fl, thi_p, fou_p)
        h = sum(edges) - max(edges) + dist_near_killpoint +a2+a3+a4
    return h
def among(f1, f2, s, e):
    flag1 = (s[0]-f1[0])*(f1[0]-e[0])
    flag2 = (s[1]-f1[1])*(f1[1]-e[1])
    flag3 = (s[0]-f2[0])*(f2[0]-e[0])
    flag4 = (s[1]-f2[1])*(f2[1]-e[1])
    if flag1 >= 0 and flag2 >= 0 and flag3 >= 0 and flag4 >= 0:
        return True
    return False

def getadditional(foodlist, s, e):
    vectors = []
    fl = foodlist.copy()
    for f1 in foodlist:
        for f2 in foodlist:
            if f1 != f2 and among(f1, f2, s, e):
                vectors.append([(f1[0]-f2[0], f1[1]-f2[1]),f1,f2])
                if f1 in fl:
                    fl.remove(f1)
                if f2 in fl:
                    fl.remove(f2)
    v_rule = (e[0] - s[0], e[1] - s[1])
    flag = v_rule[0]*v_rule[1]
    longest = None
    lon_s = None
    lon_e = None
    if flag > 0:
        for f, fs, fe in vectors:
            if f[0]*f[1]<0:
                if longest==None:
                    longest=f
                    lon_s = fs
                    lon_e = fe
                else:
                    if M((0,0), f)>M((0,0), longest):
                        longest = f 
                        lon_s = fs
                        lon_e = fe
    elif flag < 0:
        for f, fs, fe in vectors:
            if f[0]*f[1] > 0:
                if longest == None:
                    longest = f
                    lon_s = fs
                    lon_e = fe
                else:
                    if M((0, 0), f) > M((0, 0), longest):
                        longest = f
                        lon_s = fs
                        lon_e = fe
    if longest == None:
        return 0, foodlist
    else:
        return 2*min(abs(longest[0]), abs(longest[1])), fl





def get_kill_points(edges, points):
    i = edges.index(max(edges))
    if i == 0:
        return [points[0], points[1]]
    elif i == 1:
        return [points[1], points[2]]
    elif i == 2:
        return [points[2], points[3]]
    return [points[0], points[3]]
    



''' enumeration is not good
def myfoodHeuristic(position, foodlist):
    h = 0
    endpoints = []
    if len(foodlist) > 0:
        xmax = max(foodlist, key=lambda x: x[0])[0]
        xmin = min(foodlist, key=lambda x: x[0])[0]
        ymax = max(foodlist, key=lambda x: x[1])[1]
        ymin = min(foodlist, key=lambda x: x[1])[1]
        h = (xmax - xmin) + (ymax - ymin)
        # go to the nearest edge
        endpoints.append((xmin, ymin))
        endpoints.append((xmax, ymax))
        endpoints.append((xmin, ymax))
        endpoints.append((xmax, ymin))
        import sys
        dist1 = sys.maxsize
        for point in foodlist:  # set(endpoints):
            xy1 = position
            xy2 = point
            dist1 = min(dist1, util.manhattanDistance(xy1, xy2))
        dist2 = sys.maxsize
        for point in set(endpoints):
            xy1 = position
            xy2 = point
            dist2 = min(dist2, util.manhattanDistance(xy1, xy2))
        h += max(dist1, dist2)
    return h

def computeH(position, unvis_food, Distance, iteration, itermax):
    if iteration == itermax: 
        return myfoodHeuristic(position, unvis_food)
    import sys
    h = sys.maxsize
    trans_dist = 0
    for nextfood in unvis_food:
        # compute tranistional distance
        if Distance.__contains__((position, nextfood)):
            trans_dist = Distance[(position, nextfood)]
        else:
            trans_dist = util.manhattanDistance(position, nextfood)
        # update unvisited food
        unvisited_food = unvis_food.copy()
        unvisited_food.remove(nextfood)
        # update h
        h = min(h, computeH(nextfood, unvisited_food, Distance, iteration+1, itermax) + trans_dist)  


ITERATION_MAX = 2
def foodHeuristic(state, problem):
    position, foodGrid = state
    foodlist = foodGrid.asList()
    #distance dictionary initialization
    Distance = {}
    for f1 in foodlist:
        for f2 in foodlist:
            if f1 == f2:
                continue
            d = util.manhattanDistance(f1, f2)
            Distance[(f1,f2)] = d
            Distance[(f2, f1)] = d
    #enumertaion
    if len(foodlist) == 0:
        return 0
    h = computeH(position, foodlist, Distance, 0, ITERATION_MAX)
    return h
'''
class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        problem = AnyFoodSearchProblem(gameState)
        return search.breadthFirstSearch(problem)
        

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state
        return self.food[x][y]
        
