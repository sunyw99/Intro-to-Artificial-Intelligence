import mdp, util

from learningAgents import ValueEstimationAgent
import collections
import sys

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        for _ in range(self.iterations):
            V_ = util.Counter()
            for s in self.mdp.getStates():
                recording = []
                for a in self.mdp.getPossibleActions(s):
                    v = 0
                    for s_, p in self.mdp.getTransitionStatesAndProbs(s, a):
                        r = self.mdp.getReward(s, a, s_)
                        v += p * (r + self.discount * self.values[s_])
                    recording.append(v)
                if len(recording) > 0:
                    V_[s] = max(recording)
                else: V_[s] = self.values[s]
            self.values = V_
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        v = 0
        for s_, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, s_)
            v += p * (r + self.discount * self.values[s_])
        return v

    def computeActionFromValues(self, state):
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)

        best_action = None
        best_value = None
        for a in actions:
            v = 0
            for s_, p in self.mdp.getTransitionStatesAndProbs(state, a):
                r = self.mdp.getReward(state, a, s_)
                v += p * (r + self.discount * self.values[s_])
            if best_action == None or best_value < v:
                best_action = a 
                best_value = v 
        return best_action






    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        states = self.mdp.getStates()
        for i in range(self.iterations):
            s = states[i % len(states)]  
            if self.mdp.isTerminal(s):
                continue 
            recording = []
            for a in self.mdp.getPossibleActions(s):
                v = 0
                for s_, p in self.mdp.getTransitionStatesAndProbs(s, a):
                    r = self.mdp.getReward(s, a, s_)
                    v += p * (r + self.discount * self.values[s_])
                recording.append(v)
            self.values[s] = max(recording)
                

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        heap = util.PriorityQueue()

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue
            q_list = []
            for a in self.mdp.getPossibleActions(s):
                q_list.append(self.computeQValueFromValues(s,a))              
            diff = abs(max(q_list) - self.values[s])
            heap.update(s, -diff)

        for _ in range(self.iterations):
            if heap.isEmpty():
                return
            s = heap.pop()
            if self.mdp.isTerminal(s):
                continue
            recording=[]
            for a in self.mdp.getPossibleActions(s):
                v = 0
                for s_, p in self.mdp.getTransitionStatesAndProbs(s, a):
                    r = self.mdp.getReward(s, a, s_)
                    v += p * (r + self.discount * self.values[s_])
                recording.append(v)
            self.values[s] = max(recording)
            for p in self.computePred(s):
                if self.mdp.isTerminal(p):
                    continue
                q_list = []
                for a in self.mdp.getPossibleActions(p):
                    q_list.append(self.computeQValueFromValues(p, a))
                diff = abs(max(q_list) - self.values[p])
                if diff > self.theta:
                    heap.update(p, -diff)




    def computePred(self, state):
        predecessors = []
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                for s_, _ in self.mdp.getTransitionStatesAndProbs(s,a):
                    if s_ == state:
                        predecessors.append(s)
        return list(set(predecessors))
