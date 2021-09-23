# Hint: from collections import deque
from Interface import *


# = = = = = = = QUESTION 1  = = = = = = = #


def consistent(assignment, csp, var, value):
    """
    Checks if a value assigned to a variable is consistent with all binary constraints in a problem.
    Do not assign value to var.
    Only check if this value would be consistent or not.
    If the other variable for a constraint is not assigned,
    then the new value is consistent with the constraint.

    Args:
        assignment (Assignment): the partial assignment
        csp (ConstraintSatisfactionProblem): the problem definition
        var (string): the variable that would be assigned
        value (value): the value that would be assigned to the variable
    Returns:
        boolean
        True if the value would be consistent with all currently assigned values, False otherwise
    """
    # TODO: Question 1
    for constraint in csp.binaryConstraints:
        if not constraint.affects(var):
            continue
        var1 = None
        var2 = None
        if var == constraint.var1:
            var1 = var
            var2 = constraint.otherVariable(var1)
            if not assignment.isAssigned(var2):
                continue
            value1 = value
            value2 = assignment.assignedValues[var2]
            if not constraint.isSatisfied(value1, value2):
                return False
        else:
            var2 = var
            var1 = constraint.otherVariable(var2)
            if not assignment.isAssigned(var1):
                continue
            value1 = assignment.assignedValues[var1]
            value2 = value
            if not constraint.isSatisfied(value1, value2):
                return False
    return True


def recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod):
    """
    Recursive backtracking algorithm.
    A new assignment should not be created.
    The assignment passed in should have its domains updated with inferences.
    In the case that a recursive call returns failure or a variable assignment is incorrect,
    the inferences made along the way should be reversed.
    See maintainArcConsistency and forwardChecking for the format of inferences.

    Examples of the functions to be passed in:
    orderValuesMethod: orderValues, leastConstrainingValuesHeuristic
    selectVariableMethod: chooseFirstVariable, minimumRemainingValuesHeuristic
    inferenceMethod: noInferences, maintainArcConsistency, forwardChecking

    Args:
        assignment (Assignment): a partial assignment to expand upon
        csp (ConstraintSatisfactionProblem): the problem definition
        orderValuesMethod (function<assignment, csp, variable> returns list<value>):
            a function to decide the next value to try
        selectVariableMethod (function<assignment, csp> returns variable):
            a function to decide which variable to assign next
        inferenceMethod (function<assignment, csp, variable, value> returns set<variable, value>):
            a function to specify what type of inferences to use
    Returns:
        Assignment
        A completed and consistent assignment. None if no solution exists.
    """
    # TODO: Question 1
    #print(assignment.assignedValues)
    #inferenceMethod = forwardChecking
    #inferenceMethod = maintainArcConsistency
    if assignment.isComplete():
        return assignment
    var = selectVariableMethod(assignment, csp)
    for val in orderValuesMethod(assignment, csp, var):
        #inferences = None
        if consistent(assignment, csp, var, val):
            assignment.assignedValues[var] = val
            oldDomain = assignment.varDomains[var]
            assignment.varDomains[var] = {val}
            inferences = inferenceMethod(assignment, csp, var, val)
            if not inferences == None:
                result = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
                if not result == None:
                    return result
            assignment.assignedValues[var] = None
            assignment.varDomains[var] = oldDomain
            if not inferences == None:
                for i_var, i_val in inferences:
                    #print(assignment.varDomains[i_var], 'before', i_val)
                    assignment.varDomains[i_var].add(i_val)
                    #print(assignment.varDomains[i_var], 'after', i_val)
        
    return None


def eliminateUnaryConstraints(assignment, csp):
    """
    Uses unary constraints to eleminate values from an assignment.

    Args:
        assignment (Assignment): a partial assignment to expand upon
        csp (ConstraintSatisfactionProblem): the problem definition
    Returns:
        Assignment
        An assignment with domains restricted by unary constraints. None if no solution exists.
    """
    domains = assignment.varDomains
    for var in domains:
        for constraint in (c for c in csp.unaryConstraints if c.affects(var)):
            for value in (v for v in list(domains[var]) if not constraint.isSatisfied(v)):
                domains[var].remove(value)
                # Failure due to invalid assignment
                if len(domains[var]) == 0:
                    return None
    return assignment


def chooseFirstVariable(assignment, csp):
    """
    Trivial method for choosing the next variable to assign.
    Uses no heuristics.
    """
    for var in csp.varDomains:
        if not assignment.isAssigned(var):
            return var


# = = = = = = = QUESTION 2  = = = = = = = #


def minimumRemainingValuesHeuristic(assignment, csp):
    """
    Selects the next variable to try to give a value to in an assignment.
    Uses minimum remaining values heuristic to pick a variable. Use degree heuristic for breaking ties.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
    Returns:
        the next variable to assign
    """
    nextVar = None
    domains = assignment.varDomains

    # TODO: Question 2
    remaining_cnt = None
    degree = None
    for var, dom in domains.items():
        if assignment.isAssigned(var):
            continue
        r = len(dom)
        d = 0
        for constraint in csp.binaryConstraints:
            if not constraint.affects(var):
                continue
            v = constraint.otherVariable(var)
            if not assignment.isAssigned(v):
                d += 1

        if nextVar == None or r < remaining_cnt or (r == remaining_cnt and d > degree):
            nextVar = var
            remaining_cnt = r
            degree = d

    return nextVar


def orderValues(assignment, csp, var):
    """
    Trivial method for ordering values to assign.
    Uses no heuristics.
    """
    return list(assignment.varDomains[var])


# = = = = = = = QUESTION 3  = = = = = = = #


def leastConstrainingValuesHeuristic(assignment, csp, var):
    """
    Creates an ordered list of the remaining values left for a given variable.
    Values should be attempted in the order returned.
    The least constraining value should be at the front of the list.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable to be assigned the values
    Returns:
        list<values>
        a list of the possible values ordered by the least constraining value heuristic
    """
    # TODO: Question 3
    cntDict = {}
    for val in assignment.varDomains[var]:
        cnt = 0
        for constraint in csp.binaryConstraints:
            if not constraint.affects(var):
                continue
            other_var = constraint.otherVariable(var)
            for v in assignment.varDomains[other_var]:
                if (var == constraint.var1 and not constraint.isSatisfied(val, v)) or\
                    (var == constraint.var2 and not constraint.isSatisfied(v, val)):
                    cnt += 1
        cntDict[val] = cnt
    L = list(cntDict.items())
    L.sort(key = lambda x: x[1], reverse = False)
    l = [con for con, cnt in L]
    return l
    

def noInferences(assignment, csp, var, value):
    """
    Trivial method for making no inferences.
    """
    return set([])


# = = = = = = = QUESTION 4  = = = = = = = #


def forwardChecking(assignment, csp, var, value):
    """
    Implements the forward checking algorithm.
    Each inference should take the form of (variable, value)
    where the value is being removed from the domain of variable.
    This format is important so that the inferences can be reversed
    if they result in a conflicting partial assignment.
    If the algorithm reveals an inconsistency,
    any inferences made should be reversed before ending the function.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable that has just been assigned a value
        value (string): the value that has just been assigned
    Returns:
        set< tuple<variable, value> >
        the inferences made in this call or None if inconsistent assignment
    """
    inferences = set([])

    # TODO: Question 4
    if not consistent(assignment, csp, var, value):
        return None
    for constraint in csp.binaryConstraints:
        if not constraint.affects(var):
            continue
        other_var = constraint.otherVariable(var)
        if assignment.isAssigned(other_var):
            continue
        count = 0
        for val in assignment.varDomains[other_var]:
            if var == constraint.var1 and not constraint.isSatisfied(value, val):
                inferences.add((other_var, val))
                count = count + 1
            elif var == constraint.var2 and not constraint.isSatisfied(val, value):
                inferences.add((other_var, val))
                count = count + 1
            if count == len(assignment.varDomains[other_var]):
                return None
                
    for var, val in inferences:
        assignment.varDomains[var].remove(val)
    
    return inferences


# = = = = = = = QUESTION 5  = = = = = = = #


def revise(assignment, csp, var1, var2, constraint):
    """
    Helper function to maintainArcConsistency and AC3.
    Remove values from var2 domain if constraint cannot be satisfied.
    Each inference should take the form of (variable, value)
    where the value is being removed from the domain of variable.
    This format is important so that the inferences can be reversed
    if they result in a conflicting partial assignment.
    If the algorithm reveals an inconsistency,
    any inferences made should be reversed before ending the function.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var1 (string): the variable with consistent values
        var2 (string): the variable that should have inconsistent values removed
        constraint (BinaryConstraint): the constraint connecting var1 and var2
    Returns:
        set<tuple<variable, value>>
        the inferences made in this call or None if inconsistent assignment
    """
    inferences = set([])

    # TODO: Question 5
    for val2 in assignment.varDomains[var2]:
        flag_remove = True
        for val1 in assignment.varDomains[var1]:
            if var1 == constraint.var1 and constraint.isSatisfied(val1, val2):
                flag_remove = False
            elif var1 == constraint.var2 and constraint.isSatisfied(val2, val1):
                flag_remove = False
        if flag_remove:
            inferences.add((var2, val2))
    if len(inferences) == len(assignment.varDomains[var2]):
        return None
    return inferences

class Queue:
    def __init__(self):
        self.list = []

    def push(self,item):
        if not item in self.list:
            self.list.insert(0, item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0

def maintainArcConsistency(assignment, csp, var, value):
    """
    Implements the maintaining arc consistency algorithm.
    Inferences take the form of (variable, value)
    where the value is being removed from the domain of variable.
    This format is important so that the inferences can be reversed
    if they result in a conflicting partial assignment.
    If the algorithm reveals an inconsistency,
    and inferences made should be reversed before ending the function.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
        var (string): the variable that has just been assigned a value
        value (string): the value that has just been assigned
    Returns:
        set<<variable, value>>
        the inferences made in this call or None if inconsistent assignment
    """
    inferences = set([])
    domains = assignment.varDomains

    # TODO: Question 5
    #  Hint: implement revise first and use it as a helper function"""
    queue = Queue()
    for constraint in csp.binaryConstraints:
        if constraint.affects(var):
            if not assignment.isAssigned(constraint.otherVariable(var)):
                queue.push((var, constraint.otherVariable(var)))
    while not queue.isEmpty():
        var1, var2 = queue.pop()
        for constraint in csp.binaryConstraints:
            if constraint.affects(var1) and constraint.affects(var2):
                inferences_revise = revise(assignment, csp, var1, var2, constraint)
                if inferences_revise is None:
                    for var, val in inferences:
                        assignment.varDomains[var].add(val)
                    return None
                elif inferences_revise:
                    for i_constraint in csp.binaryConstraints:
                        if i_constraint.affects(var2):
                            queue.push((var2, i_constraint.otherVariable(var2)))
                    for inference_revise in inferences_revise:
                        inferences.add(inference_revise)
                        assignment.varDomains[inference_revise[0]].remove(inference_revise[1])
    return inferences
    


# = = = = = = = QUESTION 6  = = = = = = = #


def AC3(assignment, csp):
    """
    AC3 algorithm for constraint propagation.
    Used as a pre-processing step to reduce the problem
    before running recursive backtracking.

    Args:
        assignment (Assignment): the partial assignment to expand
        csp (ConstraintSatisfactionProblem): the problem description
    Returns:
        Assignment
        the updated assignment after inferences are made or None if an inconsistent assignment
    """
    inferences = set([])

    # TODO: Question 6
    #  Hint: implement revise first and use it as a helper function"""
    queue = Queue()
    for var in csp.varDomains:
        for constraint in csp.binaryConstraints:
            if constraint.affects(var):
                queue.push((var, constraint.otherVariable(var)))
    while not queue.isEmpty():
        var1, var2 = queue.pop()
        for constraint in csp.binaryConstraints:
            if constraint.affects(var1) and constraint.affects(var2):
                inferences_revise = revise(assignment, csp, var1, var2, constraint)
                if inferences_revise is None:
                    for var, val in inferences:
                        assignment.varDomains[var].add(val)
                    return None
                elif inferences_revise:
                    for i_constraint in csp.binaryConstraints:
                        if i_constraint.affects(var2):
                            queue.push((var2, i_constraint.otherVariable(var2)))
                    for inference_revise in inferences_revise:
                        inferences.add(inference_revise)
                        assignment.varDomains[inference_revise[0]].remove(inference_revise[1])
    return assignment
    

def solve(csp, orderValuesMethod=leastConstrainingValuesHeuristic,
          selectVariableMethod=minimumRemainingValuesHeuristic,
          inferenceMethod=forwardChecking, useAC3=True):
    """
    Solves a binary constraint satisfaction problem.

    Args:
        csp (ConstraintSatisfactionProblem): a CSP to be solved
        orderValuesMethod (function): a function to decide the next value to try
        selectVariableMethod (function): a function to decide which variable to assign next
        inferenceMethod (function): a function to specify what type of inferences to use
        useAC3 (boolean): specifies whether to use the AC3 pre-processing step or not
    Returns:
        dictionary<string, value>
        A map from variables to their assigned values. None if no solution exists.
    """
    assignment = Assignment(csp)

    assignment = eliminateUnaryConstraints(assignment, csp)
    if assignment is None:
        return assignment

    if useAC3:
        assignment = AC3(assignment, csp)
        if assignment is None:
            return assignment

    assignment = recursiveBacktracking(assignment, csp, orderValuesMethod, selectVariableMethod, inferenceMethod)
    if assignment is None:
        return assignment

    return assignment.extractSolution()
