import time
import math
import random
import numpy
from reasoning.MCTS.base import treeNode
import warnings

def MCTS_search(mcts_task):
    """
    Performs the Monte Carlo Tree Search algorithm until time/iteration limit is reached or solution is found.
    
    This function implements the main MCTS loop that repeatedly executes selection-expansion-simulation-backpropagation
    rounds until either:
    1. A solution is found (a node with value above the end gate threshold)
    2. Time limit is reached (if limit_type is 'time')
    3. Iteration limit is reached (if limit_type is not 'time')
    
    Args:
        mcts_task: An object containing MCTS parameters, domain functions, and search limits
                   (time_limit or iteration_limit based on limit_type)
    
    Returns:
        tuple: (root, solution_node, search_metric) where:
            - root: The root node of the search tree
            - solution_node: Node representing the solution if found, otherwise None
            - search_metric: Time elapsed (in seconds) if using time limit, 
                            iteration count (starting from 1) if using iteration limit,
                            or None if no solution found
    """
    
    root = treeNode('')
    # If time limit is set, use time limit, otherwise use iteration limit
    if mcts_task.limit_type == 'time':
        # mcts_task.time_limit is in milliseconds
        timeLimit = time.time() + mcts_task.time_limit / 1000
        time_start = time.time()
        while time.time() < timeLimit:
            print(f'<beging new round, current time:{time.time() - time_start}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('found solution!\n')
                return root, node, time.time() - time_start
    else:
        for i in range(mcts_task.iteration_limit):
            print(f'<beging new round, current round:{i}>\n')
            flag, node, root = executeRound(root, mcts_task)
            if flag:
                print('found solution!\n')
                return root, node, i + 1
    return root, None, None

def executeRound(root, mcts_task):
    
    # execute a selection-expansion-simulation-backpropagation round
    # record the time for each phase
    
    time1 = time.time()
    print('-' * 40)
    print('selection phase\n')
    flag, node = selectNode(root, mcts_task)
    print(f'selected node: {node.y}')
    if flag: # If a high-value node is found, return the node and the root
        return True, node, root
    time2 = time.time()
    print('-' * 40)
    print('expansion phase\n')
    if node.isTerminal: # skip this phase if the node is terminal
        print('skip this phase.\n')
    else:
        node = expand(node, mcts_task)
    time3 = time.time()
    print('-' * 40)
    print('simulation phase\n')
    if node.isTerminal or len(node.children) == 0: # skip this phase if the node is terminal or has no children
        print('skip this phase.\n')
    else:
        roll_node = getBestChild(node, mcts_task)
        best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node, mcts_task)
        roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
        roll_node.numVisits += 1
    time4 = time.time()
    print('-' * 40)
    print('backpropagation phase\n')
    back_propagate(node) # Need to update numVisits even for terminal nodes
    time5 = time.time()
    # root.print_tree()
    print(f'time for each phase: selection: {time2 - time1}, expansion: {time3 - time2}, simulation: {time4 - time3}, backpropagation: {time5 - time4}')
    # print the tree
    root.print_tree()
    return False, node, root

def selectNode(node, mcts_task): # True means already found a solution with very high reward
    while len(node.children) > 0: 
        node = getBestChild(node, mcts_task)
    if aboveEndGate(node, mcts_task) and node.isTerminal: # node.isTerminal is just for the development
        node.final_ans_flag = 1
        return True, node
    else:
        return False, node
    
def aboveEndGate(node, mcts_task):
    
    return node.V >= mcts_task.end_gate

def getBestChild(node, mcts_task):
    bestValue = mcts_task.low
    bestNodes = []
    for child in node.children.values():
        nodeValue = child.V + mcts_task.exploration_constant * math.sqrt(
            2 * math.log(node.numVisits) / child.numVisits) if child.numVisits > 0 else child.V + mcts_task.INF
        if nodeValue > bestValue:
            bestValue = nodeValue
            bestNodes = [child]
        elif nodeValue == bestValue:
            bestNodes.append(child)
    return random.choice(bestNodes)

def expand(node: treeNode, mcts_task): # should be careful on how to set isTerminal, expand always choose a leaf node
 
    if node.isTerminal: 
        return node
    else:
        actions = get_next_steps_expand(node, mcts_task) 
        for action in actions:
            if action not in node.children.keys():
                node.append_children(action)
                child = node.children[action]
                value = mcts_task.get_step_value(child.y)
                child.V = value

                child.visit_sequence = mcts_task.node_count # note that this is not the visit times
                mcts_task.update_count()

        return node

def greedyPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1

    for i in range(mcts_task.roll_forward_steps):
        actions = get_next_steps_roll(strs, cur_step, mcts_task)  # str_list
        if not actions:
            break
        new_ys = [strs + action for action in actions]
        cur_step += 1
        values = [mcts_task.get_step_value(new_y) for new_y in new_ys]
        idx = numpy.argmax(values)
        strs = new_ys[idx]
        value = values[idx]
        if value > max_V:
            max_V = value
    return max_V

def randomPolicy(node: treeNode, mcts_task):
    max_V = mcts_task.low
    strs = node.y
    cur_step = node.depth + 1
    if mcts_task.use_reflection == 'common':
        reflection = mcts_task.get_reflection(strs, cur_step)
    else:
        reflection = mcts_task.get_simple_reflection(strs, cur_step)
    node.update_reflection(reflection)
    if reflection == '<end>':
        print('This step has been resolved and does not require simulation.\n')
        return node.V
    for i in range(mcts_task.roll_forward_steps):
        next_steps = get_next_steps_roll(strs, cur_step, mcts_task)
        if not next_steps:
            break
        action = random.choice(next_steps)  # str
        strs = strs + action
        cur_step += 1
        value = mcts_task.get_step_value(strs)
        if value > max_V:
            max_V = value
    return max_V

def back_propagate(node):
    while node is not None:
        node.numVisits += 1

        child_Vs = [child.V * child.numVisits for child in node.children.values()]
        total_num_visits = sum([child.numVisits for child in node.children.values()])
        if total_num_visits > 0:
            node.V = sum(child_Vs) / total_num_visits
        node = node.parent

def MCTS(mcts_task):
    """
    Main function to perform Monte Carlo Tree Search (MCTS).
    
    This function drives the MCTS process by calling MCTS_search to build the search tree,
    then returns the best solution found based on various criteria.
    
    Args:
        mcts_task: An object containing MCTS parameters and domain-specific functions
                   for generating steps, evaluating states, etc.
    
    Returns:
        dict: {'best_node': best_node, 'best_terminal_node': best_terminal_node, 'finish': finish, 'root': root} where:
            - best_node: The node representing the best solution found (may not be terminal)
            - best_terminal_node: The node representing the best solution found with terminal
            - finish: If solution found, the time taken or iteration count (start from 1); otherwise None
            - root: The root node of the search tree
    """
    # Run the search algorithm to build the tree
    # If 'finish' is not None, it is either running time or the iteration index for finding the solution

    root, node, finish = MCTS_search(mcts_task)
    
    # This should be removed
    if mcts_task.sample_value == 'full': 
        print('sampling completed.\n')
        raise NotImplementedError('sampling is not implemented yet')
        return None, -1, root
    else:
        # Case 1: Solution successfully found within limits
        if finish is not None:
            print(f'Solution found!\nSolution:{node.y}\n')
            return node, finish, root
        # Case 2: No solution found within the time/iteration limit
        else:
            # Find the node with the highest value regardless of being terminal
            best_node, best_V = root.getBestV()
            if not best_node.isTerminal:
                print('The highest value solution is not terminal')
            if best_node is not None:
                print(f'The highest value solution is:{best_node.y}')
                print(f'The highest value is:{best_V}\n')
            else:
                warnings.warn('No highest value solution found. This is not expected. Check if the input is legal.')

            # Find the terminal node with the highest value
            best_terminal_node, best_terminal_V = root.getBestTerminalV()
            # Consistency check
            if best_node.y == best_terminal_node.y and best_node.isTerminal:
                raise ValueError('The highest value node is terminal, but the highest value node with terminal is another node')

            if best_terminal_node is not None:
                print(f'The highest value node with terminal:{best_terminal_node.y}\n')
                print(f'The highest value:{best_terminal_V}\n')
            else:
                print('No terminal node found')

    return {'best_node': best_node, 'best_terminal_node': best_terminal_node, 'finish': finish, 'root': root}
            
# think about if adding isFullyexpanded is necessary

def get_next_steps_expand(node: treeNode, mcts_task):
    next_steps = []
    for i in range(mcts_task.branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(node.y, node.depth + 1)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps

def get_next_steps_roll(y: str, step_n: int, mcts_task):
    next_steps = []
    for i in range(mcts_task.roll_branch):
        proposal = ''
        cnt = 3
        while not proposal and cnt:
            proposal = mcts_task.get_next_step(y, step_n)
            cnt -= 1
        if not proposal:
            continue
        next_steps.append(proposal)
    return next_steps