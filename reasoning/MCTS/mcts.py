import time
import math
import random
import numpy
from functools import partial
import copy
from reasoning.MCTS.base import treeNode


def MCTS_search(mcts_task):
    root = treeNode('')

    if mcts_task.limit_type == 'time':
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

    print('-' * 40)
    print('selection phase\n')
    flag, node = selectNode(root, mcts_task)
    print(f'selected node: {node.y}')
    if flag:
        return True, node, root

    print('-' * 40)
    print('expansion phase\n')
    if node.isTerminal:
        print('skip this phase.\n')
    else:
        node = expand(node, mcts_task)

    print('-' * 40)
    print('simulation phase\n')
    roll_node = getBestChild(node, mcts_task)
    best_V = greedyPolicy(roll_node, mcts_task) if mcts_task.roll_policy == 'greedy' else randomPolicy(roll_node,
                                                                                                        mcts_task)
    roll_node.V = roll_node.V * (1 - mcts_task.alpha) + best_V * mcts_task.alpha
    roll_node.numVisits += 1

    print('-' * 40)
    print('backpropagation phase\n')
    back_propagate(node)
    
    root.print_tree()
    
    
    
    return False, node, root

def selectNode(node, mcts_task): # True means already found a solution with very high reward
    while len(node.children) > 0: 
        node = getBestChild(node, mcts_task)
    if aboveEndGate(node, mcts_task):
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

def expand(node: treeNode, mcts_task): # should be careful on how to set isTerminal
 
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
        # if mcts_task.use_reflection == 'common':
        #     cur_ref = mcts_task.get_reflection(strs, cur_step)
        # else:
        #     cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        # if cur_ref == '<end>':
        #     break
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
        # if mcts_task.use_reflection == 'common':
        #     cur_ref = mcts_task.get_reflection(strs, cur_step)
        # else:
        #     cur_ref = mcts_task.get_simple_reflection(strs, cur_step)
        # if cur_ref == '<end>':
        #     break
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
    root, node, finish = MCTS_search(mcts_task)

    if mcts_task.sample_value == 'full':
        print('采样完成。\n')
        return None, -1, root
    else:
        if finish is not None:
            print(f'已找到最终解!\nSolution:{node.y}\n')
            return node, finish, root

        else:
            best_node, best_V = root.getBestV()
            print(f'在规定时间/轮次内未找到满足要求价值的解答，采用最高价值价值解答代替。\nSolution:{best_node.y}\n')
            best_terminal_node, best_terminal_V = root.getBestTerminalV()
            print(f'最高价值解答:{best_terminal_node.y}\n') if best_terminal_node is not None else print('没有终端节点')
            return best_node, -1, root

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