import math
import random


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, exploration_weight=1.0):
        """Select the best child node based on the UCT value."""
        if not self.children:
            raise ValueError("No children to select from.")
        return max(
            self.children,
            key=lambda child: child.value / (child.visits + 1e-6)
            + exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )


class MCTS:
    def __init__(self, exploration_weight=1.0):
        self.exploration_weight = exploration_weight

    def search(self, initial_state, num_simulations):
        root = Node(state=initial_state)

        for _ in range(num_simulations):
            # Selection
            node = self._select(root)
            # Expansion
            if not node.state.is_terminal() and not node.is_fully_expanded():
                node = self._expand(node)
            # Simulation
            reward = self._simulate(node.state)
            # Backpropagation
            self._backpropagate(node, reward)

        return root.best_child(exploration_weight=0)  # Return the best child deterministically

    def _select(self, node):
        """Traverse the tree to select the best node to expand."""
        while not node.state.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node):
        """Expand a node by adding one of its unexplored children."""
        actions = node.state.get_possible_actions()
        for action in actions:
            if all(child.state != node.state.perform_action(action) for child in node.children):
                new_state = node.state.perform_action(action)
                new_node = Node(state=new_state, parent=node)
                node.children.append(new_node)
                return new_node
        raise ValueError("No valid actions to expand.")

    def _simulate(self, state):
        """Simulate a random playout from the given state."""
        current_state = state
        while not current_state.is_terminal():
            action = random.choice(current_state.get_possible_actions())
            current_state = current_state.perform_action(action)
        return current_state.get_reward()

    def _backpropagate(self, node, reward):
        """Propagate the simulation result back through the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent


class GameState:
    """An abstract class representing the state of the game."""
    def get_possible_actions(self):
        raise NotImplementedError

    def perform_action(self, action):
        raise NotImplementedError

    def is_terminal(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError


# Example Usage
# To use this, subclass `GameState` to implement a specific game or decision problem.
# Example for Tic Tac Toe, Chess, or other games.
# Implement the `GameState` methods according to the rules of your specific problem.

if __name__ == "__main__":
    initial_state = YourGameState()  # Replace with your GameState subclass
    mcts = MCTS(exploration_weight=1.4)
    best_node = mcts.search(initial_state, num_simulations=1000)
    print("Best action:", best_node.state)  # Interpret the state to get the best action
