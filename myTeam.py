# myTeam.py

from captureAgents import CaptureAgent
import random
import math
from game import Directions
import time

#####################
# Team Creation
#####################

def createTeam(firstIndex, secondIndex, isRed,
               first='MCTSAgent', second='MCTSAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

#####################
# MCTS Agent Definition
#####################

class MCTSAgent(CaptureAgent):
    def __init__(self, index):
        super().__init__(index)
        self.simulations = 50  # Fewer simulations for faster moves
        self.simulation_depth = 15  # Reduced depth to avoid loops
        self.exploration_factor = 1.4
        self.visited_positions = {}  # Changed to a dict to track frequency
        self.prev_action = None
        self.stuck_count = 0  # Track how many times stuck
        self.last_score = 0  # Track score to detect progress

    #####################
    # Action Selection
    #####################
    def chooseAction(self, gameState):
        current_position = gameState.getAgentPosition(self.index)
        current_score = self.getScore(gameState)
        
        # Check if score changed - reset stuck counter if we're making progress
        if current_score != self.last_score:
            self.stuck_count = 0
            self.last_score = current_score
        
        # Update position frequency counter
        if current_position in self.visited_positions:
            self.visited_positions[current_position] += 1
            # If we've been at this position too many times, we're stuck
            if self.visited_positions[current_position] >= 3:
                self.stuck_count += 1
        else:
            self.visited_positions[current_position] = 1
        
        # Adjust exploration dynamically based on being stuck
        original_exploration = self.exploration_factor
        if self.stuck_count > 2:
            self.exploration_factor = 2.5  # Increase exploration when stuck
            
        # If really stuck, take random action to break out
        if self.stuck_count > 5:
            legal_actions = gameState.getLegalActions(self.index)
            # Filter out reverse direction if possible to avoid back-and-forth
            if len(legal_actions) > 1 and self.prev_action:
                reverse = self.getReverse(self.prev_action)
                if reverse in legal_actions:
                    legal_actions.remove(reverse)
            # Choose random action that's not the reverse
            action = random.choice(legal_actions)
            self.stuck_count = 0  # Reset counter
            self.visited_positions = {}  # Clear history when taking emergency action
            self.prev_action = action
            return action
        
        # Normal MCTS search
        action = self.mctsSearch(gameState)
        
        # Reset exploration factor to original
        self.exploration_factor = original_exploration
        
        # If we're taking the same action repeatedly, increase stuck counter
        if action == self.prev_action:
            self.stuck_count += 1
        else:
            self.stuck_count = max(0, self.stuck_count - 1)
        
        # Remember this action
        self.prev_action = action
        
        # Limit the size of visited_positions dictionary (keep most recent)
        if len(self.visited_positions) > 100:
            # Only keep positions visited frequently
            self.visited_positions = {k: v for k, v in self.visited_positions.items() if v > 1}
        
        return action

    def getReverse(self, action):
        """Returns the reverse of an action"""
        if action == Directions.NORTH:
            return Directions.SOUTH
        elif action == Directions.SOUTH:
            return Directions.NORTH
        elif action == Directions.EAST:
            return Directions.WEST
        elif action == Directions.WEST:
            return Directions.EAST
        return action  # STOP remains STOP

    #####################
    # MCTS Core Algorithm
    #####################
    def mctsSearch(self, gameState):
        root = Node(gameState, None)
        for _ in range(self.simulations):
            node = self.selection(root)
            if not node.is_fully_expanded():
                child = self.expansion(node)
                reward = self.simulation(child.state)
                self.backpropagation(child, reward)
        
        # Choose action based on best child, with a penalty for frequently visited positions
        best_score = float('-inf')
        best_action = None
        
        for child in root.children:
            next_pos = child.state.getAgentPosition(self.index)
            # Apply a penalty for frequently visited positions
            position_penalty = self.visited_positions.get(next_pos, 0) * 0.5
            child_score = child.visits - position_penalty
            
            if child_score > best_score:
                best_score = child_score
                best_action = child.action
        
        # If all actions lead to penalized positions, just use regular selection
        if best_action is None and root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            best_action = best_child.action
        
        return best_action

    #####################
    # MCTS Steps
    #####################

    # Selection: Choose node with highest UCB1
    def selection(self, node):
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1(self.exploration_factor))
        return node

    # Expansion: Add a new child node
    def expansion(self, node):
        legal_actions = node.state.getLegalActions(self.index)
        for action in legal_actions:
            new_state = node.state.generateSuccessor(self.index, action)
            if not any(child.action == action for child in node.children):
                child_node = Node(new_state, node, action)
                node.children.append(child_node)
        return random.choice(node.children)

    # Simulation: Perform a random playout with heuristic bias
    def simulation(self, state):
        depth = 0
        current_state = state
        visited_in_sim = set()  # Track positions visited during simulation
        
        while depth < self.simulation_depth and not current_state.isOver():
            legal_actions = current_state.getLegalActions(self.index)
            my_pos = current_state.getAgentPosition(self.index)
            
            # Add current position to simulation history
            visited_in_sim.add(my_pos)
            
            # If too few actions or we're in a loop, stop simulation
            if len(legal_actions) <= 1 or len(visited_in_sim) > self.simulation_depth + 5:
                break
                
            # Use heuristic bias for action selection
            weights = []
            for action in legal_actions:
                successor = current_state.generateSuccessor(self.index, action)
                new_pos = successor.getAgentPosition(self.index)
                
                # Penalize revisiting positions in this simulation
                if new_pos in visited_in_sim:
                    weights.append(0.1)  # Very small chance to revisit
                    continue
                
                # Get food and enemy positions
                food_list = self.getFood(successor).asList()
                enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
                ghosts = [e for e in enemies if not e.isPacman and e.getPosition() and e.scaredTimer <= 0]
                
                # Calculate weights based on distance to food and ghosts
                if food_list:
                    food_dist = min([self.getMazeDistance(new_pos, food) for food in food_list])
                    food_weight = 1.0 / (food_dist + 1)
                else:
                    food_weight = 0
                
                # Avoid ghosts
                ghost_weight = 0
                for ghost in ghosts:
                    if ghost.getPosition():
                        ghost_dist = self.getMazeDistance(new_pos, ghost.getPosition())
                        if ghost_dist < 3:
                            ghost_weight = -5 / (ghost_dist + 0.1)  # Strong negative weight for nearby ghosts
                
                # Combine weights
                total_weight = food_weight + ghost_weight
                if total_weight <= 0:
                    total_weight = 0.1  # Ensure non-negative weight
                    
                weights.append(total_weight)
            
            # Normalize weights for selection
            if sum(weights) == 0:
                weights = [1] * len(legal_actions)  # Equal weights if all are zero
            
            # Choose action probabilistically based on weights
            chosen_idx = self.weightedRandomChoice(weights)
            best_action = legal_actions[chosen_idx]
            
            current_state = current_state.generateSuccessor(self.index, best_action)
            depth += 1
            
        return self.evaluateState(current_state)
    
    def weightedRandomChoice(self, weights):
        """Choose an index based on weights"""
        total = sum(weights)
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                return i
        return len(weights) - 1  # Fallback
    
    def evaluateState(self, state):
        """Evaluate state more carefully than just score"""
        # Base score from game
        score = self.getScore(state)
        
        # Add bonus for food carried
        my_state = state.getAgentState(self.index)
        carrying = my_state.numCarrying
        score += carrying * 0.5  # Partial credit for food being carried
        
        # Penalize being chased by ghosts
        my_pos = state.getAgentPosition(self.index)
        enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
        ghosts = [e for e in enemies if not e.isPacman and e.getPosition() and e.scaredTimer <= 0]
        
        for ghost in ghosts:
            if ghost.getPosition():
                ghost_dist = self.getMazeDistance(my_pos, ghost.getPosition())
                if ghost_dist < 3:
                    score -= (3 - ghost_dist) * carrying  # Bigger penalty if carrying food
        
        return score

    # Backpropagation: Update the value and visits for each node in the path
    def backpropagation(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

#####################
# MCTS Node Class
#####################

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0

    # Check if the node is fully expanded
    def is_fully_expanded(self):
        return len(self.children) == len(self.state.getLegalActions(0))

    # Calculate the UCB1 score
    def ucb1(self, exploration_factor):
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + exploration_factor * math.sqrt(math.log(self.parent.visits + 1) / (self.visits + 1))