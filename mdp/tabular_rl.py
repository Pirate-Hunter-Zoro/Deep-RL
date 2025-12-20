import random
from mdp.mdp import MDP, MDPState

class QLearningAgent:
    
    def __init__(self, mdp: MDP, discount_factor: float=0.9, epsilon: float=0.1):
        self.mdp = mdp
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {} # (state, action) -> value
        self.n_table = {} # (state, action) -> count
        
    def get_q_value(self, state: MDPState, action: int) -> float:
        """Return the value associated with this state/action pair

        Args:
            state (MDPState): State
            action (int): Respective action to take at said state

        Returns:
            float: Resulting q-value
        """
        return self.q_table.get((state.i, action), 0.0)
    
    def choose_action(self, state: MDPState, use_epsilon: bool=True) -> int:
        """Method to choose an action at the given state

        Args:
            state (MDPState): State to make decision at

        Returns:
            int: Resulting action
        """
        legal_actions = self.mdp.actions_at(state=state)
        if random.random() < self.epsilon and use_epsilon:
            # Random action
            return legal_actions[int(random.random()*len(legal_actions))]
        else:
            q_values = [self.get_q_value(state=state, action=a) for a in legal_actions]
            m = max(q_values)
            best_actions = [a for a in legal_actions if self.get_q_value(state=state, action=a)==m]
            # Randomly pick from all the best actions
            return best_actions[int(random.random()*len(best_actions))]
        
    def update(self, state: MDPState, action: int, reward: float, next_state: MDPState):
        """Helper method to update agent's q-table given the state, action, reward, and next state achieved

        Args:
            state (MDPState): State at which action was taken
            action (int): Action taken
            reward (float): Resulting reward
            next_state (MDPState): Resulting next state
        """
        key = (state.i, action)
        if key not in self.n_table.keys():
            self.n_table[key] = 0
        self.n_table[key] += 1

        # Learning rate -> 1/count
        alpha = 1/self.n_table[key]
        
        # Is the next state terminal?
        q_new = reward
        if not self.mdp.is_terminal(state=next_state):
            # We add discounted future rewards - the discounted max q-value of whatever can be achieved from the next state
            q_new += self.discount_factor * max([self.get_q_value(state=next_state, action=a) for a in self.mdp.actions_at(state=next_state)])
        # Update the q-value with the new value multiplied by the learning rate
        q_old = self.get_q_value(state=state, action=action)
        self.q_table[key] = q_old + alpha * (q_new - q_old)
     
        
class SARSAAgent(QLearningAgent):
    
    def update(self, state: MDPState, action: int, reward: float, next_state: MDPState, next_action: int):
        """Helper method to update agent's q-table given the state, action, reward, next state achieved, AND next action taken (since this is SARSA)

        Args:
            state (MDPState): State at which action was taken
            action (int): Action taken
            reward (float): Resulting reward
            next_state (MDPState): Resulting next state
            next_action (int): Action taken at next state according to policy
        """
        key = (state.i, action)
        if key not in self.n_table.keys():
            self.n_table[key] = 0
        self.n_table[key] += 1

        # Learning rate -> 1/count
        alpha = 1/self.n_table[key]
        
        # Is the next state terminal?
        q_new = reward
        if not self.mdp.is_terminal(state=next_state):
            # We add discounted future rewards - the discounted max q-value of whatever can be achieved from the next state
            q_new += self.discount_factor * self.get_q_value(state=next_state, action=next_action)
        # Update the q-value with the new value multiplied by the learning rate
        q_old = self.get_q_value(state=state, action=action)
        self.q_table[key] = q_old + alpha * (q_new - q_old)