import random

class GridEnvironment:
    # Create our intial environment with cells initialized to -1
    def __init__(self, width, height, default_reward=0, varying_r_value=0, terminal_state_reward=1):
        
        # Environment dimensions
        self.width = width
        self.height = height

        # Set terminal state
        self.terminal = (0, self.width - 1)

        # Rewards and states
        self.rewards = [[default_reward for cell in range(self.width)] for row in range(self.height)]

        # Utility Grid, initialize all to 0
        self.utility_grid = [[0 for cell in range(width)] for row in range(height)]

        # Define set of actions as keys in a dictionary, which point to a tuple representing the change they represent
        self.actions = {
            '↑': (-1, 0),
            '↓': (1, 0),
            '←': (0, -1),
            '→': (0, 1)
        }

        # Get the right angle alternative actions for when the intended action failsr
        self.alternatives = {
            '↑': ['←', '→'],
            '↓': ['←', '→'],
            '←': ['↑', '↓'],
            '→': ['↑', '↓']
        }

        # Policy grid, intialize all to a random action
        self.policy_grid = [[random.choice(list(self.actions.keys())) for cell in range(self.width)] for row in range(self.height)]
        
        # Mark terminal state as no actions should be taken
        self.policy_grid[0][self.width - 1] = '*'

        # Set the terminal state reward in the top right corner
        self.rewards[0][self.width-1] = terminal_state_reward

        # Set the varying r value using the passed argument
        self.rewards[0][0] = varying_r_value

        # Discount factor (gamma)
        self.gamma = 0.50

    def is_terminal(self, state):
        
        # Only one terminal state, compare the current state to it
        return state == self.terminal

    '''
        This method takes a current state and action to determine what s' will be
        if the chosen action is actually performed with 100% certainty.
    '''
    def calculate_s_prime(self, current_state, action):

        # Get delta for action
        action_delta = self.actions[action]

        # Calculate s'
        new_state_row = current_state[0] + action_delta[0]
        new_state_column = current_state[1] + action_delta[1]

        # Make sure s' is a valid state, if so return it. Otherwise return the original s.
        if new_state_row >= 0 and new_state_row < self.height and new_state_column >= 0 and new_state_column < self.width:

            new_state = (new_state_row, new_state_column)

            # Return the newly calculated s' as a result of the action
            return new_state
        else:

            # Intended action would take agent outside of the grid, therefore agent stays in current state
            return current_state
            
    def visualize_reward_grid(self):
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.rewards]))

    def visualize_utility_grid(self):
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.utility_grid]))

    def visualize_policy_grid(self):
        print('\n'.join([' '.join([str(cell) for cell in row]) for row in self.policy_grid]))

    '''
        Uses the current state and action to get neighboring s primes
        and their probabilies as a list of tuples.

        There will be three s' primes stored for each of the three actions. The intended action, and 
        the two alternative actions. It is possible at times that two different actions will lead
        to the same s' with different probabilties. However, these will simply be aggregated inside the
        calculate q value function.
    '''
    def get_s_primes_and_probabilities(self, state, action):

        # We return a list of tuples to represent each s prime and it's calculated probability
        s_primes_and_probabilities = []

        # Determine s' prime of the main action
        intended_action_s_prime = self.calculate_s_prime(state, action)

        # Append intended s prime and its probability to list
        s_primes_and_probabilities.append((intended_action_s_prime, 0.8))

        # Calculate alternative s primes and their probabilities
        for alternative_action in self.alternatives[action]:

            # Alt action s prime
            alternative_action_s_prime = self.calculate_s_prime(state, alternative_action)

            # Append alt action s prime along with its probability of 10%
            s_primes_and_probabilities.append((alternative_action_s_prime, 0.10))

        # Return the neighboring s primes and their probabilities as a result of those chosen action
        return s_primes_and_probabilities

    '''
        Calculate the q value of a state action pair. 

        The value iteration function uses the max of these to determine the utility of being in a state
        for an iteration.
    '''
    def calculate_q_value(self, state, action):

        # Check if the state is terminal
        if self.is_terminal(state):
            
            # Return the terminal state reward as the Q value
            return self.rewards[state[0]][state[1]]

        # Calculate the Q value of a state action pair
        # Sum over the possible s' probabilities multiplied by (the reward + plus gamma times stored q value of s') for that s'
        # given the passed action and state.
        current_q_value = 0

        for s_prime, probability in self.get_s_primes_and_probabilities(state, action):
            
            # Reward for traveling to s prime from current state by taking action s
            reward_to_s_prime = self.rewards[s_prime[0]][s_prime[1]] 

            # Gamma times times the utility grid location of the next state is the discounted q value of s'prime calculated so far
            current_q_value += probability * (reward_to_s_prime + (self.gamma * self.utility_grid[s_prime[0]][s_prime[1]]))

        return current_q_value


    '''
        Value iteration function to find the optimal utilities
        This can be used to find the optimal policy for each path. 
    '''
    def value_iteration(self, epsilon=0.001):
        
        # Keep going until we break out when stopping criteria is met
        while True:
            
            # Set / Reset delta to 0 for this iteration
            delta = 0

            # Create a temporary copy of the utility grid at the start of the iteration, used for calculating U' while we still have access to U
            utility_grid_prime = [row[:] for row in self.utility_grid]

            # Go through each state in this iteration
            for row in range(self.height):
                for column in range(self.width): 
                    if self.is_terminal((row, column)):
                        utility_grid_prime[row][column] = self.rewards[row][column]
                        continue

                    # Get original state utility to calculate delta later
                    old_state_utility = self.utility_grid[row][column]
                
                    # Start with currently stored action as the best by default
                    best_action = None

                    # Keep track of the max q found this iteration
                    max_q_value = float('-inf')

                    # Find action which provides the maximum Q value (utility) for this state
                    for action in self.actions:

                        new_q_value = self.calculate_q_value((row, column), action)

                        # New utility of this state is the max of the previous utility stored and the new q value with the new action that was calculated
                        if new_q_value > max_q_value:
                            max_q_value = new_q_value
                            best_action = action

                    # Update the environment with the new found best q value and action
                    if best_action is not None:
                        utility_grid_prime[row][column] = max_q_value
                        self.policy_grid[row][column] = best_action

                    # Get the utility change between the previous utility value for this state and the new one
                    utility_change = abs(max_q_value - old_state_utility)
            
                    # If the utility change so far is the largest, update delta (largest change this iteration)
                    if utility_change > delta:
                        delta = utility_change

            # Update utility grid after processing all states
            self.utility_grid = utility_grid_prime

            # Max delta is less than the stopping criteria dervied from epsilon, utilties have converged to or close to optimal
            if delta <= epsilon * (1 - self.gamma) / self.gamma:
                break

        # Utility Grid and Optimal Policy have been filled out
        return 

def main():
        
    # Set MDP dimensions here
    grid_width = 3
    grid_height = 3

    # Set the default reward for each state
    initial_reward_value = -1

    # Set the terminal state reward
    terminal_state_reward = 10

    # Create a different MDP environment for each of the different varying r values
    # We show the initial environments, utilities, and policies for each
    # Afterwards we show the max utilities and optimal policies after performing value iteration on each
    
    counter = 1
    
    r_values = [-100, -3, 0, 3]

    for r_value in r_values:

        print(f"MDP #{counter}")

        counter += 1

        print(f"Default R Value Per State: {initial_reward_value}")
        print(f"Varying R Value in Top Left Corner: {r_value}")
        print(f"Terminal State Reward: {terminal_state_reward}")
        
        MDP_environment = GridEnvironment(grid_width, grid_height, initial_reward_value, r_value, terminal_state_reward)

        print("Reward Grid for Traveling to Each Cell")
        MDP_environment.visualize_reward_grid()

        # Show initial grids
        print("Environment Before Value Iteration")

        print("Utility of Each Cell")
        MDP_environment.visualize_utility_grid()

        print("Initial Random Policy")
        MDP_environment.visualize_policy_grid()

        # Perform value Iteration
        MDP_environment.value_iteration(0.001)

        print("Environment After Value Iteration")

        print("Max Utility of Each Cell Given R")
        MDP_environment.visualize_utility_grid()

        print("Optimal Policy Given R")
        MDP_environment.visualize_policy_grid()

        print("----------------------------------------")

if __name__ == "__main__":
    main()
