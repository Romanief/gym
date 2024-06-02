import gymnasium as gym
import numpy as np
import random

# Hyperparamteters
alpha = 0.05
gamma = 0.90
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.1
episodes = 20000
max_steps = 200

# Initialise environment
env = gym.make("CartPole-v1")
state_space = [20, 20, 50, 50] #cart_position, cart_velocity, pole_angle, pole_angular_velocity
q_table = np.zeros(state_space + [env.action_space.n])

def discretize_state(state):
    """
    Discretize a space means to convert all continuous actions into discrete and finite actions. 
    Continuous action can be infinite or very large and will therefore be difficult to handle. 
    This function takes a state representing all the values for each dimension [cart position, cart velocity, pole angle, pole velocity]
    and returns a discretised tuple rounded to the closest integer. 
    """

    # Normalization formula = (state - min) / (max - min). Returns a value between 0 - 1
    normalised_state = (state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)

    # Scales the normalised values into the number and size of bins, then it rounds each direction into their closest integer value.
    discretized = np.round(normalised_state * (np.array(state_space) - 1)).astype(int)

    return tuple(discretized)

# Q-learning loop algorithm
print("Training started:\n-----------------------------------\n")
for episode in range(episodes):
    state = discretize_state(env.reset()[0])
    total_reward = 0

    for step in range(max_steps):
        """
        Decide wether to explore or exploit based on epsilon. with probability epsilon the 
        algorithm will explore by taking a random possible action. With probability 1 - epsilon
        the algorithm will take the best possible action based on the q-value of the previously
        explored actions. 
        As epsilon starts with a value of 1, the first action will always be random. 
        """
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state)
        total_reward += reward

        # Q-Learning algorithm:  Q(s, a) <- Q(s, a) + alpha[R + gamma * max(Q(s1, a1)) - Q(s, a))]
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action] # Temporal Difference Target -> sum of total reward and the discounted q value of best action for next state
        td_error = td_target - q_table[state][action] # Temporal Difference Error -> Difference between TDTarget and the current q-value
        q_table[state][action] = q_table[state][action] + alpha * td_error # Update current q-value based on the larning rate (alpha)

        state = next_state

        if done:
            break
        
    # Reduce epsilon by epsilon decay rate to gradually reduce exploration and favour learning on previous experiences
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Epsiode {episode + 1}: Total reward: {total_reward}")
print("Training finished.")

# Test trained agend
print("Testing:\n----------------------------------\n")
for episode in range(10):
    state = discretize_state(env.reset()[0])
    total_reward = 0

    for step in range(max_steps):
        action = np.argmax(q_table[state])
        next_state, reward, done, _, _ = env.step(action)
        state = discretize_state(next_state)
        total_reward += reward

        if done:
            print(f"Episode {episode + 1} - Total reward: {total_reward}")
            break

env.close()
