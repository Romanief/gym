import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

env = gym.make("CartPole-v1")

# Set up MatPlotLib
is_python = "inline" in matplotlib.get_backend()
if is_python:
  from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
We will be using experience replay memory for training our DQN. 
It stores the transitions that the agent observes, allowing us to reuse this data later. By sampling from it randomly, 
the transitions that build up a batch are decorrelated. 
It has been shown that this greatly stabilizes and improves the DQN training procedure.
"""

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class ReplayMemory(object):

  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args):
    """ Save a transition """
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class DQN(nn.Module):
  """
  feed forward neural network that takes in the difference between the current and previous screen patches. It has two outputs, 
  representing Q(s, left) and Q(s, right)  (where s is the input to the network). 
  In effect, the network is trying to predict the expected return of taking each action given the current input.
  """
  def __init__(self, n_observation, n_actions):
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observation, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
      x = F.relu(self.layer1(x))
      x = F.relu(self.layer2(x))
      return self.layer3(x)

# Hyperparameters and utilities
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005 # update rate of the target network
LR = 1e-4 # learning rate of the ``AdamW`` optimizer

# Get number of actions and number of observations (2 and 4 for this specific env)
n_actions = env.action_space.n
state, info = env.reset()
n_observation = len(state)

"""
In Deep Q-Learning, the policy network (policy_net) is updated frequently to learn the Q-values for each 
state-action pair based on the agent's experiences. However, using this network directly for updating the 
Q-values can lead to instability because the network is continuously changing.

To mitigate this instability, a target network (target_net) is used. The target network is a copy of the 
policy network but is updated less frequently. This helps to stabilize the learning process by providing 
more consistent Q-value targets. Periodically, the target network is updated with the latest weights from 
the policy network (using the load_state_dict method again).

This approach allows the learning algorithm to benefit from the stability provided by the slowly updated 
target network while still being able to learn and improve the policy network rapidly based on new experiences.
"""

policy_net = DQN(n_observation, n_actions).to(device)
target_net = DQN(n_observation, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(1000)

steps_done = 0

def select_action(state):
  """
  Gets a state as input and returns either a random action or the best possible action given that state
  based on a probability of epsilon to explore and 1-epsilon to exploit. 
  Epsilon is calculated each time using the epsilon decay value previously defined
  """
  global steps_done #  indicates that steps_done is a global variable
  sample = random.random() 
  eps_treshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
  steps_done += 1
  if sample > eps_treshold:
    with torch.no_grad():
      # t.max(1) will return the largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.      
      return policy_net(state).max(1).indices.view(1,1)
  
  else: 
    return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

episode_duration = []

def plot_duration(show_result = False):
  """
  Use MatPlotLib to generate an image showing the result 
  """
  plt.figure(1)
  duration_t = torch.tensor([episode_duration], dtype=torch.float)
  if show_result:
    plt.title("Result")
  else:
    plt.clf()
    plt.title("Training...")
  plt.xlabel("Episode")
  plt.ylabel("Duration")
  plt.plot(duration_t.numpy())
  # Take 100 episode averages and plot them too
  if len(duration_t) >= 100:
    means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99)), means)
    plt.plot(means.numpy())

    plt.pause(0.001)
    if is_python:
      if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
      else:
        display.display(plt.gcf())


def optimize_model():
  """
  performs a single step of the optimization. It first samples a batch, concatenates all the tensors into a single one, computes
  Q(st, at) and V(st + 1, a) = max(Q(st + 1, a)), and combines them into our loss. By definition we set V(s) = 0 if 
  s is a terminal state. We also use a target network to compute V(st + 1) for added stability. 
  The target network is updated at every step with a soft update controlled by the hyperparameter TAU, which was previously defined
  """
  if len(memory) < BATCH_SIZE:
    return

  transitions = memory.sample(BATCH_SIZE)
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
    batch.next_state)), device=device, dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state 
    if s is not None])

  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1).values
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(BATCH_SIZE, device=device)
  with torch.no_grad:
    next_state_values[non_final_mask] = (next_state_values * GAMMA) + reward_batch
  # Compute the expected Q-Value
  expected_state_action_value = (next_state_values * GAMMA) + reward_batch

  # Compute Huber loss function: 
  criteration = nn.SmoothL1Loss()
  loss = criteration(state_action_values, expected_state_action_value.unsqueeze(1))

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  # In-place gradient clipping
  torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
  optimizer.step()


  """
  Below, you can find the main training loop. At the beginning we reset the environment and obtain the initial state Tensor. 
  Then, we sample an action, execute it, observe the next state and the reward (always 1), and optimize our model once. 
  When the episode ends (our model fails), we restart the loop.

  Below, num_episodes is set to 600 if a GPU is available, otherwise 50 episodes are scheduled so training does not take too long. 
  However, 50 episodes is insufficient for to observe good performance on CartPole. You should see the model constantly achieve 
  500 steps within 600 training episodes. Training RL agents can be a noisy process, so restarting training can produce better 
  results if convergence is not observed.
  """

  if torch.cuda.is_available():
    num_episodes = 600
  else:
    num_episodes = 50

  for i_episodes in range(num_episodes):
    # Initialise env end get state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
      action = select_action(state)
      observation, reward, terminated, truncated, _ = env.step(action.item())
      reward = torch.tensor([reward], device=device)
      done = terminated or truncated

      if terminated: 
        next_state = None
      else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

      # store transition in memory
      memory.push(state, action, next_state, reward)

      # move to the next state
      state = next_state

      # perform one step optimization (on policy network)
      optimize_model()

      # Soft update of the target network
      # θ′ ← τ θ + (1 −τ )θ′
      target_next_state_dict = target_net.state_dict()
      policy_net_state_dict = policy_net.state_dict()
      for key in policy_net_state_dict:
        target_next_state_dict[key] = policy_net_state_dict[key]*TAU + target_next_state_dict[key]*(1-TAU)
      target_net.load_state_dict(target_next_state_dict)

      if done:
        episode_duration.append(t + 1)
        plot_duration()
        break


print("Complete")
plot_duration(show_result=True)
plt.ioff()
plt.show()
