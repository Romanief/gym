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

env = gym.Env("CartPole-v1")

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


class DQN(nn.modules):
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