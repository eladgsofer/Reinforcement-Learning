

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
import numpy as np

device = torch.device("mps" if not torch.backends.mps.is_available() else "cpu")

def torch_zeros(*args, **kwargs):
    kwargs.pop('device', None)
    return torch.zeros(*args, **kwargs, device=device)

torch_zeros = torch_zeros

env = gym.make("CartPole-v1")

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def temporal_difference(Qnet_target,Qnet,S,S_1,a,R,done,gamma): # batched temporal difference function
    '''
        The temporal difference is our estimation to the model's error. This is a dynamic term which can be calculated in the
        middle of an episode. The function is written to work efficiently with a batch of inputs.
    '''
    reference = torch_zeros(done.shape,device=device)
    # if done then the reference signal is only the reward
    reference[done] = R[done]
    # if not done then the reference signal is the reward plus the estimated accumulated reward for state S+1 and the best action
    with torch.no_grad():
        best_next_acc_reward = torch.max(Qnet_target(S_1),dim=1)[0].unsqueeze(1) # the reference is with respect to the target Q
    reference[~done] = R[~done]+gamma*best_next_acc_reward.squeeze(1)[~done]

    a_reshaped = a.type(torch.int64).view(-1, 1)
    # the estimation uses the changing Qnet
    estimation = Qnet(S).gather(1, a_reshaped) # gather the estimated Q value chosen by action(not necessarly the best action)
    return estimation,reference
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class QNet(nn.Module):
    '''
        The deep learning model
    '''
    def __init__(self,input_states_size,output_actions_size, hidden_layers_size=None):
        super(QNet, self).__init__()
        all_layers_sizes = np.zeros(len(hidden_layers_size)+2,dtype=np.uint)
        layers = []
        all_layers_sizes[0] = input_states_size
        all_layers_sizes[-1] = output_actions_size
        all_layers_sizes[1:-1] = hidden_layers_size
        for i in range(len(all_layers_sizes)-1):
            layers.append(nn.Linear(all_layers_sizes[i],all_layers_sizes[i+1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4



# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

layers = [128,128,128]
policy_net = QNet(n_observations, n_actions, layers).to(device)
target_net = QNet(n_observations, n_actions, layers).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
epsilon = 1

def epsilon_greedy_action(state, epsilon):
    global steps_done
    sample = np.random.uniform()
    steps_done += 1

    if sample > epsilon:
        # with torch.no_grad():
        return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch_zeros(99).to('cpu'), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    estimation,reference = temporal_difference(target_net,
                        policy_net,
                        torch.cat(batch.state),
                        torch.cat([torch.tensor([[0,0,0,0]],device=device) if s is None else s for s in batch.next_state]),
                        torch.cat(batch.action),
                        torch.cat(batch.reward),
                        torch.tensor([True if s is None else False for s in batch.next_state],device=device),
                        GAMMA)
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_action_values =estimation

    next_state_values = torch_zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = reference
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 600

for i_episode in range(num_episodes):

    env = gym.make("CartPole-v1")
    # Initialize the environment and get its state
    state, info = env.reset()

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        epsilon = epsilon * 0.9995 if epsilon * 0.9995 > 0.1 else 0.1

        action = epsilon_greedy_action(state, epsilon)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if i_episode%10==0:
            QNetTarget = QNet(4, 2, layers)
            target_net.load_state_dict(policy_net.state_dict())
            plot_durations()

        if done:
            episode_durations.append(t + 1)


            break
    print(i_episode)
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

