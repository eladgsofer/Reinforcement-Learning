import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

ENV = gym.make('CartPole-v1')  # graphics disabled


def temporal_difference(Qnet_target, Qnet, S, S_1, a, R, done, gamma):  # batched temporal difference function
    '''
        The temporal difference is our estimation to the model's error. This is a dynamic term which can be calculated in the
        middle of an episode. The function is written to work efficiently with a batch of inputs.
    '''
    reference = torch.zeros(done.shape)
    # if done then the reference signal is only the reward
    reference[done] = R[done]
    # if not done then the reference signal is the reward plus the estimated accumulated reward for state S+1 and the best action
    best_next_acc_reward = torch.max(Qnet_target(S_1), dim=1)[0].unsqueeze(
        1)  # the reference is with respect to the target Q
    reference[~done] = R[~done] + gamma * best_next_acc_reward[~done]

    a_reshaped = a.type(torch.int64).view(-1, 1)
    # the estimation uses the changing Qnet
    estimation = Qnet(S).gather(1,
                                a_reshaped)  # gather the estimated Q value chosen by action(not necessarly the best action)
    # temporal_diff = reference-estimation
    return estimation, reference


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class QNet(nn.Module):
    '''
        The deep learning model
    '''

    def __init__(self, input_states_size=ENV.observation_space.shape[0], output_actions_size=ENV.action_space.n,
                 hidden_layers_size=[16, 32, 16]):
        super(QNet, self).__init__()
        all_layers_sizes = np.zeros(len(hidden_layers_size) + 2, dtype=np.uint)
        layers = []
        all_layers_sizes[0] = input_states_size
        all_layers_sizes[-1] = output_actions_size
        all_layers_sizes[1:-1] = hidden_layers_size
        for i in range(len(all_layers_sizes) - 1):
            layers.append(nn.Linear(all_layers_sizes[i], all_layers_sizes[i + 1]))
            # layers.append(nn.BatchNorm1d(all_layers_sizes[i+1]))
            layers.append(nn.Dropout(0.05))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)


class DQN():
    '''
        A class encapsulating the Deep Q-learning process (includes the model's and the training and testing procedures)
    '''

    def __init__(self, batch_size, hidden_layers=[16, 32, 16], replay_buffer_memory_size=1000):
        self.batch_size = batch_size
        self.env = ENV
        self.hidden_layers = hidden_layers
        self.Qnet = QNet(hidden_layers_size=hidden_layers)
        self.QNetTarget = QNet(hidden_layers_size=hidden_layers)
        n_states = self.env.observation_space
        n_actions = self.env.action_space
        self.replay_buffer = []
        self.replay_buffer_idx = 0
        self.replay_buffer_memory_size = replay_buffer_memory_size
        self.replay_buffer_full = False
        self.acc_reward_list = []

    def sample_minibatch(self):
        '''
            get a random minibatch from the memory buffer(from the last 5000 experiences)
        '''
        minibatch = random.sample(self.replay_buffer,
                                  self.batch_size)  # p=...) we can add a distribution here according to the td value as explained in the lecture
        minibatch_dict = {
            "state": torch.vstack([tup[0] for tup in minibatch]),
            "action": torch.vstack([tup[1] for tup in minibatch]),
            "reward": torch.vstack([torch.tensor(tup[2]) for tup in minibatch]),
            "next_state": torch.vstack([torch.from_numpy(tup[3]) for tup in minibatch]),
            "done": torch.vstack([torch.tensor(tup[4]) for tup in minibatch])
        }
        return minibatch_dict

    def epsilon_greedy_action(self, epsilon, Qnet, state):
        '''
            Either take the best action with probabilty 1-epsilon or choose a random action with probability epsilon.
            This helps the model explore more and not get stuck in exploitation
        '''
        if np.random.uniform() > epsilon:  # ep
            action = torch.argmax(self.Qnet(state))
        else:
            action = torch.tensor(random.randint(0, self.env.action_space.n - 1))
        return action

    def append_to_replay_buffer(self, item):
        '''
            Add a new memory to the replay buffer.
            This is a cyclic buffer, it will start rewriting itself when it is full
        '''
        if self.replay_buffer_full:
            self.replay_buffer[self.replay_buffer_idx] = item
        else:
            self.replay_buffer.append(item)
        self.replay_buffer_idx = self.replay_buffer_idx + 1
        if self.replay_buffer_idx >= self.replay_buffer_memory_size:
            self.replay_buffer_idx = 0
            self.replay_buffer_full = True

    def train(self, n_episodes, T, epsilon, gamma, lr, C, improved_mode=False):
        '''
            Train the model for n episodes with a max iteration count of T per episode using epsilon greedy policy with
            a reward degradation of gamma a learning rate lr and update period for the target Q model of C iterations
        '''
        # weight_decay = 0.00005
        Qnet_optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=lr)
        step_counter = 0
        flag = False
        for ep in range(n_episodes):
            # add graphics every x episodes
            #            if ep%100==0:
            #               self.env = gym.make('CartPole-v1',render_mode="human") # graphics enabled
            #          else:
            self.env = gym.make('CartPole-v1')  # graphics disabled

            state, _ = self.env.reset()
            state = torch.tensor(state)
            acc_reward = 0
            ep_loss_list = []
            for t in range(T):  # max T steps in each experience

                # # Early stopping
                if sum(self.acc_reward_list[-100:])/100 > 475:
                    flag = True
                    break

                # choose next action
                epsilon = epsilon * 0.9998 if epsilon * 0.9998 > 0.001 else 0.001

                action = self.epsilon_greedy_action(epsilon, self.Qnet, state)

                # advance the environment
                next_state, reward, done, truncated, info = self.env.step(int(action))

                # save to memory
                single_step = (state, action, reward, next_state, done)
                self.append_to_replay_buffer(single_step)
                state = torch.tensor(next_state, dtype=torch.float32)

                acc_reward = acc_reward + reward
                # if done==True:
                #     reward = -10
                # print("blop")
                # reward = reward*(t**(0.5))

                if len(self.replay_buffer) < self.batch_size:  # only sample a batch if you have enough elements
                    continue

                minibatch = self.sample_minibatch()
                # TODO: from this part and forward, not fully tested

                # the error in DQN is the temporal difference function
                Qnet_optimizer.zero_grad()
                esstimation, reference = temporal_difference(self.QNetTarget,
                                                             self.Qnet,
                                                             minibatch["state"],
                                                             minibatch["next_state"],
                                                             minibatch["action"],
                                                             minibatch["reward"],
                                                             minibatch["done"],
                                                             gamma)
                # learning:

                # MSE_loss = torch.mean(error**2)
                criterion = nn.SmoothL1Loss()
                loss = criterion(esstimation, reference)
                loss.backward()
                Qnet_optimizer.step()
                step_counter = step_counter + 1
                ep_loss_list.append(loss.item())

                if done or truncated:  # debug print(if the model is learning then the accumulated reward should be increasing)
                    print("total reward  in episode {0} is {1} last Qnet {2} epsilon {3:.5f} avg loss {4:.4f}".format(
                        ep,
                        acc_reward,
                        self.Qnet(state).detach().numpy(),
                        epsilon,
                        sum(ep_loss_list) / len(ep_loss_list)))
                    self.acc_reward_list.append(acc_reward)
                    break

            if flag:
                break

            # update Q-target
            if improved_mode:
                target_net_state_dict = self.QNetTarget.state_dict()
                adjusting_net_state_dict = self.Qnet.state_dict()
                for key in adjusting_net_state_dict:
                    target_net_state_dict[key] = adjusting_net_state_dict[key] * 0.005 + target_net_state_dict[key] * (
                            1 - 0.005)
                self.QNetTarget.load_state_dict(target_net_state_dict)
            else:
                if ep % C == 0:
                    self.QNetTarget = type(self.Qnet)(hidden_layers_size=self.hidden_layers)
                    self.QNetTarget.load_state_dict(self.Qnet.state_dict())

        plt.plot(self.acc_reward_list, alpha=0.7)
        running_avg_acc_reward = np.convolve(np.array(self.acc_reward_list), np.ones(30) / 30, mode='valid')
        plt.figure()
        plt.plot(running_avg_acc_reward, linewidth=2.5)
        plt.ylabel("accumulated reward C: {0} LR: {1}".format(C, lr))
        plt.xlabel("episode")
        plt.legend(["accumulated reward", "average of last 30 runs"])
        # plt.show()


def main():

    T = 100000
    g = 0.99
    epsilon = 0.9
    lr = 0.0001
    episodes = 1450
    c = 5
    batch_size = 64
    replay_size = 10000
    # for c in range(2, 9, 2):
    #    for lr in np.linspace(0.0005,0.0005*10, 4):
    #        for g in np.linspace(0.9, 0.99, 4):
    # print("########################## C: {0} LR: {1} G: {2} ##########################".format(c, lr, g))
    epsilon = 1
    d = DQN(batch_size, hidden_layers=[128, 128, 128], replay_buffer_memory_size=replay_size)
    d.train(episodes, T, epsilon=epsilon, gamma=g, lr=lr, C=c,improved_mode=False)

    plt.show()


if __name__ == "__main__":
    main()

