import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
import time


# optimized for Tf2
tf.disable_v2_behavior()

class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.A_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

# class ValueNetwork:
#     def __init__(self, state_size, learning_rate, name='state_value_network'):
#         self.state_size = state_size
#         self.learning_rate = learning_rate

#         with tf.variable_scope(name):

#             self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
#             self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
#             self.R_t = tf.placeholder(tf.float32, name="total_rewards")

#             tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
#             self.W1 = tf.get_variable("W1", [self.state_size, 8], initializer=tf2_initializer)
#             self.b1 = tf.get_variable("b1", [8], initializer=tf2_initializer)
#             self.W2 = tf.get_variable("W2", [8, 1], initializer=tf2_initializer)
#             self.b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)

#             self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
#             self.A1 = tf.nn.relu(self.Z1)
#             self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            
#             # Calculate the advantage function
#             self.advantage = self.R_t - tf.squeeze(self.output)
            
#             # Mean squared error loss
#             self.loss = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t)
#             self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.A_t = tf.placeholder(tf.float32, name="discounted_advantage")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 16], initializer=tf2_initializer)  # Increased size to 16
            self.b1 = tf.get_variable("b1", [16], initializer=tf2_initializer)  # Increased size to 16
            self.W2 = tf.get_variable("W2", [16, 8], initializer=tf2_initializer)  # Increased size to 8
            self.b2 = tf.get_variable("b2", [8], initializer=tf2_initializer)  # Increased size to 8
            self.W3 = tf.get_variable("W3", [8, 1], initializer=tf2_initializer)  # Added another layer to output
            self.b3 = tf.get_variable("b3", [1], initializer=tf2_initializer)  # Bias for output layer

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.Z2 = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            self.A2 = tf.nn.relu(self.Z2)
            self.output = tf.add(tf.matmul(self.A2, self.W3), self.b3)  # Output layer

            # Calculate the advantage function
            self.advantage = self.R_t - tf.squeeze(self.output)

            # Mean squared error loss
            self.loss = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)



def run(discount_factor, policy_learning_rate ,sv_learning_rate):
    env = gym.make('CartPole-v1')
    rewards, mean_rewards, losses = [], [], []
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    discount_factor = discount_factor
    policy_learning_rate = policy_learning_rate
    sv_learning_rate = sv_learning_rate
    render = False

    # Initialize the policy and the state-value network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, policy_learning_rate)
    state_value = ValueNetwork(state_size, sv_learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state, _ = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []
            since = time.time()
            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        time_elapsed = time.time() - since
                        print("Algorithm {} converged after {} seconds".format(algorithm_name, time_elapsed))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                # Compute Rt
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Gt
                
                # Compute advantage
                feed_dict = {state_value.state: transition.state, state_value.R_t: total_discounted_return}
                advantage = sess.run(state_value.advantage, feed_dict)
                
                # Update the state_value network weights
                feed_dict = {state_value.state: transition.state, state_value.R_t: total_discounted_return}
                _, loss_state = sess.run([state_value.optimizer, state_value.loss], feed_dict)

                # Update the policy network weights
                feed_dict = {policy.state: transition.state, policy.A_t: advantage, policy.action: transition.action}
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict)
            
            
            rewards.append(episode_rewards[episode])
            mean_rewards.append(average_rewards)
            losses.append(loss_policy)
    return episode, rewards, mean_rewards, losses


if __name__ == '__main__':
    optimal_sv_lr = 0.0007
    optimal_policy_lr = 0.0005
    optimal_df = 0.99
    algorithm_name = "reinforce_wt_baseline"
    last_episode, rewards, mean_rewards, losses = run(discount_factor=optimal_df, policy_learning_rate=optimal_policy_lr, sv_learning_rate=optimal_sv_lr)