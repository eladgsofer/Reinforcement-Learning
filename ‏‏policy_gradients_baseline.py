import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make('CartPole-v1')
np.random.seed(1)

class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")  # Change R_t to delta for advantage

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.delta)  # Use delta instead of R_t
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, 1], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.value = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            self.loss = tf.reduce_mean(tf.square(self.R_t - self.value))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

def run():
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004
    render = False

    # Initialize the policy and value networks
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    value_network = ValueNetwork(state_size, learning_rate)

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

            for step in range(max_steps):
                # Policy action
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
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Update policy and value networks
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:]))
                value_curr_state = sess.run(value_network.value, {value_network.state: transition.state})
                advantage = total_discounted_return - value_curr_state
                feed_dict_policy = {policy.state: transition.state, policy.delta: advantage, policy.action: transition.action}
                _, policy_loss = sess.run([policy.optimizer, policy.loss], feed_dict_policy)
                feed_dict_value = {value_network.state: transition.state, value_network.R_t: total_discounted_return}
                _, value_loss = sess.run([value_network.optimizer, value_network.loss], feed_dict_value)

if __name__ == '__main__':
    run()
