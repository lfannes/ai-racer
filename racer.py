import numpy as np
import random
from keras.models import Sequential, clone_model, Model
from keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from env import RacerEnv
import tensorflow as tf
import matplotlib.pyplot as plt
import gym

from collections import deque

class DQNAgent():
    def __init__(self, env):
        #TRAIN VARIABLE
        self.train_eps = 10000
        self.batch_size = 1000
        self.buffer_memory = deque(maxlen=8000)
        self.eps = 1.0
        self.eps_decay = 0.999
        self.gamma = 0.98
        self.target_model_update_rate = 50 #after ... eps update the weights and biases of target_model
        self.learning_rate = 1e-3

        self.env = env
        self.point_history = []
        self.mean_point_history = []

        #MODELS
        self.model = self.get_model()
        self.target_model = self.get_model()
        self.target_model.set_weights(self.model.get_weights())

    def get_model(self):
        X_input = Input((4,))

        X = Dense(32, input_shape=(4, ), activation="relu", kernel_initializer='he_uniform')(X_input)
        X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X_input)
        X = Dense(32, activation="relu", kernel_initializer='he_uniform')(X)

        X = Dense(2, activation="linear", kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='racer_network')
        model.compile(loss="mse", optimizer=Adam(learning_rate=self.learning_rate))

        model.summary()
        return model

    def get_q(self, observation):
        q_value = self.model(np.array(observation), training=False).numpy() #get Q value from model network
        return q_value

    def get_action(self, observation):
        if np.random.random() > self.eps:
            action = self.target_model(observation, training=False).numpy()
            action = np.argmax(action)
        else:
            action = random.randint(0, 1)
        return action

    def train(self):
        if len(self.buffer_memory) < self.batch_size:
            return

        print("aids")
        batch = random.sample(self.buffer_memory, self.batch_size)

        observation = np.zeros((self.batch_size, 4))
        next_observation = np.zeros((self.batch_size, 4))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            observation[i] = batch[i][0]
            action.append(batch[i][1])
            reward.append(batch[i][2])
            next_observation[i] = batch[i][3]
            done.append(batch[i][4])

        X = []
        y = []

        predicted_q = self.model(observation, training=False).numpy()
        predicted_q_next = self.model(next_observation, training=False).numpy()

        for i in range(self.batch_size): #observations, actions, rewards, new_observation, dones
            if done:
                predicted_q[i][action[i]] = reward[i]
            else:
                predicted_q[i][action[i]] = reward[i] + self.gamma * np.amax(predicted_q_next[i])
            #print(observation[0])
            X.append(observation[i])
            y.append(predicted_q[i])

        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, verbose=0)

    def reduce_eps(self):
        self.eps *= self.eps_decay

    def add_step(self, observation, action, reward, new_observation, done):
        self.buffer_memory.append((observation, action, reward, new_observation, done))

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

env = gym.make("CartPole-v1")
agent = DQNAgent(env)
best_reward = 0

for episode in range(agent.train_eps):
    observation = env.reset()
    observation = np.reshape(observation, [1, 4])
    epsReward = 0
    steps = 0
    done = False

    while not done:
        action = agent.get_action(observation)
        new_observation, reward, done, info = env.step(action) #observation, reward, done, info
        new_observation = np.reshape(new_observation, [1, 4])
        agent.add_step(observation, action, reward, new_observation, done)

        steps += 1
        epsReward += reward
        observation = new_observation

    agent.point_history.append(epsReward)
    if len(agent.point_history) % 40 == 0:
        agent.mean_point_history.append(np.mean(agent.point_history))
        agent.point_history = []
    if epsReward > best_reward:
        best_reward = epsReward
    print(f"{episode}/{agent.train_eps}: reward: {epsReward}, steps: {steps}, epsilon: {agent.eps}, best: {best_reward}")
    agent.reduce_eps()
    agent.train()

    if episode % agent.target_model_update_rate == 0:
        agent.update_target_model()

plt.plot([i for i in range(len(agent.mean_point_history))], agent.mean_point_history)
plt.ylabel("points")
plt.xlabel("epochs")
plt.show()




# import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# import random
# import gym
# import numpy as np
# from collections import deque
# from keras.models import Model, load_model
# from keras.layers import Input, Dense
# from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras.callbacks import TensorBoard
# from env import RacerEnv
# import matplotlib.pyplot as plt
#
#
# def OurModel(input_shape, action_space):
#     X_input = Input(input_shape)
#
#     # 'Dense' is the basic form of a neural network layer
#     # Input Layer of state size(4) and Hidden Layer with 512 nodes
#     X = Dense(32, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
#
#     # Hidden layer with 256 nodes
#     X = Dense(16, activation="relu", kernel_initializer='he_uniform')(X)
#
#     # Hidden layer with 64 nodes
#     X = Dense(32, activation="relu", kernel_initializer='he_uniform')(X)
#
#     # Output Layer with # of actions: 2 nodes (left, right)
#     X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)
#
#     model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
#     model.compile(loss="mse", optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=["accuracy"])
#
#     model.summary()
#     return model
#
#
# class DQNAgent:
#     def __init__(self):
#         self.env = gym.make("CartPole-v1")
#         # by default, CartPole-v1 has max episode steps = 500
#         self.state_size = 4
#         self.action_size = 2
#         self.EPISODES = 10000
#         self.memory = deque(maxlen=8000)
#         self.name = "TEST1"
#         self.tensorboard = TensorBoard(log_dir=f"logs/{self.name}")
#         self.points_history = []
#         self.mean_point_history = []
#
#         self.gamma = 0.98  # discount rate
#         self.epsilon = 1.0  # exploration rate
#         self.epsilon_min = 0.03
#         self.epsilon_decay = 0.999
#         self.batch_size = 1000
#         self.train_eps = 1000
#         #         self.batch_size = 1000
#         #         self.buffer_memory = deque(maxlen=8000)
#         #         self.eps = 1.0
#         #         self.eps_decay = 0.999
#         #         self.gamma = 0.98
#         #         self.target_model_update_rate = 50 #after ... eps update the weights and biases of target_model
#         #         self.learning_rate = 1e-3
#
#         # create main model
#         self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
#
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))
#
#     def reduce_eps(self):
#         if len(self.memory) > self.batch_size:
#             if self.epsilon > self.epsilon_min:
#                 self.epsilon *= self.epsilon_decay
#
#     def act(self, state):
#         if np.random.random() <= self.epsilon:
#             return random.randint(0, 1)
#         else:
#             return np.argmax(self.model(state, training=False).numpy())
#
#
#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         # Randomly sample minibatch from the memory
#         minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
#
#         state = np.zeros((self.batch_size, self.state_size))
#         next_state = np.zeros((self.batch_size, self.state_size))
#         action, reward, done = [], [], []
#
#         # do this before prediction
#         # for speedup, this could be done on the tensor level
#         # but easier to understand using a loop
#         for i in range(self.batch_size):
#             state[i] = minibatch[i][0]
#             action.append(minibatch[i][1])
#             reward.append(minibatch[i][2])
#             next_state[i] = minibatch[i][3]
#             done.append(minibatch[i][4])
#
#         # do batch prediction to save speed
#         target = self.model(state, training=False).numpy()
#         target_next = self.model(next_state, training=False).numpy()
#
#         for i in range(self.batch_size):
#             # correction on the Q value for the action used
#             if done[i]:
#                 target[i][action[i]] = reward[i]
#             else:
#                 # Standard - DQN
#                 # DQN chooses the max Q value among next actions
#                 # selection and evaluation of action is on the target Q Network
#                 # Q_max = max_a' Q_target(s', a')
#                 target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])
#
#         # Train the Neural Network with batches
#         self.model.fit(state, target, batch_size=self.batch_size, verbose=0, callbacks=[self.tensorboard])
#
#     def load(self, name):
#         self.model = load_model(name)
#
#     def save(self, name):
#         print("Saving trained model as cartpole-dqn.h5")
#         self.model.save(f"cartpole-dqn-{name}.h5")
#
#     def run(self):
#         for e in range(self.EPISODES):
#             state = self.env.reset()
#             state = np.reshape(state, [1, self.state_size])
#             done = False
#             epsReward = 0
#             i = 0
#             while not done:
#                 #self.env.render()
#                 action = self.act(state)
#                 next_state, reward, done, info = self.env.step(action)
#                 next_state = np.reshape(next_state, [1, self.state_size])
#                 epsReward += reward
#                 self.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 i += 1
#                 if done:
#                     print("episode: {}/{}, score: {}, e: {:.2}, steps: {}".format(e, self.EPISODES, epsReward, self.epsilon, i))
#                     self.points_history.append(epsReward)
#                     if epsReward >= 4000:
#                         self.save("4000")
#             if e % 40 == 0:
#                 self.mean_point_history.append(np.mean(self.points_history))
#                 self.points_history = []
#             self.replay()
#             self.reduce_eps()
#
#     def test(self):
#         self.load("cartpole-dqn.h5")
#         for e in range(self.EPISODES):
#             state = self.env.reset()
#             state = np.reshape(state, [1, self.state_size])
#             done = False
#             i = 0
#             while not done:
#                 self.env.render()
#                 action = np.argmax(self.model.predict(state))
#                 next_state, reward, done, _ = self.env.step(action)
#                 state = np.reshape(next_state, [1, self.state_size])
#                 i += 1
#                 if done:
#                     print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
#                     break
#
#
# if __name__ == "__main__":
#     agent = DQNAgent()
#     agent.run()
#     agent.save("final")
#     plt.plot([i for i in range(len(agent.mean_point_history))], agent.mean_point_history)
#     plt.ylabel("points")
#     plt.xlabel("epochs")
#     plt.show()