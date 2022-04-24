import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import random
import gym
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard
from env import RacerEnv
import matplotlib.pyplot as plt

def OurModel(input_shape, action_space):
    X_input = Input(input_shape)

    X = Dense(64, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
    X = Dense(32, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model

class Episode:
    def __init__(self):
        self.episodeReward = 0
        self.observations = []
        self.actions = []

    def add_step(self, observation, action, reward):
        self.observations.append(np.asarray(observation))
        self.actions.append(np.asarray(action))
        self.episodeReward += reward

class DQNAgent:
    def __init__(self):
        self.env = RacerEnv()
        self.total_eps = 10000
        self.eps_start_training = 10
        self.best_eps_percentage = 25
        self.amount_best_eps = int(self.eps_start_training * (self.best_eps_percentage / 100))
        self.epsilon = 0
        self.epsilon_decay = 0.998

        self.memory = []

        self.points_history = []
        self.mean_point_history = []

        # create main model
        #self.model = OurModel(input_shape=(self.env.n_observations, ), action_space=self.env.n_actions)
        self.model = load_model("cartpole-dqn-4000.h5")

    def remember(self, observation, action, reward, done):
        if done or len(self.memory) == 0:
            self.memory.append(Episode())
        self.memory[-1].add_step(observation, action, reward)

    def reduce_eps(self):
        self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.getRandomAction()
        else:
            return np.argmax(self.model(state, training=False).numpy())


    def train(self):
        if len(self.memory) < self.eps_start_training:
            return

        print("training")

        X = [] #for observations
        y = [] #for action

        reward_list = np.zeros((self.eps_start_training, 2))
        for i in range(self.eps_start_training):
            reward_list[i][0] = i
            reward_list[i][1] = self.memory[i].episodeReward

        for n in range(self.eps_start_training):
            for i in range(self.eps_start_training - n - 1):
                if reward_list[i][1] > reward_list[i + 1][1]:
                    tmp = [reward_list[i + 1][0], reward_list[i + 1][1]]
                    reward_list[i + 1][0] = reward_list[i][0]
                    reward_list[i + 1][1] = reward_list[i][1]
                    reward_list[i][0] = tmp[0]
                    reward_list[i][1] = tmp[1]

        for i in range(self.amount_best_eps):
            for obs in self.memory[int(reward_list[self.eps_start_training - i - 1][0])].observations:
                X.append(np.asarray(obs))

            for act in self.memory[int(reward_list[self.eps_start_training - i - 1][0])].actions:
                y.append(np.asarray(act))

        self.model.fit(np.array(X), np.array(y), batch_size=self.amount_best_eps, verbose=0, epochs=3)
        self.memory = []

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        print("Saving trained model as cartpole-dqn.h5")
        self.model.save(f"cartpole-dqn-{name}.h5")

    def run(self):
        for e in range(self.total_eps):
            observation = self.env.reset()
            observation = np.reshape(observation, [1, self.env.n_observations])
            done = False
            epsReward = 0
            line = 0
            i = 0
            while not done:
                #self.env.render()
                action = self.act(observation)
                observation, reward, done, info = self.env.step(action)
                self.remember(observation, action, reward, done)
                observation = np.reshape(observation, [1, self.env.n_observations])
                line = self.env.rewards.rewardIndex if self.env.rewards.rewardIndex > line else line
                epsReward += reward
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {}, line: {}".format(e, self.total_eps, epsReward, self.epsilon, line))
                    self.points_history.append(epsReward)
                    if epsReward >= 3000:
                        self.save("3000")
            if e % 40 == 0:
                self.mean_point_history.append(np.mean(self.points_history))
                self.points_history = []
            self.train()
            self.reduce_eps()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    agent.save("final2")
    plt.plot([i for i in range(len(agent.mean_point_history))], agent.mean_point_history)
    plt.ylabel("points")
    plt.xlabel("epochs")
    plt.show()