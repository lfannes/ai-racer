import os
import math
import numpy as np
from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import TensorBoard
from env import RacerEnv
import matplotlib.pyplot as plt

class Episode:
    def __init__(self, start_place):
        self.reward = 0
        self.observations = []
        self.actions = []
        self.start_place = start_place

    def add_step(self, observation, action, reward):
        self.observations.append(observation[0])
        self.actions.append(action)
        self.reward += reward

    def __repr__(self):
        return "reward: " + str(self.reward)

class DQNAgent:
    def __init__(self):
        self.env = RacerEnv()
        self.episodes = 800
        self.required_episodes = 50 #episodes needed to train
        self.best_percentage = 20
        self.amount_best_eps = math.ceil(self.required_episodes * (self.best_percentage / 100))

        self.memory = [[] for i in range(self.env.rewards.numRaycasts)]
        self.eps = 1.
        self.eps_decay = 0.998

        self.points_history = []
        self.mean_point_history = []

        self.model = DQNAgent.get_model((self.env.n_observations, ), self.env.n_actions)

    def get_model(input_shape, output_shape):
        X_input = Input(input_shape)

        X = Dense(64, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)
        X = Dense(32, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(output_shape, activation="softmax", kernel_initializer='he_uniform')(X)

        model = Model(inputs=X_input, outputs=X, name='crossentropy_racer')
        model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01, rho=0.95, epsilon=0.01), metrics=["accuracy"])

        return model

    def reduce_eps(self):
        self.eps *= self.eps_decay

    def remember(self, observation, action, reward, done, start_place):
        if done or len(self.memory[start_place]) == 0:
            self.memory[start_place].append(Episode(start_place)) #[[], [], [], [], []]
        crossentropy_action = [0., 0.] #convert action to array -> NN can train
        crossentropy_action[action] = 1.
        self.memory[start_place][-1].add_step(observation, crossentropy_action, reward)

    def get_action(self, observation): #0: LEFT, 1: RIGHT
        self.reduce_eps()
        if np.random.random() >= self.eps:
            return self.env.getAction(weights=self.model(observation, training=False).numpy().flatten())
        else:
            return self.env.getAction()

    def train(self):
        print("training")

        def by_reward(episode):
            return episode.reward

        best_episodes = []
        for start_place in range(self.env.rewards.numRaycasts):
            self.memory[start_place].sort(key=by_reward, reverse=True)
            best_episodes.append(self.memory[start_place][0:self.amount_best_eps - 1])
            break

        #https://www.educative.io/edpresso/how-to-flatten-a-list-of-lists-in-python
        best_episodes_flatten = []
        for i in range(len(best_episodes)):  # Traversing through the main list
            for j in range(len(best_episodes[i])):  # Traversing through each sublist
                best_episodes_flatten.append(best_episodes[i][j])  # Appending elements into our flat_list

        X = [data.observations for data in best_episodes_flatten]
        y = [data.actions for data in best_episodes_flatten]

        self.model.fit(np.array(X[0]), np.array(y[0]), batch_size=self.amount_best_eps, verbose=0)
        self.memory = [[] for i in range(self.env.rewards.numRaycasts)]

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        print("Saving trained model as cartpole-dqn.h5")
        self.model.save(f"cartpole-dqn-{name}.h5")

    def run(self):
        i = 0
        for episode in range(self.episodes):
            meanReward = []
            for start_place in range(self.env.rewards.numRaycasts):
                observation = self.env.reset(start_place).reshape([1, self.env.n_observations])
                done = False
                episodeReward = 0
                while not done:
                    #self.env.render()
                    action = self.get_action(observation)
                    next_observation, reward, done, info = self.env.step(action)
                    self.remember(observation, action, reward, done, start_place)
                    episodeReward += reward
                    observation = next_observation.reshape([1, self.env.n_observations])
                meanReward.append((episodeReward))
                i += 1
                break

            print(f"{episode}/{self.episodes}: {sum(meanReward)/len(meanReward)}, eps: {self.eps}")
            self.mean_point_history.append(sum(meanReward)/len(meanReward))
            if i >= self.required_episodes:
                self.train()
                i = 0

if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    agent.save("final2")
    plt.plot([i for i in range(len(agent.mean_point_history))], agent.mean_point_history)
    plt.ylabel("points")
    plt.xlabel("epochs x12")
    plt.show()