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
    X = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="categorical_crossentropy", optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

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
        self.total_eps = 8000
        self.eps_start_training = 100
        self.best_eps_percentage = 20
        self.amount_best_eps = int(self.eps_start_training * (self.best_eps_percentage / 100))
        self.epsilon = 1.0
        self.epsilon_decay = 0.998

        self.memory = []
        self.changePos = False

        self.points_history = []
        self.mean_point_history = []

        # create main model
        self.model = OurModel(input_shape=(self.env.n_observations, ), action_space=self.env.n_actions)
        #self.model = load_model("cartpole-dqn-4000.h5")

    def remember(self, observation, action, reward, done):
        if done or len(self.memory) == 0:
            self.memory.append(Episode())
        crossentropyAction = [0., 0.]
        crossentropyAction[np.argmax(action)] = 1.
        self.memory[-1].add_step(observation, crossentropyAction, reward)

    def reduce_eps(self):
        self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.getAction()
        else:
            return self.env.getAction(weights=self.model(state, training=False).numpy().flatten())


    def train(self):
        if len(self.memory) < self.eps_start_training:
            return
        self.changePos = True
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

        self.model.fit(np.array(X), np.array(y), batch_size=self.amount_best_eps, verbose=0, epochs=1)
        self.memory = []

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        print("Saving trained model as cartpole-dqn.h5")
        self.model.save(f"cartpole-dqn-{name}.h5")

    def getData(self):
        

        print("score: {}, e: {}, line: {}".format(epsReward, self.epsilon, self.line))
        self.points_history.append(epsReward)
        if epsReward >= 3000:
            self.save("3000")
        self.train()
        self.reduce_eps()

    def run(self):
        for e in range(self.total_eps):
            observation = self.env.reset(self.changePos)
            self.changePos = False
            #observation = np.reshape(observation, [1, self.env.n_observations])
            done = False
            epsReward = 0
            line = 0
            i = 0
            while not done:
                #self.env.render()
                observation = np.reshape(observation, [1, self.env.n_observations])
                action = self.act(observation)
                observation, reward, done, info = self.env.step(action)

                self.remember(observation, action, reward, done)

                line = self.env.linesCrossed
                epsReward += reward
                i += 1

            print(f"{e}: score: {epsReward}, epsilon: {self.epsilon}, start place: {self.env.startPlace}, line: {line}")
            self.reduce_eps()
            self.train()


if __name__ == "__main__":
    agent = DQNAgent()
    agent.run()
    agent.save("final2")
    plt.plot([i for i in range(len(agent.mean_point_history))], agent.mean_point_history)
    plt.ylabel("points")
    plt.xlabel("epochs")
    plt.show()