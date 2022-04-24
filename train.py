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

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(64, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X_input)

    # Hidden layer with 256 nodes
    X = Dense(32, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    # Output Layer with # of actions: 2 nodes (left, right)
    X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole_DQN_model')
    model.compile(loss="mse", optimizer=RMSprop(lr=0.0001, rho=0.95, epsilon=0.01), metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:
    def __init__(self):
        self.env = RacerEnv()
        # by default, CartPole-v1 has max episode steps = 500
        self.state_size = self.env.n_observations
        self.action_size = self.env.n_actions
        self.EPISODES = 20000
        self.memory = deque(maxlen=8000)
        self.name = "TEST1"
        self.tensorboard = TensorBoard(log_dir=f"logs/{self.name}")
        self.points_history = []
        self.mean_point_history = []

        self.gamma = 0.98  # discount rate
        self.epsilon = 0.25  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 2000

        # create main model
        #self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size)
        self.model = load_model("cartpole-dqn-4000.h5")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reduce_eps(self):
        if len(self.memory) > self.batch_size:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.random() <= self.epsilon:
            return self.env.getRandomAction()
        else:
            return np.argmax(self.model(state, training=False).numpy())


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # Randomly sample minibatch from the memory
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        target = self.model(state, training=False).numpy()
        target_next = self.model(next_state, training=False).numpy()

        for i in range(self.batch_size):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # Standard - DQN
                # DQN chooses the max Q value among next actions
                # selection and evaluation of action is on the target Q Network
                # Q_max = max_a' Q_target(s', a')
                target[i][action[i]] = reward[i] + self.gamma * np.amax(target_next[i])

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0, callbacks=[self.tensorboard])

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        print("Saving trained model as cartpole-dqn.h5")
        self.model.save(f"cartpole-dqn-{name}.h5")

    def run(self):
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            epsReward = 0
            line = 0
            i = 0
            while not done:
                #self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                epsReward += reward
                self.remember(state, action, reward, next_state, done)
                line = self.env.rewards.rewardIndex if self.env.rewards.rewardIndex > line else line
                state = next_state
                i += 1
                if done:
                    print("episode: {}/{}, score: {}, e: {}, line: {}".format(e, self.EPISODES, epsReward, self.epsilon, line))
                    self.points_history.append(epsReward)
                    if epsReward >= 3000:
                        self.save("3000")
            if e % 40 == 0:
                self.mean_point_history.append(np.mean(self.points_history))
                self.points_history = []
            self.replay()
            self.reduce_eps()

    def test(self):
        self.load("cartpole-dqn.h5")
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODES, i))
                    break


if __name__ == "__main__":
    agent = DQNAgent() 
    agent.run()
    agent.save("final2")
    plt.plot([i for i in range(len(agent.mean_point_history))], agent.mean_point_history)
    plt.ylabel("points")
    plt.xlabel("epochs")
    plt.show()