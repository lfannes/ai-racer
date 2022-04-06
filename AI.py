import numpy as np
import random
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from env import RacerEnv
import tensorflow as tf

from collections import deque

env = RacerEnv()

EPOCHS = 1000
BATCH_SIZE = 150

epsilon = 1.0
EPSILON_REDUCE = 0.997

LEARNING_RATE = 1e-3
GAMMA = 0.95

model = Sequential()
model.add(Dense(64, input_shape=(env.n_observations, )))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(env.n_actions))
model.add(Activation('linear'))

target_model = clone_model(model)
print(model.summary())


def epsilon_greedy_action_selection(model, epsilon, observation):
    with tf.device('/cpu:0'):
        if np.random.random() > epsilon: #let the model predict the best action
            prediction = model.predict(np.array([observation, ]))
            action = np.argmax(prediction)
        else: #explore more of the environment
            action = env.getRandomAction()
        return action

replay_buffer = deque(maxlen=20000)
update_target_model = 10 #update the target model after ... epochs

def train(replay_buffer, model, target_model, batch_size):
    if len(replay_buffer) < batch_size:
        return

    minibatch = random.sample(replay_buffer, batch_size)

    #print("predicting targets and q_values...")
    current_states = np.array([data[0] for data in minibatch])
    current_qs_list = model.predict(current_states)

    new_current_states = np.array([data[3] for data in minibatch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    y = []

    #print("starting to update model based on previous data...")
    for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):

        # If not a terminal state, get new q from future states, otherwise set it to 0
        # almost like with Q Learning, but we use just part of equation here
        if not done:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + GAMMA * max_future_q
        else:
            new_q = reward

        # Update Q value for given state
        current_qs = current_qs_list[index]
        #print(f"current_qs: {current_qs}, action: {action}")
        current_qs[action] = new_q

        # And append to our training data
        X.append(current_state)
        y.append(current_qs)

        # Fit on all samples as one batch, log only on terminal state
    model.fit(np.array(X), np.array(y), batch_size=batch_size, verbose=0, shuffle=False)

def replay(replay_buffer, batch_size, model, target_model): #update the model based on previous decisions and rewards
    with tf.device('/cpu:0'):
        if len(replay_buffer) < batch_size:
            return

        samples = random.sample(replay_buffer, batch_size) #create a random sample with {batch_size} elements
        for i in range(batch_size):
            replay_buffer.pop()
        target_batch = [] #list of predictions by target_model

        zipped_samples = list(zip(*samples))
        states, actions, rewards, new_states, dones = zipped_samples #unzip the samples into different variables
        targets = []
        q_values = []

        print("predicting targets and q_values...")
        for i in range(batch_size):
            targets.append(target_model.predict(states[i])) #predict probability of the actions
            q_values.append(model.predict(new_states[i])) #predict q_value by state

        print("starting to update model based on previous data...")
        for i in range(batch_size):
            q_value = max(q_values[i][0]) #get the best action by q_value
            target = targets[i].copy()
            target = np.reshape(target, (1, np.product(target.shape)))
            if dones[i]:
                target[0][actions[i]] = rewards[i]
            else: #belmann equation to calc best action
                target[0][actions[i]] = rewards[i] + q_value[actions[i]] * GAMMA

            target_batch.append(target)

        #for i in range(batch_size):
        print(f"st: {list(zip(*states))}")
        print(f"\n{list(zip(*target_batch))}")
        model.fit(x=list(zip(*states)), y=list(zip(*target_batch)), batch_size=batch_size, epochs=2, verbose=2)

def update_model_handler(epoch, update_target_model, model, target_model): #update target model
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())
        model.save_weights("model.h5", overwrite=True)
        target_model.save_weights("target_model.h5", overwrite=True)

model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

#training routine
best_so_far = 0

model.save_weights("model.h5")

for epoch in range(EPOCHS):
    observation = env.reset()
    #observation = np.reshape(observation, (1, env.n_observations))

    done = False

    points = 0
    line = 0

    while not done:

        action = epsilon_greedy_action_selection(model, epsilon, observation)

        next_observation, reward, done, info = env.step(action)
        line = env.rewards.rewardIndex if env.rewards.rewardIndex > line else line
        #print("action:, ", str(action), ", obs: ", str(next_observation))
        #next_observation = np.reshape([next_observation], (1, env.n_observations))
        #print(np.array(next_observation).shape)
        #print(next_observation.shape)

        replay_buffer.append((observation, action, reward, next_observation, done)) #states, actions, rewards, new_states, dones
        train(replay_buffer, model, target_model, BATCH_SIZE)

        observation = next_observation #update observation
        points += reward

        #env.render("human")

    epsilon *= EPSILON_REDUCE

    update_model_handler(epoch, update_target_model, model, target_model)
    if points > best_so_far:
        best_so_far = points

    print(f"{epoch}: POINTS: {points}, LINE:{line} EPS: {epsilon}: BEST: {best_so_far}")

model.save_weights("model.h5", overwrite=True)
target_model.save_weights("target_model.h5", overwrite=True)