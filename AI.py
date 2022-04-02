import numpy as np
import random
from keras.models import Sequential, clone_model
from keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from env import RacerEnv

from collections import deque

env = RacerEnv()

EPOCHS = 1000
BATCH_SIZE = 500

epsilon = 1.0
EPSILON_REDUCE = 0.995

LEARNING_RATE = 1e-3
GAMMA = 0.95

model = Sequential()
model.add(Dense(256, input_shape=[1, env.n_observations]))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(env.n_actions))
model.add(Activation('linear'))

target_model = clone_model(model)
print(model.summary())


def epsilon_greedy_action_selection(model, epsilon, observation):
    if np.random.random() > epsilon: #let the model predict the best action
        prediction = model.predict(observation)
        action = np.argmax(prediction)
    else: #explore more of the environment
        action = env.getRandomAction()
    return action

replay_buffer = deque(maxlen=20000)
update_target_model = 10 #update the target model after ... epochs

def replay(replay_buffer, batch_size, model, target_model): #update the model based on previous decisions and rewards
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

    for i in range(batch_size):
        model.fit(np.array(states[i]), np.array(target_batch[i]), epochs=1, verbose=0)

def update_model_handler(epoch, update_target_model, model, target_model): #update target model
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())

model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))

#training routine
best_so_far = 0

model.save_weights("model.h5")

for epoch in range(EPOCHS):
    observation = env.reset()
    observation = np.reshape(observation, (1, 1, env.n_observations))

    done = False

    points = 0

    while not done:
        print(env.steps)
        action = epsilon_greedy_action_selection(model, epsilon, observation)

        next_observation, reward, done, info = env.step(action)
        #print("action:, ", str(action), ", obs: ", str(next_observation))
        next_observation = np.reshape(next_observation, (1, 1, env.n_observations))
        #print(next_observation)

        replay_buffer.append((observation, action, reward, next_observation, done)) #states, actions, rewards, new_states, dones

        observation = next_observation #update observation
        points += reward

        replay(replay_buffer, BATCH_SIZE, model, target_model)

    epsilon *= EPSILON_REDUCE

    update_model_handler(epoch, update_target_model, model, target_model)
    if points > best_so_far:
        best_so_far = points

    print(f"{epoch}: POINTS: {points}: EPS: {epsilon}: BEST: {best_so_far}")

model.save_weights("model.h5", overwrite=True)