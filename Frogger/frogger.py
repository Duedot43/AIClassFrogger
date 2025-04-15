# REQUIRES PYTHON 3.12.0
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam
import cv2
import random
from collections import deque
import time
import os
from ale_py import ALEInterface, roms


def reduceFrame(frame): # Make frame B&W and slim it down to reduce compute time
    gar=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    crap=gar[12:-30, 7:-7]
    resized = cv2.resize(crap, (84, 110), interpolation=cv2.INTER_AREA)
    return resized 


def build_model(action_size):
    """CNN architecture"""
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(110, 84, 1)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))
    return model


# Epsilon-greedy policy -- balances exploration and exploitation
def epsilon_greedy_action(model, state, epsilon, action_size):
    """Choose action using epsilon-greedy policy"""
    if np.random.random() <= epsilon:
        return random.randrange(action_size)  # Explore: choose random action
    else:
        # Exploit: choose best action
        q_values = model.predict(np.expand_dims(state, axis=0), verbose=0)[0]
        return np.argmax(q_values)  # Choose action with highest Q-value
    

def makeSnapshotSystem(size): # Make the memory space for snapshots and history tracking
    return deque(maxlen=size)

def addSnapshot(snapshots, state, action, reward, next_state, done): # TODO I need to finish this 
    snapshots.append((state, action, reward, next_state, done))

def sampleSnapshot(memory, batch_size):
    return random.sample(memory, batch_size)



def trainModel(episodes=1000, # How many episodes to run
               maxStep=50000, # Max ammount of steps the frog can take before giving up   
               epsilon=1.0, # Initial epsilon this is used for determining weather to use knowlage or explore
               epsilon_decay=.99,
               snapshots_size=100000
               ):
    
    env = gym.make("ALE/Frogger-v5", render_mode='rgb_array')
    

    action_size = env.action_space.n
    model = build_model(action_size) # Build the model with all the possible action spaces

    snapshots = makeSnapshotSystem(snapshots_size)
    print("Running Episodes")
    for episode in range(episodes):
        episode_reward = 0
        total_steps = 0

        frame, info= env.reset()
        state = reduceFrame(frame)
        state = np.expand_dims(state, axis=-1)


        for step in range(maxStep):
            print(f"steps taken {step}/{maxStep} episode: {episode}/{episodes}", end="\r")
            action = epsilon_greedy_action(model, state, epsilon, action_size)

            next_frame, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_state = reduceFrame(next_frame)
            next_state = np.expand_dims(next_state, axis=-1)

            addSnapshot(snapshots, state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward
            total_steps += 1

            if len(snapshots) >= snapshots_size:
                # Wipe the model clean and start new with previously learned data
                previousActions = sampleSnapshot(snapshots, snapshots_size)

                previousActionsArray = {
                    "states": [],
                    "actions": [],
                    "rewards": [],
                    "next_states": [],
                    "dones": []
                }

                for snapshot in previousActions:
                    previousActionsArray['states'].append(snapshot[0])
                    previousActionsArray['actions'].append(snapshot[1])
                    previousActionsArray['rewards'].append(snapshot[2])
                    previousActionsArray['next_states'].append(snapshot[3])
                    previousActionsArray['dones'].append(snapshot[4])

                previousActionsArray['states'] = np.expand_dims(previousActionsArray['states'], axis=-1)
                

                q_values = model.predict(previousActionsArray['states'])

                model.fit(previousActionsArray['states'], q_values, epochs=1, verbose=0)
                snapshots = makeSnapshotSystem(snapshots_size)
            if done:
               break

                
                


trainModel()