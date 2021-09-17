import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# TensorFlow ≥2.0 is required
import tensorflow as tf
from tensorflow import keras

assert tf.__version__ >= "2.0"

import matplotlib as mpl

env = gym.make("CartPole-v1")


def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch


def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim


def render_policy_net(model, n_max_steps=200, seed=42):
    frames = []
    env = gym.make("CartPole-v1")
    env.seed(seed)
    np.random.seed(seed)
    obs = env.reset()
    for step in range(n_max_steps):
        frames.append(env.render(mode="rgb_array"))
        left_proba = model.predict(obs.reshape(1, -1))
        action = int(np.random.rand() > left_proba)
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    return frames
    # totals = []
    # for episode in range(500):
    #     episode_rewards = 0
    #     obs = env.reset()
    #     for step in range(200):
    #         action = basic_policy(obs)
    #         obs, reward, done, info = env.step(action)
    #         episode_rewards += reward
    #         if done:
    #             break
    #     totals.append(episode_rewards)
    # env.close()


keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

n_inputs = 4  # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

frames = render_policy_net(model)
plot_animation(frames)
