import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from Environment import GTS_Environment
from GTS_Environment import GeneticToggle
env = GeneticToggle()



model = PPO(ActorCriticPolicy, env, verbose=1,gamma=1.0)

num_episodes = 10000
episode_length = 1
total_timesteps = num_episodes * episode_length
model.learn(total_timesteps=total_timesteps)

mean_rewards = []

for episode in range(num_episodes):
    obs = env.reset()
    episode_reward = 0

    for _ in range(episode_length):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if done:
            break

    mean_rewards.append(episode_reward / episode_length)


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

window_size = 10  # Adjust the window size according to your needs
moving_avg_rewards = moving_average(mean_rewards, window_size)

print("moving average rewards:", moving_avg_rewards)
print("aTc values:", env.aTc_values)
print("Length of aTc values",len(env.aTc_values))
print("IPTG values:",env.IPTG_values)
print("Length of IPTG values:", len(env.IPTG_values))
print("Euclidan distance:", env.euclidean_distances)
print("Length of ED:", len(env.euclidean_distances))

ISE = sum(env.errors) * 1
print(f"Integral of Squared Error (ISE): {ISE}")



# Set font size and weight
plt.rcParams['font.size'] = 11
plt.rcParams['font.weight'] = 'heavy'

plt.figure(dpi=200)
plt.plot(moving_avg_rewards, label='Moving Average', linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Mean Reward')
plt.grid()
plt.show()

plt.figure(dpi=200)
plt.plot(np.linspace(0,len(env.aTc_values), len(env.aTc_values)), env.aTc_values)
plt.xlabel("time")
plt.ylabel("aTc")
plt.grid()
plt.show()

plt.figure(dpi=200)
plt.plot(np.linspace(0,len(env.IPTG_values),len(env.IPTG_values)), env.IPTG_values)
plt.xlabel("Time")
plt.ylabel("IPTG")
plt.grid()
plt.show()

plt.figure(dpi=200)
plt.plot(np.linspace(0,len(env.euclidean_distances), len(env.euclidean_distances)), env.euclidean_distances)
plt.xlabel("Time (mins)")
plt.ylabel("Euclidean Distance")
plt.grid()
plt.show()


env.render()
