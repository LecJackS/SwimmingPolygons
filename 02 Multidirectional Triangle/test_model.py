from stable_baselines3 import PPO
import numpy as np

from triangles import SwimmingAgentEnv

# Create the environment instance.
env = SwimmingAgentEnv(render_every=1)

# Load model and test it again
model = PPO.load("ppo_swimming_agent")
obs = env.reset()
#print("a0, a1 - x_pos, y_pos, x_vel, y_vel, orient, hinge, prev_rot, prev_hinge, food_x, food_y")
print("a0, a1 - hinge, prev_rot, prev_hinge")
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    print(np.round(action, 2), np.round(obs[5:8], 2), np.round(reward, 3))
    if done:
        obs = env.reset()
        print("-"*50)

env.close()