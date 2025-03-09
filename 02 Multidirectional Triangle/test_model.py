from stable_baselines3 import PPO
import numpy as np

from triangles import SwimmingAgentEnv

# Create the environment instance.
save_animation = True
env = SwimmingAgentEnv(render_every=1, save_animation=save_animation)

# Load model and test it again

model = PPO.load("ppo_swimming_agent")
obs = env.reset()

#print("a0, a1 - x_pos, y_pos, x_vel, y_vel, orient, hinge, prev_rot, prev_hinge, food_x, food_y")
#print("a0, a1 - hinge, prev_rot, prev_hinge")

max_frames = 100
for i in range(max_frames):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    #print(np.round(action, 2), np.round(obs[5:8], 2), np.round(reward, 3))
    if done:
        obs = env.reset()

    if save_animation and i == max_frames - 1:
        env.save_animation_file("swimming_agent.gif", fps=10)

env.close()