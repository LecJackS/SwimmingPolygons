from stable_baselines3 import PPO

from triangles import SwimmingAgentEnv

# Create the environment instance.
save_animation=True
env = SwimmingAgentEnv(render_every=1, save_animation=save_animation)

# Load model and test it again
model = PPO.load("ppo_swimming_agent")
obs = env.reset()
max_frames = 100
for i in range(max_frames):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
    
    if save_animation and i == max_frames - 1:
        env.save_animation_file("swimming_agent.gif", fps=10)

env.close()