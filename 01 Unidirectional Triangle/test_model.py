from stable_baselines3 import PPO

from triangles import SwimmingAgentEnv

# Create the environment instance.
env = SwimmingAgentEnv(render_every=1)

# Load model and test it again
model = PPO.load("ppo_swimming_agent")
obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()