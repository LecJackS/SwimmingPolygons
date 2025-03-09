from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from triangles import SwimmingAgentEnv

# Create the environment instance.
env = SwimmingAgentEnv(render_every=1)

# (Optional) Check that the environment follows the Gym interface.
#check_env(env, warn=True)

policy_kwargs = dict(
    net_arch=[dict(pi=[8, 8],
                   vf=[8, 8])]
)

def lr_schedule(progress_remaining):
    lr_start = 1e-1
    lr_end = 1e-2
    # When progress_remaining = 1, lr = lr_start.
    # When progress_remaining = 0, lr = lr_end.
    return lr_start * ((lr_end / lr_start) ** (1 - progress_remaining))


# Initialize PPO with the MLP policy.
model = PPO("MlpPolicy", env,
            learning_rate=1e-2,
            verbose=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_swimming_tensorboard/")

# Train the model. Adjust total_timesteps as needed.
model.learn(total_timesteps=100_000)

print("Saving the trained model...")
# Save the trained model.
model.save("ppo_swimming_agent")
print("Model saved!")

print("\nEvaluating the trained model...")
# Evaluate the trained model.
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Console code to run tensorboard: tensorboard --logdir ./ppo_swimming_tensorboard/


