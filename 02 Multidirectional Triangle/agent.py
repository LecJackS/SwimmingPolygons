from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from triangles import SwimmingAgentEnv
import torch as th

# Create the environment instance.
env = SwimmingAgentEnv(render_every=1)

# (Optional) Check that the environment follows the Gym interface.
#check_env(env, warn=True)

policy_kwargs = dict(
    net_arch=dict(pi=[16]*2,
                   vf=[16]*2),
    activation_fn=th.nn.Tanh
)

def lr_schedule(progress_remaining):
    lr_start = 1e-2
    lr_end = 1e-3
    # When progress_remaining = 1, lr = lr_start.
    # When progress_remaining = 0, lr = lr_end.
    return lr_start * ((lr_end / lr_start) ** (1 - progress_remaining))

# Load model and continue training it.
# model = PPO.load("ppo_swimming_agent",
#                  env
#                  )

# Initialize PPO with the MLP policy.
model = PPO("MlpPolicy", env,
            learning_rate=1e-2,
            gamma=0.99,
            verbose=2,
            n_steps=100*16,
            n_epochs=20,
            batch_size=8,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./ppo_swimming_tensorboard/",
            device="cpu")

# Train the model. Adjust total_timesteps as needed.
model.learn(total_timesteps=150_000,
            )

print("Saving the trained model...")
# Save the trained model.
model.save("ppo_swimming_agent")
print("Model saved!")

print("\nEvaluating the trained model...")
# Evaluate the trained model.
obs = env.reset()
for i in range(200):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Console code to run tensorboard: tensorboard --logdir ./ppo_swimming_tensorboard/


