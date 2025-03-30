from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from torch.distributions import Categorical

#import gymnasium as gym
import gym
from triangles import OctopusEnv

import numpy as np

class PrintAvgDurationCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.env = env

    def _on_step(self):
        return super()._on_step()
    
    def on_rollout_end(self):
        if hasattr(self.env, "run_avg_duration"):
            print(f"Running average duration: {np.round(self.env.run_avg_duration, 2)} - Food max distance: {np.round(self.env.food_dist, 2)}")
        return super().on_rollout_end()
    
class ExponentialExplorationCallback(BaseCallback):
    def __init__(self, env, epsilon_max=1.0, epsilon_min=0.05, total_timesteps=100_000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.total_timesteps = total_timesteps
        self.k = np.log(epsilon_max / epsilon_min) / total_timesteps  # Compute decay rate

    def _on_rollout_start(self):
        # Compute exponentially decayed exploration rate
        new_epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * np.exp(-self.k * self.num_timesteps)
        
        # Update the environment with the new exploration rate
        if hasattr(self.env, "update_exploration"):
            self.env.update_exploration(new_epsilon)

        if self.verbose > 0:
            print(f"Step {self.num_timesteps}: Exploration rate = {new_epsilon:.3f}")

    def _on_step(self):
        return super()._on_step()

# Create the environment instance.
env = OctopusEnv(epsilon=0.0)
#env = gym.make("Acrobot-v1")
total_timesteps=10_000_000

# callback = ExponentialExplorationCallback(env,
#                                           epsilon_max=1.0,
#                                           epsilon_min=0.05,
#                                           total_timesteps=total_timesteps)

callback = PrintAvgDurationCallback()

class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom policy that:
    - Accepts an observation with shape (batch_size, 5, 10)
    - Uses a single encoder layer with 10 output channels, treating each feature independently.
    - Processes the encoded vector with a shared network.
    - Outputs two categorical action distributions (each with 3 options) and a value estimate.
    """
    def __init__(self, *args, **kwargs):
        super(CustomActorCriticPolicy, self).__init__(*args, **kwargs)
        hid_dim = 16  # Hidden dimension for the shared network
        
        nHist = 3
        nObs = 10 # Number of observation features
        # Encoder layers that maps (batch_size, nHist, nObs) -> (batch_size, nObs)
        enc_dim = 8
        self.encoder1 = nn.Linear(nHist, enc_dim, bias=True)  # 10 independent channels
        self.enc_relu1 = nn.ReLU()
        self.encoder2 = nn.Linear(enc_dim, 1, bias=True)
        #self.bn1 = nn.BatchNorm1d(nObs)
        self.enc_relu2 = nn.ReLU()
        
        # Shared network that processes the 10-dimensional encoded vector.
        self.shared_net = nn.Sequential(
            nn.Linear(nObs, hid_dim),
            #nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            # nn.Linear(hid_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            # nn.ReLU(),
            # nn.Linear(hid_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            # nn.ReLU()
        )
        
        # Policy heads for the two action dimensions.
        self.nA = 3  # Number of actions
        self.turn_head = nn.Linear(hid_dim, self.nA)   # Turning: 3 options.
        self.move_head = nn.Linear(hid_dim, self.nA)   # Movement: 3 options.
        
        # Value head.
        self.value_head = nn.Linear(hid_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.encoder1,
                      self.encoder2,
                      self.shared_net,
                      self.turn_head,
                      self.move_head,
                      self.value_head]:
            for m in layer.modules() if isinstance(layer, nn.Sequential) else [layer]:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, deterministic=False): 
        """
        obs: Tensor of shape (batch_size, 5, 10)
        Process:
        1. Transpose obs to (batch_size, 10, 5) to treat each feature as a channel.
        2. Apply the encoder to obtain a (batch_size, 10) encoded vector.
        3. Process the encoded vector with the shared network.
        4. Compute logits for the two action heads.
        5. Create categorical distributions and sample actions.
        6. Compute the combined log probability and value estimate.
        """
        # Transpose to (batch_size, 10, 5) so each feature is an independent channel
        obs_t = obs.transpose(1, 2)
        
        # Apply encoder: (batch_size, 10, 5) -> (batch_size, 10)
        encoded = self.encoder1(obs_t)
        encoded = self.enc_relu1(encoded)
        encoded = self.encoder2(encoded).squeeze(-1)  # Remove last unitary dimension (the encoded one)
        #encoded = self.bn1(encoded)
        encoded = self.enc_relu2(encoded)
        
        # Process the encoded vector with the shared network
        shared_out = self.shared_net(encoded)  # (batch_size, hid_dim)
        
        # Compute logits for each action head
        turn_logits = self.turn_head(shared_out)  # (batch_size, 3)
        move_logits = self.move_head(shared_out)  # (batch_size, 3)
        
        # Create categorical distributions
        turn_dist = Categorical(logits=turn_logits)
        move_dist = Categorical(logits=move_logits)
        
        # Sample actions or choose the most likely actions
        if deterministic:
            turn_action = turn_dist.probs.argmax(dim=-1)
            move_action = move_dist.probs.argmax(dim=-1)
        else:
            turn_action = turn_dist.sample()
            move_action = move_dist.sample()
        
        # Combine actions into a tensor of shape (batch_size, 2)
        actions = torch.stack([turn_action, move_action], dim=1)
        
        # Compute the log probabilities of the sampled actions
        log_prob = turn_dist.log_prob(turn_action) + move_dist.log_prob(move_action)
        
        # Compute the value estimate
        values = self.value_head(shared_out)
        
        return actions, values, log_prob
    
    def _predict(self, observation, deterministic=False):
        """
        Returns just the action (used for inference).
        """
        actions, _, _ = self.forward(observation, deterministic)
        return actions

class SimpleActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, *args, **kwargs):
        super(SimpleActorCriticPolicy, self).__init__(*args, **kwargs)
        hid_dim = 64  # Hidden dimension for the shared network
        
        nHist = 1
        nObs = 10 # Number of observation features
        # Encoder layers that maps (batch_size, nHist, nObs) -> (batch_size, nObs)
        enc_dim = 8
        
        # Shared network that processes the 10-dimensional encoded vector.
        self.shared_net = nn.Sequential(
            nn.Linear(nObs, hid_dim),
            #nn.BatchNorm1d(hid_dim),
            nn.Tanh(),
            nn.Linear(hid_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            nn.Tanh(),
            # nn.Linear(hid_dim, hid_dim),
            # nn.BatchNorm1d(hid_dim),
            # nn.ReLU()
        )
        
        # Policy heads for the two action dimensions.
        self.nA = 3  # Number of actions
        self.turn_head = nn.Linear(hid_dim, self.nA)   # Turning: 3 options.
        self.move_head = nn.Linear(hid_dim, self.nA)   # Movement: 3 options.
        
        # Value head.
        self.value_head = nn.Linear(hid_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for layer in [self.shared_net,
                      self.turn_head,
                      self.move_head,
                      self.value_head]:
            for m in layer.modules() if isinstance(layer, nn.Sequential) else [layer]:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, obs, deterministic=False): 

        # Rechape obs to be flatten but without making a copy
        x = obs.view(obs.size(0), -1)

        # Process the encoded vector with the shared network
        x = self.shared_net(x)  # (batch_size, hid_dim)
        
        # Compute logits for each action head
        turn_logits = self.turn_head(x)  # (batch_size, 3)
        move_logits = self.move_head(x)  # (batch_size, 3)
        
        # Create categorical distributions
        turn_dist = Categorical(logits=turn_logits)
        move_dist = Categorical(logits=move_logits)
        
        # Sample actions or choose the most likely actions
        if deterministic:
            turn_action = turn_dist.probs.argmax(dim=-1)
            move_action = move_dist.probs.argmax(dim=-1)
        else:
            turn_action = turn_dist.sample()
            move_action = move_dist.sample()
        
        # Combine actions into a tensor of shape (batch_size, 2)
        actions = torch.stack([turn_action, move_action], dim=1)
        
        # Compute the log probabilities of the sampled actions
        log_prob = turn_dist.log_prob(turn_action) + move_dist.log_prob(move_action)
        
        # Compute the value estimate
        values = self.value_head(x)
        
        return actions, values, log_prob
    
    def _predict(self, observation, deterministic=False):
        """
        Returns just the action (used for inference).
        """
        actions, _, _ = self.forward(observation, deterministic)
        return actions



policy_kwargs = dict(
    net_arch=dict(pi=[64,64,32,16],
                  vf=[64,64,32,16]),
    #activation_fn=torch.nn.Tanh,
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
model = PPO(
    "MlpPolicy", #TODO: Ver por qué la custom policy similar a MLP policy no funciona igual
    #CustomActorCriticPolicy,
    #SimpleActorCriticPolicy,
    env,
    #policy_kwargs=policy_kwargs,
    learning_rate=1e-3,
    gamma=0.9,
    verbose=2,
    n_steps=100*32,
    n_epochs=10,
    batch_size=100,
    ent_coef=0.01,
    #clip_range=0.1,
    #vf_coef=0.5,
    #max_grad_norm=0.2,
    tensorboard_log="./ppo_swimming_tensorboard/",
    device="cpu"
)

# Train the model. Adjust total_timesteps as needed.
model.learn(total_timesteps=total_timesteps,
            callback=callback
            )

print("Saving the trained model...")
# Save the trained model.
model.save("ppo_swimming_agent")
print("Model saved!")

print("\nEvaluating the trained model...")
# Evaluate the trained model.
#obs = env.reset()
obs = env.reset()
for i in range(10_000):
    action, _states = model.predict(obs, deterministic=True)
    # Convert action to integer
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

# Console code to run tensorboard: tensorboard --logdir ./ppo_swimming_tensorboard/


