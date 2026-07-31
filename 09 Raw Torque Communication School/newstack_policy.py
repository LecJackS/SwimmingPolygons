"""Custom new-stack PPO RLModule for V9 motion/message mixed actions."""

from __future__ import annotations

from collections import OrderedDict
from math import log
from typing import Any, Dict

import gymnasium as gym
import numpy as np
import tree
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchCategorical,
    TorchDiagGaussian,
    TorchMultiDistribution,
)
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

from eval_utils import DEFAULT_NUM_MESSAGE_TOKENS, SHARED_POLICY_ID

torch, nn = try_import_torch()

DEFAULT_MOTION_STD_MIN = 0.15
DEFAULT_MOTION_STD_MAX = 1.0
DEFAULT_MOTION_STD_INIT = 0.35


def _activation_factory(name: str) -> type[nn.Module]:
    key = str(name).strip().lower()
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
        "identity": nn.Identity,
        "linear": nn.Identity,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported activation for V9 new-stack policy: {name}")
    return mapping[key]


class V9MotionMessagePPORLModule(TorchRLModule, ValueFunctionAPI):
    """Shared-policy RLModule with separate motion and message heads."""

    @override(RLModule)
    def setup(self) -> None:
        super().setup()
        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError(f"Expected Box observation space, got {self.observation_space!r}")
        if not isinstance(self.action_space, gym.spaces.Dict):
            raise TypeError(f"Expected Dict action space, got {self.action_space!r}")

        motion_space = self.action_space.spaces.get("motion")
        message_space = self.action_space.spaces.get("message")
        if not isinstance(motion_space, gym.spaces.Box) or len(tuple(motion_space.shape)) != 1:
            raise TypeError("V9 new-stack policy expects a 1-D continuous 'motion' Box.")
        self.motion_dim = int(motion_space.shape[0])
        if self.motion_dim <= 0:
            raise TypeError("V9 new-stack policy expects motion dimension > 0.")
        if not isinstance(message_space, gym.spaces.Discrete) or message_space.n != DEFAULT_NUM_MESSAGE_TOKENS:
            raise TypeError("V9 new-stack policy expects a 4-way discrete 'message' branch.")

        obs_dim = int(np.prod(self.observation_space.shape, dtype=np.int32))
        hidden_dims = [int(size) for size in self.model_config.get("fcnet_hiddens", [512, 512, 256])]
        activation_cls = _activation_factory(self.model_config.get("fcnet_activation", "tanh"))
        training_phase = str(self.model_config.get("training_phase", "forage_full")).strip().lower()
        self.phase_signal_dim = int(self.model_config.get("phase_signal_dim", 0))
        if self.phase_signal_dim not in {0, 2}:
            raise ValueError("V9 new-stack phase adapter expects phase_signal_dim 0 or 2.")
        self.base_obs_dim = obs_dim - self.phase_signal_dim
        if self.base_obs_dim <= 0:
            raise ValueError("V9 new-stack base observation dim must be > 0.")
        fixed_zero_mode = training_phase in {
            "locomotion_only",
            "locomotion_teacher",
            "locomotion_self",
            "locomotion_propulsion_easy",
            "locomotion_propulsion_robust",
        }
        self.message_head_mode = str(
            self.model_config.get(
                "message_head_mode",
                "fixed_zero" if fixed_zero_mode else "trainable",
            )
        )
        motion_std_min = float(self.model_config.get("motion_std_min", DEFAULT_MOTION_STD_MIN))
        motion_std_max = float(self.model_config.get("motion_std_max", DEFAULT_MOTION_STD_MAX))
        if motion_std_min <= 0.0 or motion_std_max < motion_std_min:
            raise ValueError("Invalid motion std bounds for V9 new-stack policy.")
        self.motion_log_std_min = float(log(motion_std_min))
        self.motion_log_std_max = float(log(motion_std_max))
        motion_std_init = float(np.clip(self.model_config.get("motion_std_init", DEFAULT_MOTION_STD_INIT), motion_std_min, motion_std_max))

        layers: list[nn.Module] = []
        in_dim = self.base_obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_cls())
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.motion_mean_head = nn.Linear(in_dim, self.motion_dim)
        self.phase_adapter = nn.Linear(self.phase_signal_dim, self.motion_dim, bias=False) if self.phase_signal_dim > 0 else None
        self.message_head = nn.Linear(in_dim, DEFAULT_NUM_MESSAGE_TOKENS)
        self.value_head = nn.Linear(in_dim, 1)
        self.motion_log_std = nn.Parameter(torch.full((self.motion_dim,), float(log(motion_std_init)), dtype=torch.float32))

        self._dist_child_struct = OrderedDict(
            [
                ("motion", TorchDiagGaussian),
                ("message", TorchCategorical),
            ]
        )
        self._input_lens_struct = OrderedDict(
            [
                ("motion", self.motion_dim * 2),
                ("message", DEFAULT_NUM_MESSAGE_TOKENS),
            ]
        )
        self._multi_dist_cls = TorchMultiDistribution.get_partial_dist_cls(
            space=self.action_space,
            child_distribution_cls_struct=self._dist_child_struct,
            input_lens=self._input_lens_struct,
        )

    def _split_observation(self, obs_batch: TensorType) -> tuple[TensorType, TensorType | None]:
        flat_obs = obs_batch.float().reshape(obs_batch.shape[0], -1)
        if self.phase_signal_dim <= 0:
            return flat_obs, None
        return flat_obs[:, : self.base_obs_dim], flat_obs[:, self.base_obs_dim :]

    def _encode(self, obs_batch: TensorType) -> tuple[TensorType, TensorType | None]:
        base_obs, phase_obs = self._split_observation(obs_batch)
        return self.encoder(base_obs), phase_obs

    def _fixed_message_logits(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> TensorType:
        logits = torch.full((batch_size, DEFAULT_NUM_MESSAGE_TOKENS), -20.0, dtype=dtype, device=device)
        logits[:, 0] = 20.0
        return logits

    def _compute_branch_logits(
        self,
        embeddings: TensorType,
        phase_obs: TensorType | None,
    ) -> tuple[TensorType, TensorType, TensorType]:
        motion_mean = self.motion_mean_head(embeddings)
        if self.phase_adapter is not None:
            if phase_obs is None:
                phase_obs = torch.zeros((embeddings.shape[0], self.phase_signal_dim), dtype=embeddings.dtype, device=embeddings.device)
            motion_mean = motion_mean + self.phase_adapter(phase_obs)
        motion_mean = torch.tanh(motion_mean)
        log_std = torch.clamp(self.motion_log_std, self.motion_log_std_min, self.motion_log_std_max)
        motion_logits = torch.cat([motion_mean, log_std.unsqueeze(0).expand(embeddings.shape[0], -1)], dim=-1)
        if self.message_head_mode == "fixed_zero":
            message_logits = self._fixed_message_logits(
                embeddings.shape[0],
                device=embeddings.device,
                dtype=embeddings.dtype,
            )
        else:
            message_logits = self.message_head(embeddings)
        logits_struct = OrderedDict(
            [
                ("motion", motion_logits),
                ("message", message_logits),
            ]
        )
        flat_logits = torch.cat(tree.flatten(logits_struct), dim=-1)
        return motion_logits, message_logits, flat_logits

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        embeddings, phase_obs = self._encode(batch[Columns.OBS])
        _, _, flat_logits = self._compute_branch_logits(embeddings, phase_obs)
        return {
            Columns.ACTION_DIST_INPUTS: flat_logits,
        }

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        embeddings, phase_obs = self._encode(batch[Columns.OBS])
        _, _, flat_logits = self._compute_branch_logits(embeddings, phase_obs)
        return {
            Columns.ACTION_DIST_INPUTS: flat_logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        embeddings, phase_obs = self._encode(batch[Columns.OBS])
        _, _, flat_logits = self._compute_branch_logits(embeddings, phase_obs)
        return {
            Columns.ACTION_DIST_INPUTS: flat_logits,
            Columns.EMBEDDINGS: embeddings,
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, TensorType],
        embeddings: Any = None,
    ) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._encode(batch[Columns.OBS])
        return self.value_head(embeddings).squeeze(-1)

    @override(TorchRLModule)
    def get_inference_action_dist_cls(self):
        return self._multi_dist_cls

    @override(TorchRLModule)
    def get_exploration_action_dist_cls(self):
        return self._multi_dist_cls

    @override(TorchRLModule)
    def get_train_action_dist_cls(self):
        return self._multi_dist_cls

    def load_motion_encoder_state_from(self, donor: "V9MotionMessagePPORLModule") -> list[str]:
        self.encoder.load_state_dict(donor.encoder.state_dict())
        self.motion_mean_head.load_state_dict(donor.motion_mean_head.state_dict())
        with torch.no_grad():
            self.motion_log_std.copy_(donor.motion_log_std.detach())
        return ["encoder", "motion_mean_head", "motion_log_std"]


def build_v9_newstack_multi_module_spec(
    *,
    observation_space: gym.Space,
    action_space: gym.Space,
    model_config: dict[str, Any],
    inference_only: bool = False,
) -> MultiRLModuleSpec:
    return MultiRLModuleSpec(
        rl_module_specs={
            SHARED_POLICY_ID: RLModuleSpec(
                module_class=V9MotionMessagePPORLModule,
                observation_space=observation_space,
                action_space=action_space,
                inference_only=inference_only,
                model_config=dict(model_config),
            )
        }
    )
