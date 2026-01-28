import torch
import torch.nn as nn


class StateTargetEncoder(nn.Module):
    def __init__(self, sample_obs: dict):
        super().__init__()

        self.feature_size = 0
        state_proj_out = 128
        target_proj_out = 128

        self.state_encoder = None
        if "state" in sample_obs:
            state_in = sample_obs["state"].shape[-1]
            self.state_encoder = nn.Linear(state_in, state_proj_out)
            self.feature_size += state_proj_out

        self.target_encoder = None
        if "target" in sample_obs:
            target_in = sample_obs["target"].shape[-1]
            self.target_encoder = nn.Sequential(
                nn.Linear(target_in, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, target_proj_out),
            )
            self.feature_size += target_proj_out

    def forward(self, observations: dict) -> torch.Tensor:
        encoded = []

        if self.state_encoder is not None and "state" in observations:
            encoded.append(self.state_encoder(observations["state"]))
        if self.target_encoder is not None and "target" in observations:
            encoded.append(self.target_encoder(observations["target"]))

        if not encoded:
            raise ValueError("StateTargetEncoder received no valid inputs ('state'/'target').")

        return torch.cat(encoded, dim=1)
