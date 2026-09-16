"""Small, dependency-light loader for the published NLR PT-MELT model."""

import torch
from torch import nn


class _PoolHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(64, 64)
        # The published PT-MELT checkpoint contains v.weight but no v.bias.
        self.v = nn.Linear(64, 1, bias=False)

    def forward(self, x):
        scores = self.v(torch.tanh(self.w(x))).squeeze(-1)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        return (weights * x).sum(dim=1)


class _OutputHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_layer = nn.Linear(64, 1)
        self.log_var_layer = nn.Linear(64, 1)
        self.mix_coeffs_layer = nn.Linear(64, 1)


class _Layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn_block = nn.LSTM(input_size=15, hidden_size=64, num_layers=1, batch_first=True)
        self.pool_head = _PoolHead()
        self.output = _OutputHead()


class PTMELTHydrogenModel(nn.Module):
    """Architecture matching h2e_lstm_model_60s_attn_15inputs.safetensors."""

    def __init__(self):
        super().__init__()
        self.layer_dict = _Layers()

    def forward(self, x):
        sequence, _ = self.layer_dict.rnn_block(x)
        pooled = self.layer_dict.pool_head(sequence)
        mean = self.layer_dict.output.mean_layer(pooled)
        log_var = self.layer_dict.output.log_var_layer(pooled)
        return mean, log_var


def load_ptmelt_model(path, device="cpu"):
    from safetensors.torch import load_file

    model = PTMELTHydrogenModel().to(device)
    state = load_file(path, device=str(device))
    model.load_state_dict(state, strict=True)
    model.eval()
    return model
