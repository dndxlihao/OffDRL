import torch
import torch.nn as nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def soft_clamp(x: torch.Tensor, _min=None, _max=None):
    if _max is not None:
        x = _max - F.softplus(_max - x)
    if _min is not None:
        x = _min + F.softplus(x - _min)
    return x


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        activation=None,
        layer_norm=True,
        with_residual=True,
        dropout=0.1
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = activation if activation is not None else Swish()
        self.layer_norm = nn.LayerNorm(output_dim) if layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.with_residual = with_residual
        self.res_proj = nn.Linear(input_dim, output_dim) if (with_residual and input_dim != output_dim) else None

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.dropout is not None:
            y = self.dropout(y)
        if self.with_residual:
            res = self.res_proj(x) if self.res_proj is not None else x
            y = res + y
        if self.layer_norm is not None:
            y = self.layer_norm(y)
        return y


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[200, 200, 200, 200],
        rnn_num_layers=2,
        dropout_rate=0.1,
        device="cpu"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.device = torch.device(device)

        self.activation = Swish()
        rnn_hid = self.hidden_dims[0]
        self.rnn_layer = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_hid,
            num_layers=rnn_num_layers,
            batch_first=True
        )

        module_list = []
        self.input_layer = ResBlock(input_dim, rnn_hid, dropout=dropout_rate, with_residual=False)
        dims = self.hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            module_list.append(ResBlock(in_dim, out_dim, dropout=dropout_rate))
        self.backbones = nn.ModuleList(module_list)

        # concat input_feature (rnn_hid) + rnn_output (rnn_hid) -> 2*rnn_hid
        self.merge_layer = nn.Linear(2 * rnn_hid, rnn_hid)
        self.output_layer = nn.Linear(self.hidden_dims[-1], output_dim)

        self.to(self.device)

    def forward(self, input, h_state=None):
    
        batch_size, num_timesteps, _ = input.shape
        input = torch.as_tensor(input, dtype=torch.float32).to(self.device)

        if h_state is not None:
            h0, c0 = h_state
            h0 = torch.as_tensor(h0, dtype=torch.float32).to(self.device)
            c0 = torch.as_tensor(c0, dtype=torch.float32).to(self.device)
            rnn_output, (h_n, c_n) = self.rnn_layer(input, (h0, c0))
        else:
            rnn_output, (h_n, c_n) = self.rnn_layer(input)

        rnn_hid = self.hidden_dims[0]
        rnn_output = rnn_output.reshape(-1, rnn_hid)  # (batch*seq, rnn_hid)
        inp = input.view(-1, self.input_dim)          # (batch*seq, input_dim)

        out = self.input_layer(inp)                   # (batch*seq, rnn_hid)
        out = torch.cat([out, rnn_output], dim=-1)    # (batch*seq, 2*rnn_hid)
        out = self.activation(self.merge_layer(out))  # (batch*seq, rnn_hid)

        for layer in self.backbones:
            out = layer(out)

        out = self.output_layer(out)                  # (batch*seq, output_dim)
        out = out.view(batch_size, num_timesteps, -1) # (batch, seq, output_dim)

        return out, (h_n, c_n)

if __name__ == "__main__":
    model = LSTM(14, 12)
    x = torch.randn(64, 20, 14)
    y, (hn, cn) = model(x)
    print(y.shape, hn.shape, cn.shape)
