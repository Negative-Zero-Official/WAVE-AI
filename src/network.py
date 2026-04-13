import math
import torch
from torch import nn
from config import HIDDEN_SIZE, NUM_LAYERS, OMEGA_0


"""
BUILDING BLOCK
"""

class SineLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            is_first: bool = False,
            omega_0: float = None
    ) -> None:
        super().__init__()
        if omega_0 is None:
            omega_0 = OMEGA_0
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()
    
    def _init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in (-1/fan_in, 1/fan_in)
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # Hidden layers: uniform in (+-sqrt(6/fan_in) / omega_0)
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


"""
FULL NETWORK
"""

class WAVENetwork(nn.Module):
    def __init__(
            self,
            in_features: int = 4,
            out_features: int = 4,
            hidden_size: int = HIDDEN_SIZE,
            num_layers: int = NUM_LAYERS,
            omega_0: float = None,
    ) -> None:
        super().__init__()

        # Use omega_0 from config if not specified (Phase 2: reduced to 20 for better A-field fitting)
        if omega_0 is None:
            omega_0 = OMEGA_0

        if num_layers < 1:
            raise ValueError("num_layers >= 1")
        
        self.omega_0 = omega_0
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.in_features = in_features
        self.out_features = out_features

        # Hidden layers
        layers = []
        layers.append(SineLayer(in_features, hidden_size, is_first=True, omega_0=omega_0))

        for _ in range(num_layers - 1):
            layers.append(
                SineLayer(hidden_size, hidden_size, is_first=False, omega_0=omega_0)
            )
        self.hidden = nn.Sequential(*layers)

        # Final linear layer (no activation)
        self.output_layer = nn.Linear(hidden_size, out_features)

        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_size) / omega_0
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()
    
    @property
    def final_weight(self):
        # Exposes the final layer weights for dynamic gradient balancing
        return self.output_layer.weight

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        h = self.hidden(coords)
        out = self.output_layer(h)

        # Scale outputs to reasonable physical magnitude
        out[:, 0:1] = out[:, 0:1] * 1.0   # phi
        out[:, 1:]  = out[:, 1:]  * 0.1   # A fields

        return out
    
    def split_potentials(self, coords: torch.Tensor):
        out = self.forward(coords)
        return (
            out[:, 0:1],    # Phi
            out[:, 1:2],    # Ax
            out[:, 2:3],    # Ay
            out[:, 3:4],    # Az
        )
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        return (
            f"WAVENetwork("
            f"in={self.in_features}, "
            f"hidden={self.hidden_size}x{self.num_layers}, "
            f"out={self.out_features}, "
            f"omega_0={self.omega_0}, "
            f"params={self.count_parameters():,}"
        )