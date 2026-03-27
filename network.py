import torch
from torch import nn

## For sin() activation function
class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x):
        return torch.sin(self.w0 * x)

class WAVENetwork(nn.Module):
    def __init__(self):
        super(WAVENetwork, self).__init__()
        # Input: 4 (x, y, z, t) -> Hidden -> Output: 4 (Phi, Ax, Ay, Az)

        # Store normalization parameters (will be set after domain bounds are known)
        self.register_buffer('x_min', torch.tensor(0.0))
        self.register_buffer('x_max', torch.tensor(1.0))
        self.register_buffer('y_min', torch.tensor(0.0))
        self.register_buffer('y_max', torch.tensor(1.0))
        self.register_buffer('z_min', torch.tensor(0.0))
        self.register_buffer('z_max', torch.tensor(1.0))
        self.register_buffer('t_min', torch.tensor(0.0))
        self.register_buffer('t_max', torch.tensor(1.0))

        self.net = nn.Sequential(
            nn.Linear(4, 128),
            Sine(10.0),
            nn.Linear(128, 128),
            Sine(1.0),
            nn.Linear(128, 128),
            Sine(1.0),
            nn.Linear(128, 4)
        )
    
    def set_normalization(self, x_min, x_max, y_min, y_max, z_min, z_max, t_min, t_max):
        """
        Store the physical bounds for input normalization
        """
        self.x_min.fill_(x_min)
        self.x_max.fill_(x_max)
        self.y_min.fill_(y_min)
        self.y_max.fill_(y_max)
        self.z_min.fill_(z_min)
        self.z_max.fill_(z_max)
        self.t_min.fill_(t_min)
        self.t_max.fill_(t_max)
    
    def normalize(self, x, y, z, t):
        """
        Map physical coordinates to [-1, 1]
        """
        x_norm = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1
        z_norm = 2 * (z - self.z_min) / (self.z_max - self.z_min) - 1
        t_norm = 2 * (t - self.t_min) / (self.t_max - self.t_min) - 1

        return torch.cat([x_norm, y_norm, z_norm, t_norm], dim=1)
    
    def forward(self, x, y, z, t):
        inputs = self.normalize(x, y, z, t)
        outputs = self.net(inputs)

        # Slice outputs into individual potential components
        Phi = outputs[:, 0:1]
        Ax = outputs[:, 1:2]
        Ay = outputs[:, 2:3]
        Az = outputs[:, 3:4]

        return Phi, Ax, Ay, Az