import torch
from torch import nn

class WAVENetwork(nn.Module):
    def __init__(self):
        super(WAVENetwork, self).__init__()
        # Input: 4 (x, y, z, t) -> Hidden -> Output: 4 (Phi, Ax, Ay, Az)
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 4)
        )
    
    def forward(self, x, y, z, t):
        inputs = torch.cat([x, y, z, t], dim=1)
        outputs = self.net(inputs)

        # Slice outputs into individual potential components
        Phi = outputs[:, 0:1]
        Ax = outputs[:, 1:2]
        Ay = outputs[:, 2:3]
        Az = outputs[:, 3:4]

        return Phi, Ax, Ay, Az