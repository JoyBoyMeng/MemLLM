import torch
import numpy as np


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output


class MergeLayer(torch.nn.Module):
    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class ExpandMemory(torch.nn.Module):
    def __init__(self, input_dim=172, output_dim=4096, hidden_dims=[512, 1024, 2048]):
        super(ExpandMemory, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after the last layer
                layers.append(torch.nn.ReLU(inplace=True))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FixEmbed(torch.nn.Module):
    def __init__(self, input_dim=4096 * 4, hidden_dim=4096, output_dim=4096):
        super(FixEmbed, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class ShrinkMessage(torch.nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, output_dim=172):
        super(ShrinkMessage, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Score(torch.nn.Module):
    def __init__(self, input_dim=172, hidden_dim=64, output_dim=1):
        super(Score, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ReductionLayer(torch.nn.Module):
  def __init__(self, input_dim, output_dim):
    super(ReductionLayer, self).__init__()
    self.fc1 = torch.nn.Linear(input_dim, 1024)
    self.fc2 = torch.nn.Linear(1024, 512)
    self.fc3 = torch.nn.Linear(512, 256)
    self.fc4 = torch.nn.Linear(256, output_dim)
    self.relu = torch.nn.ReLU()

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    x = self.relu(self.fc3(x))
    x = self.fc4(x)
    return x