import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Architecture based on:

    Liu, Z. X., Yu, K., Dong, J., Zhao, L., Liu, Z., Zhang, Q., ... & Cheng, H. (2019).
    Precise prediction of calpain cleavage sites and their aberrance
    caused by mutations in cancer.
    Frontiers in Genetics, 10, 715.

    Yang, C., Li, C., Nip, K. M., Warren, R. L., & Birol, I. (2019).
    Terminitor: Cleavage site prediction using deep learning models.
    bioRxiv, 710699.
    """

    def __init__(self, input_size, hidden_linear1, hidden_linear2, out_neurons, dropout):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_linear1)
        self.fc2 = nn.Linear(hidden_linear1, hidden_linear2)
        self.fc_out = nn.Linear(hidden_linear2, out_neurons)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, return_features=None):
        # input shape: (batch_size, seq_len*vocab_size)
        layer1 = F.relu(self.fc1(seq))
        out = self.dropout(layer1)

        layer2 = F.relu(self.fc2(out))
        out = self.dropout(layer2)

        logits = self.fc_out(out)
        if return_features == 'fc1':
            return layer1
        if return_features == 'fc2':
            return layer2

        # default return logits
        return logits.squeeze()


class CNN(nn.Module):
    def __init__(self, input_channel, out_channel1,out_channel2, kernel_size1, kernel_size2,
                 stride1,stride2) -> None:
        super().__init__()

        self.cov1 = nn.Conv1d(input_channel, out_channel1, kernel_size=kernel_size1, stride=stride1,
                              padding=0)
        self.batchnorm = nn.BatchNorm1d(out_channel1)
        self.cov2 = nn.Conv1d(out_channel1, out_channel2, kernel_size=kernel_size2, stride=stride2,
                              padding=1)

        self.dense = nn.Linear(out_channel2,1)

    def forward(self, x):
        #x = x.permute(0, 2, 1).contiguous()
        x = x.unsqueeze(1)
        x = self.cov1(x)
        x = F.relu(x)
        x = self.cov2(x)
        x = F.relu(x)

        # Global average pooling over the sequence dimension
        x = x.mean(dim=-1)  # Now x is [batch_size, out_channel2]
        out = F.sigmoid(self.dense(x))  # [batch, 1]

        return out.squeeze()


class CNNmultichannel(nn.Module):
    def __init__(self, in_channels: int, out_channels: list[int], kernel_sizes: list[int], seq_len: int):
        super().__init__()
        assert len(out_channels) == len(kernel_sizes), "out_channels and kernel_sizes must have the same length."

        self.convs = nn.ModuleList()
        for out_ch, k_size in zip(out_channels, kernel_sizes):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_ch,
                        kernel_size=k_size,
                        padding='same'
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.MaxPool1d(2),
                )
            )
        # Flatten layer for FC transition
        self.flatten = nn.Flatten(start_dim=1)

        self.output_dim = sum(out_channels) * (seq_len // 2)  # Adjusted for MaxPool1d with kernel_size=2
        self.dense = nn.Linear(self.output_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, sequence_length, embedding_dim)
        x = x.permute(0, 2, 1)
        feature_flattened = [self.flatten(conv(x)) for conv in self.convs]
        x_out = torch.cat(feature_flattened, dim=1)
        x_out = self.dense(x_out)
        return x_out.squeeze()



