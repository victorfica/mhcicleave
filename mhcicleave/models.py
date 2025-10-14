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


class CNNFeatureExtractor(nn.Module):
    """
    Extracts features from sequence embeddings using parallel 1D convolutions.
    Each convolution has a different kernel size to capture motifs of varying lengths.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: list[int]):
        super().__init__()

        # Create a list of convolutional layers to be applied in parallel
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding='same'  # 'same' padding ensures output length equals input length
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )

        # The output dimension will be the sum of channels from all parallel convolutions
        self.output_dim = out_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x shape: (batch_size, sequence_length, embedding_dim)
        # PyTorch Conv1d expects (batch_size, channels, length)
        x = x.permute(0, 2, 1)

        # Apply parallel convolutions and collect the outputs
        feature_maps = [conv(x) for conv in self.convs]

        # Concatenate the feature maps along the channel dimension
        x_out = torch.cat(feature_maps, dim=1)
        x_out = x_out.permute(0, 2, 1)  # Back to (batch_size, sequence_length, total_out_channels)

        # global average pooling over the sequence dimension
        x_out = x_out.mean(dim=1)

        return x_out  # Shape: (batch_size, total_out_channels)
