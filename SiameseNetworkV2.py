import torch
import torch.nn as nn


class SiameseNetworkV2(nn.Module):
    def __init__(self, device):
        super(SiameseNetworkV2, self).__init__()
        self.classifier = nn.Linear(in_features=512, out_features=1, device=device)
        nn.init.xavier_uniform_(self.classifier.weight)
        self.activation = nn.Sigmoid()

    def forward(self, left_encoding, right_encoding):
        distance_tensor = torch.abs(right_encoding - left_encoding)
        output_unactivated = self.classifier(distance_tensor)
        output = self.activation(output_unactivated)
        return output
