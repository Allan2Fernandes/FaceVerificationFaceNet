import torch
import torch.nn as nn

class SiameseNetworkProject(nn.Module):
    def __init__(self, device):
        super(SiameseNetworkProject, self).__init__()
        self.hidden_layer = nn.Linear(in_features=512, out_features=128, device=device)
        self.classifier = nn.Linear(in_features=128,out_features=1, device=device)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        self.hidden_layer_activation = nn.LeakyReLU(negative_slope=0.01)
        self.activation = nn.Sigmoid()

    def forward(self, left_encoding, right_encoding):
        distance_tensor = torch.abs(right_encoding-left_encoding)
        hidden_activated_layer = self.hidden_layer_activation(self.hidden_layer(distance_tensor))
        output_unactivated = self.classifier(hidden_activated_layer)
        output = self.activation(output_unactivated)
        return output

    def freeze_model(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False
            pass
        pass