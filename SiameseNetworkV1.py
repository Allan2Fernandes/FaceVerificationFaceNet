import torch
import torch.nn as nn
# 95.5-96% accurate
class SiameseNetworkV1(nn.Module):
    def __init__(self, device):
        super(SiameseNetworkV1, self).__init__()
        self.hidden_layer1 = nn.Linear(in_features=512, out_features=256, device=device)
        self.hidden_layer2 = nn.Linear(in_features=256, out_features=128, device=device)
        self.hidden_layer3 = nn.Linear(in_features=128, out_features=128, device=device)
        self.classifier = nn.Linear(in_features=128,out_features=1, device=device)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.xavier_uniform_(self.hidden_layer1.weight)
        nn.init.xavier_uniform_(self.hidden_layer2.weight)
        nn.init.xavier_uniform_(self.hidden_layer3.weight)
        self.hidden_layer_activation1 = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_layer_activation2 = nn.LeakyReLU(negative_slope=0.01)
        self.hidden_layer_activation3 = nn.LeakyReLU(negative_slope=0.01)
        self.bn1 = nn.BatchNorm1d(num_features=256)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.activation = nn.Sigmoid()

    def forward(self, left_encoding, right_encoding):
        distance_tensor = torch.abs(right_encoding-left_encoding)
        hidden_activated_layer1 = self.hidden_layer_activation1(self.hidden_layer1(distance_tensor))
        hidden_activated_layer1_norm = self.bn1(hidden_activated_layer1)
        hidden_activated_layer2 = self.hidden_layer_activation2(self.hidden_layer2(hidden_activated_layer1_norm))
        hidden_activated_layer2_norm = self.bn2(hidden_activated_layer2)
        hidden_activated_layer3 = self.hidden_layer_activation3(self.hidden_layer3(hidden_activated_layer2_norm))
        hidden_activated_layer3_norm = self.bn3(hidden_activated_layer3)
        output_unactivated = self.classifier(hidden_activated_layer3_norm)
        output = self.activation(output_unactivated)
        return output

    def freeze_model(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False
            pass
        pass