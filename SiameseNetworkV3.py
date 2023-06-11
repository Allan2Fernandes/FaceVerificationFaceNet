import torch
import torch.nn as nn
class SiameseNetworkV3(nn.Module):
    def __init__(self, device):
        super(SiameseNetworkV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(2,2), bias=True)
        self.activation1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.activation2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.activation3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(num_features=256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), bias=True)
        self.activation4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(num_features=512)

        self.encoding_layer = nn.Linear(in_features=41472, out_features=512, bias=True)
        self.activation5 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(num_features=512)

        self.hidden_layer_1 = nn.Linear(in_features=512, out_features=256, bias=True)
        self.activation6 = nn.ReLU()
        self.bn6 = nn.BatchNorm1d(num_features=256)

        self.hidden_layer_2 = nn.Linear(in_features=256, out_features=128, bias=True)
        self.activation7 = nn.ReLU()
        self.bn7 = nn.BatchNorm1d(num_features=128)

        self.classification_layer = nn.Linear(in_features=128, out_features=1)
        self.classification_activation = nn.Sigmoid()

        pass

    def encode_batch(self, input_batch):
        x = self.conv1(input_batch)
        x = self.activation1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.activation4(x)
        x = self.bn4(x)

        x = x.view(x.size(0), -1)
        x = self.encoding_layer(x)
        x = self.activation5(x)
        x = self.bn5(x)
        return x

    def decode_encodings(self, encoding1, encoding2):
        x = self.activation6(self.hidden_layer_1(torch.abs(encoding2 - encoding1)))
        x = self.bn6(x)

        x = self.activation7(self.hidden_layer_2(x))
        x = self.bn7(x)

        x = self.classification_activation(self.classification_layer(x))
        return x


    def forward(self, left_batch, right_batch):
        left_encoding = self.encode_batch(left_batch)
        right_encoding = self.encode_batch(right_batch)
        classification_output = self.decode_encodings(left_encoding, right_encoding)
        return classification_output
