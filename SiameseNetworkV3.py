import torch
import torch.nn as nn
class SiameseNetworkV3(nn.Module):
    def __init__(self, device):
        super(SiameseNetworkV3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3,3), stride=(1,1), bias=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), bias=True)
        self.max_pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.activation4 = nn.ReLU()

        self.encoing_layer = nn.Linear(in_features=36864, out_features=512, bias=True)
        self.activation5 = nn.ReLU()

        self.classification_layer = nn.Linear(in_features=512, out_features=1)
        self.classification_activation = nn.Sigmoid()

        pass

    def encode_batch(self, input_batch):
        x = self.conv1(input_batch)
        x = self.max_pool1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.max_pool3(x)
        x = self.activation3(x)

        x = self.conv4(x)
        x = self.max_pool4(x)
        x = self.activation4(x)

        x = x.view(x.size(0), -1)
        x = self.encoing_layer(x)
        x = self.activation5(x)
        return x

    def forward(self, left_batch, right_batch):
        left_encoding = self.encode_batch(left_batch)
        right_encoding = self.encode_batch(right_batch)
        classification_output = self.classification_activation(self.classification_layer(torch.abs(right_encoding-left_encoding)))
        return classification_output
