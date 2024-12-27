import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class ReIDSiamese(nn.Module):
    def __init__(self):
        super(ReIDSiamese, self).__init__()

        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")

        # Freeze first 5 layers
        for i, param in enumerate(self.backbone.parameters()):
            if i < 5:
                param.requires_grad = False

        self.conv = nn.Conv2d(1280, 4096, 1, padding="same")
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(40960, 1)
        self.sigmoid = nn.Sigmoid()

    def forward_one(self, x):
        x = self.backbone.extract_features(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)

        # L1 distance
        distance = torch.abs(output1 - output2)
        distance = self.dropout(distance)

        out = self.fc(distance)
        out = self.sigmoid(out)

        return out


if __name__ == "__main__":
    model = ReIDSiamese()

    input_1 = torch.randn(1, 3, 160, 80)
    input_2 = torch.randn(1, 3, 160, 80)  # (batch_size, channels, height, width)

    output = model(input_1, input_2)
    print(output)
