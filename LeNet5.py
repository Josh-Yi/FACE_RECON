from torch import nn
import torch
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=625, out_features=48),
            nn.Tanh(),
            nn.Linear(in_features=48, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        logits = self.classifier(x)
        # print(logits)
        probs = F.softmax(logits, dim=1)
        # print(probs)
        return probs