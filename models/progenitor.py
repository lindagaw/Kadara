"""Progenitor model for ADDA."""

import torch.nn.functional as F
from torch import nn


class Progenitor(nn.Module):
    """Progenitor progenitorEncoder model for ADDA."""

    def __init__(self):
        """Init Progenitor progenitorEncoder."""
        super(Progenitor, self).__init__()

        self.restored = False


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3)
        self.pool9 = nn.MaxPool2d(kernel_size=2)

        self.conv10 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3)
        self.pool12 = nn.MaxPool2d(kernel_size=2)

        self.conv13 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv14 = nn.Conv2d(512, 512, kernel_size=1)
        self.pool15 = nn.MaxPool2d(kernel_size=1)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512*3*3, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 31)

    def forward(self, input):
        """Forward the Progenitor."""
        conv_out = self.pool3(F.relu(self.conv2(F.relu(self.conv1(input)))))
        conv_out = self.pool6(F.relu(self.conv5(F.relu(self.conv4(conv_out)))))
        conv_out = self.pool9(F.relu(self.conv8(F.relu(self.conv7(conv_out)))))
        conv_out = self.pool12(F.relu(self.conv11(F.relu(self.conv10(conv_out)))))
        conv_out = self.pool15(F.relu(self.conv14(F.relu(self.conv13(conv_out)))))


        #print(conv_out.shape)
        #feat = self.fc1(conv_out.view(-1, 128*29*29))

        feat = self.fc3(F.relu(self.fc2(F.relu(self.fc1(self.flatten(conv_out))))))
        out = F.dropout(F.relu(feat), training=self.training)

        return out
