import torch.nn.functional as F
from torch import nn


class Descendant(nn.Module):
    """Descendant descendantEncoder model for ADDA."""

    def __init__(self):
        """Init Descendant descendantEncoder."""
        super(Descendant, self).__init__()

        self.restored = False

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # self.fc1_descendant = nn.Linear(50 * 4 * 4, 500)
        # self.fc2_descendant = nn.Linear(500, 10)

    def forward(self, input):
        """Forward the Descendant."""
        conv_out = F.relu(self.pool1(self.conv1(input)))
        #feat = self.fc1_descendant(conv_out.view(-1, 50 * 4 * 4))
        #out = F.dropout(F.relu(feat), training=self.training)
        #out = self.fc2_descendant(out)
        out = conv_out
        return out
