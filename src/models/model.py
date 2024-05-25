from torch import nn
import torch.nn.init as init

class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()

        self.layers = nn.Sequential(

            nn.Linear(14, 32),

            nn.ReLU(),

            nn.Linear(32, 16),

            nn.ReLU(),

            nn.Linear(16, 1),

            nn.Sigmoid()

        )

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Linear):

                init.kaiming_normal_(m.weight)

    def forward(self, inputs):

        return self.layers(inputs)

