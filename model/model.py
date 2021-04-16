import torch
import torch.nn as nn
from . import initialization as init

class ChangeModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.change_head)

    def forward(self, x1, x2):
        """cat"""
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)
        
        decoder_output = self.decoder(features1, features2)

        masks = self.change_head(decoder_output)
        return masks

    def predict(self, x1, x2):

        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x1, x2)

        return x








