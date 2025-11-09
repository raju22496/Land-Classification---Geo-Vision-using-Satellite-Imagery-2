import torch
import torch.nn as nn
import pytorch_lightning as pl

class SegmentationModel(pl.LightningModule):
    def __init__(self, n_classes=7):
        super().__init__()
        # Example: Simple U-Net architecture (you can replace with your own/model.pt)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def load_model(model_path='model.pt'):
    model = SegmentationModel()  # must match architecture used for saving
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model
