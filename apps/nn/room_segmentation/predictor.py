import os
import numpy as np
from transformers import ViTImageProcessor, ViTModel
import torch

import numpy as np

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig

from .embedding import example

class RoomRawClassifier(nn.Module):
    def __init__(self, input_dim=371, hidden_dim=50,
                 num_layers=1, num_classes=6,
                 dropout_p=0.01):
        super().__init__()
        self.lstm = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(2*hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

class RoomConfig(PretrainedConfig):
    model_type = 'gru'
    def __init__(self, input_dim=371, hidden_dim=50, num_layers=1,
                 num_classes=6, dropout_p=0.01, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.device = device

class RoomClassifier(PreTrainedModel):
    config_class = RoomConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = RoomRawClassifier(input_dim=self.config.input_dim,
                              hidden_dim=self.config.hidden_dim,
                              num_layers=self.config.num_layers,
                              num_classes=self.config.num_classes,
                              dropout_p=self.config.dropout_p)\
                              .to(self.config.device)

    def forward(self, input):
        input = input.to(self.config.device)
        return self.model(input)

    def pred_list(self, images: list[np.ndarray], batch_size: int = 7):
        preds = []
        images_ = torch.tensor(np.array(images), dtype=torch.float32)

        for i in range(0, len(images_), batch_size):
            preds.append(self(images_[i:i+batch_size]).cpu())

        preds = torch.vstack(preds)

        return self(images_)

    def to_device(self, device):
        self.config.device = device
        self.model = self.model.to(device)

        return self


if __name__ == "__main__":

    model = RoomClassifier.from_pretrained('ummagumm-a/samolet-room-classifier')
    model = model.to_device('cuda')

    # Check docs for that in the `embedding` file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    data = example(device)

    with torch.no_grad():
        data = np.array([np.hstack((x, np.zeros(23, ))) for x in data])
        pred = model.pred_list(data)

        print(pred.shape)


