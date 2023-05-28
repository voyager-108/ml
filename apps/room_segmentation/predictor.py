import os
import numpy as np
from transformers import ViTImageProcessor, ViTModel
import torch

import numpy as np

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig

from apps.room_segmentation.embedding import example

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
        return self.model(input)

    def pred_list(self, images: list[np.ndarray]):
        images_ = torch.tensor(np.array(images))

        return self(images_)


if __name__ == "__main__":

    model = RoomClassifier.from_pretrained('ummagumm-a/samolet-room-classifier')

    # Check docs for that in the `embedding` file
    data = example()

    with torch.no_grad():
        pred = model.pred_list(data)

        print(pred.shape)


