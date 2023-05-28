import os
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from datasets import load_dataset
import torch
import torch.nn as nn
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class RoomDimReducer:
    """
    Reduces the dimnsionality of embeddings produced by ViT.
    """
    def __init__(self):
        self.ss = StandardScaler()
        self.pca = PCA(n_components=0.99)

    def fit_transforms(self, ds):
        self.ss = self.ss.fit(ds)
        self.pca = self.pca.fit(self.ss.transform(ds))

    def __call__(self, features: list[np.ndarray]):
        features = self.ss.transform(features)
        features = self.pca.transform(features)

        return features


class RoomEmbedderPipeline(nn.Module):
    def __init__(self, device='cpu'):
        # Load the dataset of features used during training
        dds = load_dataset("ummagumm-a/frames_room_cls", use_auth_token=os.environ['HF_AUTH_TOKEN']
)
        super().__init__()
        train_ds = dds['train']
        test_ds = dds['test']

        # Fit the dimensionality reduction module
        self.dim_reducer = RoomDimReducer()
        self.dim_reducer.fit_transforms(np.array(train_ds['data']))

        # Delete the dataset to save the space
        del train_ds, test_ds, dds

        self.device = device

        # Load a pretrained embedder
        self.processor = ViTImageProcessor.from_pretrained(
            'ummagumm-a/samolet_encoder_finetuned',
            use_auth_token=os.environ['HF_AUTH_TOKEN']
        )
        self.model = ViTModel.from_pretrained(
            'ummagumm-a/samolet_encoder_finetuned',
            use_auth_token=os.environ['HF_AUTH_TOKEN'],
            add_pooling_layer=False
        ).to(self.device)

    def __call__(self, images: list[np.ndarray], batch_size: int = 7):
        with torch.no_grad():
            # the following four lines produce embeddings for images
            inputs = self.processor(images=images, return_tensors="pt")
            # all tensors to a specific device: 'cpu' or 'cuda'
            # inputs['pixel_values'] = inputs['pixel_values'].to(self.device)
            preds = []
            for i in range(0, len(images), batch_size):
                inp = {}
                inp['pixel_values'] = inputs['pixel_values'][i:i+batch_size].to(self.device)
                out = self.model(**inp)
                out = out.last_hidden_state.mean(axis=1).cpu()
                preds.append(out)

            features = torch.vstack(preds)

#            outputs = self.model(batch_size=8, **inputs)
#            last_hidden_states = outputs.last_hidden_state
#            features = last_hidden_states.mean(axis=1)

            # reduce the dimensionality of these embeddings
            features = self.dim_reducer(features=features.cpu())

            return features


def example(device):
    # The following is the usage example
    # Insert the local path here
    video_path = os.path.join('data', 'raw', '1.mp4')
    # loads video
    cap = cv2.VideoCapture(video_path)

    # takes the first frame
    if cap.isOpened():
        ret, frame = cap.read()

    # produces embeddings for a list of frames
    room_embedder_pipeline = RoomEmbedderPipeline(device)
    x = room_embedder_pipeline([frame, frame, frame] * 100)

    return x


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device:", device)
    x = example(device)
    print(x.shape)

