import os
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from datasets import load_dataset
import torch
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class RoomDimReducer:
    """
    Reduces the dimnsionality of embeddings produced by ViT
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


class RoomEmbedderPipeline:
    def __init__(self, use_gpu: bool = False):
        # Load the dataset of features used during training
        dds = load_dataset("ummagumm-a/frames_room_cls", use_auth_token=os.environ['HF_AUTH_TOKEN']
)
        train_ds = dds['train']
        test_ds = dds['test']

        # Fit the dimensionality reduction module
        self.dim_reducer = RoomDimReducer()
        self.dim_reducer.fit_transforms(np.array(train_ds['data']))

        # Delete the dataset to save the space
        del train_ds, test_ds, dds

        # Load a pretrained embedder
        self.processor = ViTImageProcessor.from_pretrained(
            'ummagumm-a/samolet_encoder_finetuned',
            use_auth_token=os.environ['HF_AUTH_TOKEN']
        )
        self.model = ViTModel.from_pretrained(
            'ummagumm-a/samolet_encoder_finetuned',
            use_auth_token=os.environ['HF_AUTH_TOKEN'],
            add_pooling_layer=False
        )

        if use_gpu:
            self.model = self.model.to(torch.device('cuda:0'))
            # self.processor = self.processor.to(torch.device('cuda:0'))

    def __call__(self, images: list[np.ndarray]):
        with torch.no_grad():
            # the following four lines produce embeddings for images
            inputs = self.processor(images=images, return_tensors="pt")
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            features = last_hidden_states.mean(axis=1)

            # reduce the dimensionality of these embeddings
            features = self.dim_reducer(features=features)

            return features


def example():
    # The following is the usage example
    # Insert the local path here
    video_path = os.path.join('data', 'raw', '1.mp4')
    # loads video
    cap = cv2.VideoCapture(video_path)

    # takes the first frame
    if cap.isOpened():
        ret, frame = cap.read()

    # produces embeddings for a list of frames
    room_embedder_pipeline = RoomEmbedderPipeline()
    x = room_embedder_pipeline([frame, frame, frame])

    return x


if __name__ == "__main__":
    x = example()
    print(x.shape)

