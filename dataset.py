import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

# Create dataset from TF flow:
''' 
Data is already read from the pickle file and preprocessed by utils.preprocess_data()
This class replaces the utils.create_dataset() and handles all the data related operation.

Style vector is cached.
style_vectors have shape [batch, 14, 1280]
'''


class IAMDataset(Dataset):
    def __init__(self, strokes, texts, cached_style_vector_path):
        """
        Params:
            strokes: [488, 3]: (dx, dy, pen_lifted)
            texts: [50] tokenized text, padded to maximum length
            style_vectors: images [].
        """
        self.strokes = list(strokes)
        self.texts = texts
        self.style_vectors = list(np.load(cached_style_vector_path))

    def __len__(self):
        return len(self.strokes)

    def __getitem__(self, idx):
        stroke, pen_lifts = self.strokes[idx][:, :2], self.strokes[idx][:, 2:]
        # stroke = np.transpose(stroke, [1, 0])
        style_vec = self.style_vectors[idx]
        style_vec = np.transpose(style_vec, [1, 0])  # to make the shape [B, 14, 1280]
        return stroke, pen_lifts, self.texts[idx], style_vec
