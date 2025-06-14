import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.color import rgb2lab
from sklearn.neighbors import NearestNeighbors
import torch


class ColorizationDataset(Dataset):
    def __init__(self, rgb_dir, image_size=256):
        self.rgb_dir = rgb_dir
        self.image_size = image_size
        self.filenames = []
        for root, _, files in os.walk(rgb_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    self.filenames.append(full_path)

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor()
        ])

        # Load color bin centers
        self.quantized_bins = np.load("resources/pts_in_hull.npy")  # (313, 2)
        self.nn = NearestNeighbors(n_neighbors=1)
        self.nn.fit(self.quantized_bins)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.filenames[idx]
        filename = os.path.basename(path)

        rgb_image = Image.open(path).convert('RGB')
        rgb_tensor = self.transform(rgb_image)
        rgb_np = rgb_tensor.permute(1, 2, 0).numpy()  # [H, W, C]

        lab = rgb2lab(rgb_np).astype("float32")
        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:]  # Do not normalize for classification

        # Quantize ab values to nearest bin
        ab_flat = ab.reshape(-1, 2)  # (H*W, 2)
        _, indices = self.nn.kneighbors(ab_flat)  # (H*W, 1)
        q_ab = indices.reshape(L.shape[0], L.shape[1]).astype(np.int64)  # (H, W)

        # Convert to torch tensors
        L_tensor = torch.from_numpy(L).unsqueeze(0).float()      # (1, H, W)
        q_ab_tensor = torch.from_numpy(q_ab).long()              # (H, W)

        # print("L_tensor shape:", L_tensor.shape, "min/max:", L_tensor.min().item(), L_tensor.max().item())
        # print("q_ab_tensor shape:", q_ab_tensor.shape, "dtype:", q_ab_tensor.dtype, 
        #     "min/max:", q_ab_tensor.min().item(), q_ab_tensor.max().item())


        return L_tensor, q_ab_tensor, path