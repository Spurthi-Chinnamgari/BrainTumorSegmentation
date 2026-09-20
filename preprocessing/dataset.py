import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import PROCESSED_DATA_ROOT


class BrainTumorDataset(Dataset):

    def __init__(self, processed_root=PROCESSED_DATA_ROOT, split="train"):

        self.processed_root = processed_root
        self.split = split
        self.samples = []

        split_names = [split] if split not in (None, "all") else ["train", "val", "test"]

        for split_name in split_names:
            split_dir = os.path.join(processed_root, split_name)

            if not os.path.isdir(split_dir):
                continue

            for patient_id in sorted(os.listdir(split_dir)):
                patient_dir = os.path.join(split_dir, patient_id)

                if not os.path.isdir(patient_dir):
                    continue

                image_path = os.path.join(patient_dir, "images.npy")
                mask_path = os.path.join(patient_dir, "masks.npy")

                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue

                images = np.load(image_path, mmap_mode="r")
                patch_count = images.shape[0]

                for patch_index in range(patch_count):
                    self.samples.append(
                        (
                            image_path,
                            mask_path,
                            patch_index
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path, patch_index = self.samples[index]

        images = np.load(image_path, mmap_mode="r")
        masks = np.load(mask_path, mmap_mode="r")

        image = torch.from_numpy(images[patch_index].astype(np.float32))
        mask = torch.from_numpy(masks[patch_index].astype(np.int64))

        return image, mask


def create_dataloader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )