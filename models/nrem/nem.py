import torch
import torch.nn as nn


class NeighborhoodExtraction(nn.Module):
    """
    Neighborhood Extraction Module (NEM)

    Input:
        x : (B, C, D, H, W)

    Output:
        center_features   : (B, C, D, H, W)
        neighbor_features : (B, C, D, H, W, 26)
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):

        # TODO:
        # 1. Pad input
        # 2. Extract 26 shifted neighbours
        # 3. Stack neighbours
        # 4. Return center and neighbours

        # Input Validation
        if x.dim() != 5:
            raise ValueError(
                f"Expected input shape (B, C, D, H, W), got {x.shape}"
            )

        B, C, D, H, W = x.shape


        # Pad by one voxel in every spatial dimension
        padded = torch.nn.functional.pad(
            x,
            pad=(1,1,1,1,1,1),
            mode="replicate"
        )

        # Relative offsets of the 26 neighbours
        offsets = []

        for dz in [-1,0,1]:
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:

                    if dz == 0 and dy == 0 and dx == 0:
                        continue

                    offsets.append((dz,dy,dx))

        # Extract all 26 neighbours
        neighbors = []

        for dz, dy, dx in offsets:

            neighbor = padded[
                :,
                :,
                1 + dz : 1 + dz + D,
                1 + dy : 1 + dy + H,
                1 + dx : 1 + dx + W,
            ]

            neighbors.append(neighbor)

        # Stack neighbours
        neighbor_features = torch.stack(
            neighbors,
            dim=-1
        )
        # Return center and neighbours
        center_features = x
        return center_features, neighbor_features