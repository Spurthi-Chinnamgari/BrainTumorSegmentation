import torch
import torch.nn as nn
import torch.nn.functional as F


class NeighborEvidenceEvaluation(nn.Module):

    def __init__(self):
        super().__init__()

        # Tiny MLP
        # Input:
        #   1. Cosine similarity
        #   2. Euclidean distance
        #
        # Output:
        #   Reliability score

        self.mlp = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        center_features,
        neighbor_features
    ):


        # Input validation

        if center_features.dim() != 5:
            raise ValueError(
                "center_features must have shape "
                "(B,C,D,H,W)"
            )

        if neighbor_features.dim() != 6:
            raise ValueError(
                "neighbor_features must have shape "
                "(B,C,D,H,W,26)"
            )

        if neighbor_features.shape[-1] != 26:
            raise ValueError(
                "neighbor_features must contain exactly "
                "26 neighbours."
            )


        # Prepare center features

        center = center_features.unsqueeze(-1)

        # Shape:
        # (B,C,D,H,W,1)

        # 1. Cosine similarity

        cosine_similarity = F.cosine_similarity(
            center,
            neighbor_features,
            dim=1,
            eps=1e-8
        )

        # Shape:
        # (B,D,H,W,26)

        # 2. Euclidean distance

        euclidean_distance = torch.norm(
            center - neighbor_features,
            p=2,
            dim=1
        )

        # Shape:
        # (B,D,H,W,26)

        # 3. Combine evidence

        evidence = torch.stack(
            [
                cosine_similarity,
                euclidean_distance
            ],
            dim=-1
        )

        # Shape:
        # (B,D,H,W,26,2)

        # 4. Tiny MLP

        reliability = self.mlp(
            evidence
        )

        # Shape:
        # (B,D,H,W,26,1)

        reliability = reliability.squeeze(-1)

        # Shape:
        # (B,D,H,W,26)

        # 5. Convert to reliability score [0,1]

        reliability = self.sigmoid(
            reliability
        )

        return reliability