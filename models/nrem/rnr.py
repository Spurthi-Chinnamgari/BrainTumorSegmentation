import torch
import torch.nn as nn


class ResidualNeighborhoodRefinement(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        center_features,
        neighbor_features,
        reliability
    ):

        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------

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

        if reliability.dim() != 5:
            raise ValueError(
                "reliability must have shape "
                "(B,D,H,W,26)"
            )

        if neighbor_features.shape[-1] != 26:
            raise ValueError(
                "neighbor_features must contain exactly "
                "26 neighbours."
            )

        if reliability.shape[-1] != 26:
            raise ValueError(
                "reliability must contain exactly "
                "26 weights."
            )

        # --------------------------------------------------
        # Weighted neighborhood aggregation
        # --------------------------------------------------

        weights = reliability.unsqueeze(1)

        # Shape:
        # (B,1,D,H,W,26)

        weighted_neighbors = (
            neighbor_features * weights
        )

        # Shape:
        # (B,C,D,H,W,26)

        aggregated_neighbors = (
            weighted_neighbors.sum(dim=-1)
        )

        # Shape:
        # (B,C,D,H,W)

        # --------------------------------------------------
        # Residual refinement
        # --------------------------------------------------

        refined_features = (
            center_features +
            aggregated_neighbors
        )

        return refined_features