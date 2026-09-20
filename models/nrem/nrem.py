import torch
import torch.nn as nn

from .nem import NeighborhoodExtraction
from .nee import NeighborEvidenceEvaluation
from .rnr import ResidualNeighborhoodRefinement


class NeighborhoodReliabilityEnhancedModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.nem = NeighborhoodExtraction()
        self.nee = NeighborEvidenceEvaluation()
        self.rnr = ResidualNeighborhoodRefinement()

    def forward(self, x):

        # --------------------------------------------------
        # 1. Neighborhood Extraction
        # --------------------------------------------------

        center_features, neighbor_features = self.nem(x)

        # --------------------------------------------------
        # 2. Neighborhood Evidence Evaluation
        # --------------------------------------------------

        reliability = self.nee(
            center_features,
            neighbor_features
        )

        # --------------------------------------------------
        # 3. Residual Neighborhood Refinement
        # --------------------------------------------------

        refined_features = self.rnr(
            center_features,
            neighbor_features,
            reliability
        )

        return refined_features