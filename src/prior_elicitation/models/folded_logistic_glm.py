"""
Pytorch implementation of a folded logistic regression model. Provides an
outcome distribution for modelling non-negative outcomes.
"""
from typing import Dict, Optional, Tuple

import attr
import numpy as np
import torch
import torch.distributions as dists
import torch.nn as nn
from botorch.optim.numpy_converter import TorchAttr

from .linear_glm import LinearGLM
from .param_manager import ParamManager


@attr.s(eq=False, repr=False, init=False)
class FoldedLogisticGLM(nn.Module):
    linear_model: LinearGLM = attr.ib()
    param_manager: ParamManager = attr.ib()

    def __init__(
        self, linear_model: LinearGLM, param_manager: ParamManager
    ) -> None:
        super().__init__()
        self.linear_model = linear_model
        self.param_manager = param_manager

    @classmethod
    def from_input(cls, input_obj: torch.Tensor) -> "FoldedLogisticGLM":
        return cls(
            linear_model=LinearGLM.from_input(input_obj),
            param_manager=ParamManager(),
        )

    def forward(self, input_obj: torch.Tensor) -> torch.Tensor:
        raw_outputs = self.linear_model(input_obj)
        positive_outputs = nn.functional.softplus(raw_outputs)
        return positive_outputs

    def simulate(
        self,
        input_obj: torch.Tensor,
        num_sim: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        if seed:
            torch.manual_seed(seed)
        # Get the predicted location and scale parameters
        predictions = self.forward(input_obj)
        locations, scales = predictions[:, 0], predictions[:, 1]
        # Build a Folded Logistic Di    stribution
        # X ~ Uniform(0, 1)
        # f = a + b * logit(X)
        # Y ~ f(X) ~ Logistic(a, b)
        # Z ~ |Y| ~ FoldedLogistic(a, b)
        base_distribution = dists.Uniform(0, 1)
        transforms = [
            dists.transforms.SigmoidTransform().inv,
            dists.transforms.AffineTransform(loc=locations, scale=scales),
            dists.transforms.AbsTransform(),
        ]
        folded_logistic_dists = (
            dists.transformed_distribution.TransformedDistribution(
                base_distribution, transforms
            )
        )
        # Sample from the distributions
        samples = folded_logistic_dists.sample((num_sim,))
        return samples

    def get_params_numpy(self) -> Tuple[
            np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]
    ]:
        return self.param_manager.get_params_numpy(self)

    def set_params_numpy(
        self, new_param_array: torch.Tensor,
    ) -> None:
        return self.param_manager.set_params_numpy(self, new_param_array)

    def get_grad_numpy(self) -> np.ndarray:
        return self.param_manager.get_grad_numpy(self)
