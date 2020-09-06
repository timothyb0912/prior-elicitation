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

from .param_manager import ParamManager


@attr.s(eq=False, repr=False)
class LinearGLM(nn.Module):
    num_design_cols: int = attr.ib()
    weights: nn.Parameter = attr.ib(init=False)
    scale: nn.Parameter = attr.ib(init=False)
    param_manager: ParamManager = attr.ib()

    def __attrs_post_init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(
            torch.ones(self.num_design_cols, dtype=torch.double)
        )
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.double))

    @classmethod
    def from_input(cls, input_obj: torch.Tensor) -> "FoldedLogisticGLM":
        return cls(
            num_design_cols = input_obj.size()[1],
            param_manager = ParamManager(),
        )

    def forward(self, input_obj: torch.Tensor) -> torch.Tensor:
        location_params = torch.matmul(input_obj, self.weights)
        scale_params = self.scale * torch.ones(input_obj.size()[0])
        outputs = torch.cat(
            [location_params[:, None], scale_params[:, None]], dim=1
        )
        return outputs

    def simulate(
        self,
        input_obj: torch.Tensor,
        num_sim: int,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if seed is not None:
            torch.manual_seed(seed)
        # Get the predicted location and scale parameters
        predictions = self.forward(input_obj)
        locations, scales = predictions[:, 0], predictions[:, 1]
        # Get the current distributions
        current_dists = dists.Normal(loc=locations, scale=scales)
        # Sample from the distributions
        samples = current_dists.sample((num_sim,))
        return samples.numpy()

    def get_params_numpy(self) -> Tuple[
        np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]
    ]:
        return self.param_manager.get_params_numpy(self)

    def set_params_numpy(self, new_param_array: torch.Tensor) -> None:
        return self.param_manager.set_params_numpy(self, new_param_array)

    def get_grad_numpy(self) -> np.ndarray:
        return self.param_manager.get_grad_numpy(self)
