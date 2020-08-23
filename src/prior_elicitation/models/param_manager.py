"""
Classes for handling pytorch parameters and converting between torch and numpy.
"""
from typing import Any, Dict, Optional, Tuple

import attr
import botorch.optim.numpy_converter as numpy_converter
import botorch.optim.utils as optim_utils
import numpy as np
from botorch.optim.numpy_converter import TorchAttr


@attr.s
class ParamManager:

    def get_params_numpy(self, model: Any,) -> Tuple[
            np.ndarray, Dict[str, TorchAttr], Optional[np.ndarray]
    ]:
        """
        Syntatic sugar for `botorch.optim.numpy_converter.module_to_array`.

        Parameters
        ----------
        model : Any.
            An instance of nn.Module or one of its subclasses.

        Returns
        -------
        param_array : 1D np.ndarray
            Model parameters values.
        param_dict : dict.
            String representations of parameter names are keys, and the values
            are TorchAttr objects containing shape, dtype, and device
            information about the correpsonding pytorch tensors.
        bounds : optional, np.ndarray or None.
            If at least one parameter has bounds, then these are returned as a
            2D ndarray representing the bounds for each paramaeter. Otherwise
            None.
        """
        return numpy_converter.module_to_array(model)

    def set_params_numpy(
        self, model: Any, new_param_array: torch.Tensor,
    ) -> None:
        """
        Sets the model's parameters using the values in `new_param_array`.

        Parameters
        ----------
        new_param_array : 1D ndarray.
            Should have one element for each element of the tensors in
            `self.parameters`.

        Returns
        -------
        None.
        """
        # Get the property dictionary for this module
        _, property_dict, _ = self.get_params_numpy(model)
        # Set the parameters
        numpy_converter.set_params_with_array(
            model, new_param_array, property_dict
        )

    def get_grad_numpy(self, model: Any,) -> np.ndarray:
        """
        Returns the gradient of the model parameters as a 1D numpy array.
        """
        grad = np.concatenate(
            list(x.grad.data.numpy().ravel() for x in model.parameters()),
            axis=0
        )
        return grad
