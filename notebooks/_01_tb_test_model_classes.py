# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import numpy as np
import scipy.stats
import torch
from sklearn.datasets import load_boston

from prior_elicitation.models.folded_logistic_glm import FoldedLogisticGLM
from prior_elicitation.models.linear_glm import LinearGLM
# -

data_container = load_boston()

training_design_np = data_container["data"]
training_outcomes_np = data_container["target"].ravel()
training_design_column_names = data_container["feature_names"]

training_design_torch = torch.from_numpy(training_design_np).double()
training_outcomes_torch = torch.from_numpy(training_outcomes_np).double()

linear_model = LinearGLM.from_input(training_design_torch)
folded_logistic_model = FoldedLogisticGLM.from_input(training_design_torch)

with torch.no_grad():
    linear_preds = linear_model(training_design_torch)
    folded_preds = folded_logistic_model(training_design_torch)

# +
# Choose the model to simulate from
current_model = folded_logistic_model

# Establish a number of simulations from the prior
NUM_PRIOR_SIM = 100

# Create a default prior
prior_info = {
    key: {"dist": "norm", "loc": 0, "scale": 1} for key in training_design_column_names
}
prior_info["scale"] = {"dist": "foldnorm", "c": 0, "loc": 10, "scale": 1}

# Set a random seed for reproducility
SEED = 129
np.random.seed(SEED)
torch.manual_seed(SEED)

# Simulate parameters from the prior
prior_sim_parameters = np.empty((len(prior_info), NUM_PRIOR_SIM), dtype=float)

for pos, key in enumerate(training_design_column_names):
    if prior_info[key]["dist"] == "norm":
        prior_sim_parameters[pos, :] = scipy.stats.norm.rvs(
            loc = prior_info[key]["loc"],
            scale = prior_info[key]["scale"],
            size=NUM_PRIOR_SIM,
        )
    elif prior_info[key]["dist"] == "foldnorm":
        prior_sim_parameters[pos, :] = scipy.stats.foldnorm.rvs(
            c = prior_info[key]["c"],
            loc = prior_info[key]["loc"],
            scale = prior_info[key]["scale"],
            size=NUM_PRIOR_SIM,
        )
    
print(prior_sim_parameters.shape)

# +
# Draw from the prior predictive distribution
prior_sim_outcomes = np.empty((training_design_np.shape[0], NUM_PRIOR_SIM), dtype=float)

with torch.no_grad():
    for i in range(NUM_PRIOR_SIM):
        current_params = prior_sim_parameters[:, i]
        current_model.set_params_numpy(current_params)
        prior_sim_outcomes[:, i] = current_model.simulate(training_design_torch, num_sim=1).numpy()
        
print(prior_sim_outcomes.shape)
# -

# # To-Do:
# Add desired prior predictive plots.

# +
# Make desired prior predictive plots
