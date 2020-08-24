"""
UI for prior understanding via interactive, prior setting and predictive plots.
"""
from copy import deepcopy

import papermill as pm
import pyprojroot
import streamlit as st
from PIL import Image

# Set the app's title
st.title("UI for Interactive Prior Predictive Checks")
# Create each of the needed arguments for the notebook
MODEL = st.selectbox(
    "What model should we look at?",
    ["folded-logistic", "linear"],
    index=0,
)

param_list = [
    "Age",
    "B",
    "CHAS",
    "CRIM",
    "DIS",
    "INDUS",
    "LSTAT",
    "NOX",
    "PTRATIO",
    "RAD",
    "RM",
    "TAX",
    "ZN",
    "intercept",
    "scale",
]
COLUMN = st.selectbox(
    "What column should we visualize?",
    param_list[:-1],
    index=param_list[:-1].index("intercept"),
)

#####
# Create button for each parameter
#####
params_to_set = st.multiselect(
    "Which priors should we view/set/adjust?",
    param_list,
    default=["intercept", "scale"]
)

def make_distribution_box(param_name: str) -> str:
    dist_index = 1 if param_name == "scale" else 0
    box_description = "Which distribution for {}?".format(param_name)
    return st.selectbox(
        box_description,
        ["norm", "foldnorm"],
        index=dist_index,
        key="{}_distribution_box".format(param_name),
    )

def make_location_slider(param_name: str) -> int:
    location_value = 10 if param_name == "scale" else 0
    slider_min = 0 if param_name == "scale" else -10
    slider_max = 20 if param_name == "scale" else 10
    slider_description = "What location parameter for {}?".format(param_name)
    return st.slider(
        slider_description,
        min_value = slider_min,
        max_value = slider_max,
        value = location_value,
    )

def make_scale_slider(param_name: str) -> int:
    scale_value = 1.0
    slider_min = 0.5
    slider_max = 5.0
    slider_step = 0.5
    slider_description = "What scale parameter for {}?".format(param_name)
    return st.slider(
        slider_description,
        min_value = slider_min,
        max_value = slider_max,
        value = scale_value,
        step = slider_step
    )

param_display_options = {}
for selected_param in params_to_set:
    param_display_options[selected_param] = {}
    st.text(selected_param)
    param_display_options[selected_param]["dist"] = make_distribution_box(
        selected_param,
    )
    param_display_options[selected_param]["loc"] = make_location_slider(
        selected_param,
    )
    param_display_options[selected_param]["scale"] = make_scale_slider(
        selected_param,
    )

    if param_display_options[selected_param]["dist"] == "foldnorm":
        param_display_options[selected_param]["c"] = 0

DEFAULT_PRIOR_INFO: dict = {
    'AGE': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'B': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'CHAS': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'CRIM': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'DIS': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'INDUS': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'LSTAT': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'NOX': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'PTRATIO': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'RAD': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'RM': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'TAX': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'ZN': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'intercept': {'dist': 'norm', 'loc': 0, 'scale': 1},
    'scale': {'c': 0, 'dist': 'foldnorm', 'loc': 10, 'scale': 1}}

PRIOR_INFO = deepcopy(DEFAULT_PRIOR_INFO)
PRIOR_INFO.update(param_display_options)

PERCENTILE = st.number_input(
    label = "What percentile of the predictive price * {}".format(COLUMN),
    min_value = 1,
    max_value = 99,
    value = 25,
)

NUM_PRIOR_SIM = st.number_input(
    label = "How many simulations should be used?",
    min_value = 50,
    max_value = 200,
    value = 100,
)

SEED = st.number_input(
    label = "What random seed should we set to ensure reproducibility?",
    min_value = 1,
    max_value = 5000,
    value = 129,
)

PLOT_PATH = str(pyprojroot.here("reports/figures/test_plot.png"))
SAVE = True

NOTEBOOK_PATH = str(pyprojroot.here(
    "notebooks/_01_tb_test_model_classes.ipynb"
))
OUTPUT_NOTEBOOK_PATH = str(pyprojroot.here(
    "notebooks/_01_tb_test_model_classes_output.ipynb"
))

# Run the notebook
if st.button("DRAW IT!"):
    st.text("Simulating!")
    pm.execute_notebook(
        NOTEBOOK_PATH,
        OUTPUT_NOTEBOOK_PATH,
        parameters=dict(
            MODEL = MODEL,
            COLUMN = COLUMN,
            PERCENTILE = PERCENTILE,
            PRIOR_INFO = PRIOR_INFO,
            NUM_PRIOR_SIM = NUM_PRIOR_SIM,
            SEED = SEED,
            PLOT_PATH = PLOT_PATH,
            SAVE = SAVE,
        )
    )
    st.text("Done Drawing!")

    # Load and display the created image
    plot_image = Image.open(PLOT_PATH)

    st.image(
        plot_image,
        caption="Prior Predictive Check: Percentile Plot",
        use_column_width = True,
    )
