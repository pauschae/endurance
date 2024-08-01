# -*- coding: utf-8 -*-
"""
Simulate data and test the estimation

@author: pis2
"""

import pandas as pd
import numpy as np
from model.decisions import simulate_dgp, calc_marginal_likelihood
import estimagic as em
from model.diagnostic_plots import plot_simulation_data, plot_profile_likelihood

# Configure Plotly.
import plotly.io as pio
pio.renderers.default = 'browser'

# Define Parameters
params = {
    "value": [0, 1, 2, 1, 2, 1, 0.5, 5],
    "lower_bound": [-np.inf, 0, 0, 0, 0, 0, 0, 0]
}

index_names = ["alpha_mean", "alpha_variance", "sigma_shape", "sigma_scale",
               "beta_shape", "beta_scale", "k_alpha", "k_beta"]

params = pd.DataFrame(data=params, index=index_names)

# Simulate the DGP
params_questions, params_individuals, simulated_data = simulate_dgp(params, 100, 500)

# Merge parameters with the simulated data
df_with_params = pd.merge(simulated_data, params_questions, on='question_id')
df_with_params = pd.merge(df_with_params, params_individuals, on='individual_id')
df_with_params["answered_correctly"] = df_with_params.answer == df_with_params.correct_answer

# Use the function to plot distributions and trends
plot_simulation_data(params_questions, params_individuals, df_with_params)

# Estimate the Model From Simulated Data
start_params = params.assign(value=8*[0.1])
my_kwargs = {"data_questions": params_questions,
             "data_individuals": simulated_data,
             "n_draws": 10000}

# Sort individual level data by question id and individual_id because the likelihood
# function expects this to be the case
data_individuals = simulated_data.sort_values(by=['question_id', 'individual_id'], ascending=True)

# Show Likelihood Profiles around the true value
for param_name in params.index:
    print(f"starting profile plot for {param_name}")
    plot_profile_likelihood(params, 'alpha_mean', my_kwargs)

res = em.estimate_ml(
    loglike=calc_marginal_likelihood,
    params=start_params,
    optimize_options={"algorithm": "scipy_lbfgsb"},
    loglike_kwargs=my_kwargs,
    logging="..\\out\\mylog.db"
)

res.summary().round(3)
fig = em.criterion_plot("mylog.db")
fig_params = em.params_plot("mylog.db")
fig.show()