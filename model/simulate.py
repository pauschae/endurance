# -*- coding: utf-8 -*-
"""
Simulate data and test the estimation

@author: pis2
"""
import pandas as pd
from model.decisions import simulate_dgp, calc_marginal_likelihood
import estimagic as em
from model.diagnostic_plots import (plot_simulation_data,
                                    plot_profile_likelihood,
                                    plot_parameter_effects)
from datetime import datetime



# Configure Plotly.
import plotly.io as pio
pio.renderers.default = 'browser'

def main():
    # Define Parameters
    params = {
        "value": [1, 2, 1, 2, 1, 0.5, 5],
        "lower_bound": [0, 0, 0, 0, 0, 0, 0]
    }

    index_names = ["alpha_variance", "sigma_shape", "sigma_scale",
                   "beta_shape", "beta_scale", "k_alpha", "k_beta"]

    params = pd.DataFrame(data=params, index=index_names)

    # Plot Choice Probabilities
    # Example DataFrame with parameter ranges
    param_ranges_df = pd.DataFrame({
        'parameter': ['alpha_variance', 'sigma_shape', 'sigma_scale', 'beta_shape', 'beta_scale', 'k_alpha', 'k_beta'],
        'lowest': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        'highest': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'step_size': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    })

    # Call the function with the parameter ranges and number of draws
    plot_parameter_effects(param_ranges_df, n_draws=10000, t=0)

    # Simulate the DGP
    params_questions, params_individuals, simulated_data = simulate_dgp(params, 100, 100)

    # Merge parameters with the simulated data
    df_with_params = pd.merge(simulated_data, params_questions, on='question_id')
    df_with_params = pd.merge(df_with_params, params_individuals, on='individual_id')
    df_with_params["answered_correctly"] = df_with_params.answer == df_with_params.correct_answer

    # Use the function to plot distributions and trends
    plot_simulation_data(params_questions, params_individuals, df_with_params)

    # Sort individual level data by question id and individual_id because the likelihood
    # function expects this to be the case
    simulated_data = simulated_data.sort_values(by=['question_id', 'individual_id'], ascending=True)


    # Estimate the Model From Simulated Data
    start_params = params.assign(value=7*[0.1])
    my_kwargs = {"data_questions": params_questions,
                 "data_individuals": simulated_data,
                 "n_draws": 10000
                 }

    calc_marginal_likelihood(params_df=params, **my_kwargs)


    # Show Likelihood Profiles around the true value
    for param_name in params.index:
        print(f"starting profile plot for {param_name}")
        plot_profile_likelihood(params, param_name, my_kwargs)
    
    # Get the current date and time
    now = datetime.now()

    # Format the date and time into a short string suitable for filenames
    # Example format: YYYYMMDD_HHMMSS
    filename_date_time = now.strftime("%Y%m%d_%H%M%S")

    res = em.estimate_ml(
        loglike=calc_marginal_likelihood,
        params=start_params,
        optimize_options={"algorithm": "scipy_lbfgsb"},
        loglike_kwargs=my_kwargs,
        logging=f"..\\out\\mylog_{filename_date_time}.db"
    )

    res.summary().round(3)
    fig = em.criterion_plot("mylog.db")
    fig_params = em.params_plot("mylog.db")
    fig.show()
    fig_params.show()

if __name__ == '__main__':
    main()
