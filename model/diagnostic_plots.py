# -*- coding: utf-8 -*-
"""
Functions to visualize and diagnose problems with the likelihood and data 
generating process.
@author: pis2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model.decisions import calc_marginal_likelihood

def plot_simulation_data(params_questions, params_individuals, df_with_params):
    """
    Plot distributions of parameters and trends over time from simulated data.

    :param params_questions: DataFrame containing question-level parameters.
    :param params_individuals: DataFrame containing individual-level parameters.
    :param df_with_params: DataFrame containing merged data with parameter information.
    """

    # Plot distribution of 'k' values
    sns.histplot(params_individuals['k'], kde=False, bins=10)
    plt.xlabel('k values')
    plt.ylabel('Frequency')
    plt.title('Distribution of k values')
    plt.show()

    # Plot distribution of 'beta' values
    sns.histplot(params_individuals['beta'], kde=False, bins=10)
    plt.xlabel('beta values')
    plt.ylabel('Frequency')
    plt.title('Distribution of beta values')
    plt.show()

    # Plot distribution of 'alpha' values
    sns.histplot(params_questions['alpha1'], kde=False, bins=10)
    plt.xlabel('alpha values')
    plt.ylabel('Frequency')
    plt.title('Distribution of alpha values')
    plt.show()

    # First plot: Probability to answer correctly over time
    sns.regplot(data=df_with_params, x="t", y="answered_correctly", lowess=True)
    plt.xlabel('Time (t)')
    plt.ylabel('Probability of Answering Correctly')
    plt.title('Probability to Answer Correctly Over Time')
    plt.show()

    # Second plot: Remaining endurance over time
    sns.regplot(data=df_with_params, x="t", y="remaining_endurance", x_bins=np.arange(0, 100, 10), order=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Remaining Endurance')
    plt.title('Remaining Endurance Over Time')
    plt.show()


def plot_profile_likelihood(params, column_name, my_kwargs, step_size=0.2, window=1):
    """
    Plot the Shape of the Likelihood Around some Parameter Values.

    params: DataFrame containing parameter values.
    column_name: String name of the column in params to modify. This function
        plots the shape of the likelihood from original parameter value-window to
        original parameter value + window, holding all other parameters constant.
        
    my_kwargs: Dictionary with additional keyword arguments for the likelihood function.
        Includes the dataset.
        
    step_size: Step size for the sequence from -window to window.
    side effect: Plot with parameter values on the x-axis and log-likelihood values
        on the y-axis.
    """
    # Store the original value of the specified parameter
    original = params.loc[column_name].value

    # Create a sequence from -window to window with the specified step size
    sequence = np.arange(-window, window + step_size, step_size)

    # Initialize a Pandas Series to store log-likelihood values
    loglik_series = pd.Series(index=sequence, dtype=float)

    # Iterate over the sequence and calculate log-likelihood values
    for i in sequence:
        params.loc[column_name].value = original + i
        
        # Calculate contributions and log-likelihood value
        result = calc_marginal_likelihood(params, **my_kwargs)
        loglik_value = result["value"]
        loglik_series[i] = loglik_value
        
        #Progress Bar
        percent_done = round(100*(i+window)/(2*window),1)
        print(f"{percent_done} % done for {column_name}")

    # Reset the parameter to its original value
    params.loc[column_name].value = original

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(loglik_series.index + original, loglik_series.values, marker='o', linestyle='-')
    plt.xlabel(f'{column_name} values')
    plt.ylabel('Log-Likelihood')
    plt.title(f'Likelihood Profile for {column_name}')
    plt.grid(True)
    plt.show()

    return loglik_series