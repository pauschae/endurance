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
import math
import os
import copy
import csv

# Create the "out" directory if it doesn't exist
output_dir = "out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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
    plt.savefig(os.path.join(output_dir, 'distribution_of_k_values.png'))
    plt.close()

    # Plot distribution of 'beta' values
    sns.histplot(params_individuals['beta'], kde=False, bins=10)
    plt.xlabel('beta values')
    plt.ylabel('Frequency')
    plt.title('Distribution of beta values')
    plt.savefig(os.path.join(output_dir, 'distribution_of_beta_values.png'))
    plt.close()

    # Plot distribution of 'alpha' values
    sns.histplot(params_questions['alpha1'], kde=False, bins=10)
    plt.xlabel('alpha values')
    plt.ylabel('Frequency')
    plt.title('Distribution of alpha values')
    plt.savefig(os.path.join(output_dir, 'distribution_of_alpha_values.png'))
    plt.close()

    # First plot: Probability to answer correctly over time
    sns.regplot(data=df_with_params, x="t", y="answered_correctly", lowess=True)
    plt.xlabel('Time (t)')
    plt.ylabel('Probability of Answering Correctly')
    plt.title('Probability to Answer Correctly Over Time')
    plt.savefig(os.path.join(output_dir, 'probability_to_answer_correctly_over_time.png'))
    plt.close()



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
    params_series = pd.Series(index=sequence, dtype=float)

    # Iterate over the sequence and calculate log-likelihood values
    for i in sequence:
        params.loc[column_name].value = original + i
        
        # Calculate contributions and log-likelihood value
        result = calc_marginal_likelihood(params, **my_kwargs)
        loglik_value = result["value"]
        loglik_series[i] = loglik_value
        params_series[i] = params.loc[column_name].value

        # Progress Bar
        percent_done = round(100*(i+window+step_size)/(2*window),1)
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
    plt.savefig(os.path.join(output_dir, f'likelihood_profile_{column_name}.png'))
    plt.close()

    return loglik_series

def mean_choice_right_answer(alpha_variance,
                             sigma_shape,
                             sigma_scale,
                             beta_shape,
                             beta_scale,
                             k_alpha,
                             k_beta,
                             t,
                             n_draws):
    """
    Calculate the average probability to answer a question correctly.
    Given the model parameters.
    """
    params = {
        "value": [alpha_variance, sigma_shape, sigma_scale,
                  beta_shape, beta_scale, k_alpha, k_beta],
        "lower_bound": [0, 0, 0, 0, 0, 0, 0]
    }

    index_names = ["alpha_variance", "sigma_shape", "sigma_scale",
                   "beta_shape", "beta_scale", "k_alpha", "k_beta"]
    
    params_df = pd.DataFrame(data=params, index=index_names)

    data_questions = {
    'question_id': [1],
    'correct_answer': [1],
    }
    
    data_questions_df = pd.DataFrame(data_questions)
    
    data_individuals = {
    'individual_id': [1],
    'question_id': [1],
    't': [t],
    'answer': [1]
    }
    
    
    data_individuals_df = pd.DataFrame(data_individuals)

    llh = calc_marginal_likelihood(params_df, data_questions_df, data_individuals_df, n_draws)

    out = math.exp(llh["value"])
    
    return out

def plot_parameter_effects(param_ranges_df, n_draws, t=0, output_dir = "out"):
    """
    Plot the effect of each parameter on the average probability to answer correctly,
    iterating over the range of each parameter while keeping others fixed.
    Write all parameter values and predicted probabilities to a CSV file.

    :param param_ranges_df: DataFrame with columns ['parameter', 'lowest', 'highest', 'step_size'].
    :param n_draws: Number of draws for the likelihood function.
    :param output_dir: Directory where the CSV and plot files will be saved.
    :param t: The fixed time value for which to evaluate the function.
    """
    # Initial fixed parameter values
    fixed_params_initial = {
        'alpha_variance': 1,
        'sigma_shape': 2,
        'sigma_scale': 1,
        'beta_shape': 2,
        'beta_scale': 1,
        'k_alpha': 0.5,
        'k_beta': 5
    }

    # Iterate over each parameter
    for _, row in param_ranges_df.iterrows():
        # Reset the parameter to its original fixed value after iteration
        fixed_params = copy.copy(fixed_params_initial)
        
        param_name = row['parameter']
        lowest = row['lowest']
        highest = row['highest']
        step_size = row['step_size']

        # Generate a range of values for the current parameter
        param_values = np.arange(lowest, highest + step_size, step_size)

        # List to store outputs
        outputs = []

        # Prepare CSV file
        csv_filename = os.path.join(output_dir, f'{param_name}_effects.csv')
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

            # Write the header with all parameter names and probability
            headers = list(fixed_params.keys()) + ['Probability']
            csv_writer.writerow(headers)

            # Iterate over the values for the current parameter
            for value in param_values:
                # Set the current parameter to the iterated value
                fixed_params[param_name] = value

                # Call the function with the current parameter value and fixed others
                result = mean_choice_right_answer(
                    fixed_params['alpha_variance'],
                    fixed_params['sigma_shape'],
                    fixed_params['sigma_scale'],
                    fixed_params['beta_shape'],
                    fixed_params['beta_scale'],
                    fixed_params['k_alpha'],
                    fixed_params['k_beta'],
                    t,
                    n_draws
                )
                print(f"{param_name} is set to {fixed_params[param_name]}, probability to answer correctly is {result}")

                # Collect the current state of all parameters and result
                row_data = list(fixed_params.values()) + [result]
                outputs.append(result)
                print(row_data)    
            
                # Write the row to the CSV
                csv_writer.writerow(row_data)

        plot_title = f'Effect of {param_name} on Probability to Answer Correctly t={t}'

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, outputs, marker='o', linestyle='-')
        plt.xlabel(param_name)
        plt.ylabel('Average Probability')
        plt.title(plot_title)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'effect_of_{param_name}_{t}.png'))
        plt.close()

        print(f"Plot: {plot_title} is done")
