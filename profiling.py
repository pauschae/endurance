# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:29:53 2024

This script profiles the performance of two functions from the `model.decisions` module:
1. `calc_marginal_likelihood`: Calculates the marginal likelihood given certain parameters.
2. `vectorize_data`: Vectorizes data for further processing.

The profiling results are saved in `.prof` files and summarized in `.txt` files. 
Top 10 functions by cumulative time are printed to the console for each profiled function.

Author: pis2
"""

#  4.647 sec

import cProfile
import pstats
from model.decisions import calc_marginal_likelihood, simulate_dgp, vectorize_data
import pandas as pd

# Define the parameters
params = {
    "value": [1, 2, 1, 2, 1, 0.5, 5],
    "lower_bound": [0, 0, 0, 0, 0, 0, 0]
}

index_names = ["alpha_variance", "sigma_shape", "sigma_scale", 
               "beta_shape", "beta_scale", "k_alpha", "k_beta"]

params = pd.DataFrame(data=params, index=index_names)

# Simulate data
params_questions, params_individuals, simulated_data = simulate_dgp(params, n_questions = 100, n_individuals = 1000)
# Sort individual level data by question id to facilitate question level analysis
data_individuals = simulated_data.sort_values(by=['question_id', 'individual_id'], ascending=True)
  
# Starting parameters
start_params = params.assign(value=7*[0.1])
my_kwargs = {
    "data_questions": params_questions,
    "data_individuals": simulated_data,
    "n_draws": 1000
}

def profile_function(func, func_name, *args, **kwargs):
    """
    Profiles a given function and saves the profiling results.

    Parameters:
    func (callable): The function to profile.
    func_name (str): The name of the function, used for naming the result files.
    *args: Arguments to pass to the function.
    **kwargs: Keyword arguments to pass to the function.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    func(*args, **kwargs)
    profiler.disable()
    
    # Save profiling results
    profile_file = f'profile_results_{func_name}.prof'
    profiler.dump_stats(profile_file)
    
    # Create a Stats object and sort by cumulative time
    with open(f'profile_result_{func_name}.txt', 'w') as f:
        stats = pstats.Stats(profile_file, stream=f)
        stats.strip_dirs().sort_stats('cumulative').print_stats(10)
    
    # Print profiling results to console
    stats = pstats.Stats(profile_file)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)

# Profile the calc_marginal_likelihood function
#44.202
profile_function(calc_marginal_likelihood, 'calc_marginal_likelihood',
                 start_params, **my_kwargs)

# Profile the vectorize_data function
# Adjusting the call to vectorize_data with the correct parameters
profile_function(vectorize_data, 'vectorize_data',
                 params_individuals,
                 params_questions,
                 params_questions,
                 simulated_data)
