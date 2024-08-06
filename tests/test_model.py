# -*- coding: utf-8 -*-
"""
Test for Decisions.py

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

import numpy as np
import numpy.testing as npt
from model.decisions import (calculate_utilities,
                             vectorize_data, transform_question_data,
                             scale_to_item_level,
                             calculate_choice_probabilities,
                             simulate_question_parameters,
                             simulate_individual_parameters)
#import pytest 
import pandas as pd

def test_utilities():
    alpha_1 = np.array([1, 2, 3, 4])
    beta_1 = np.array([0.5, 0.5, 1, 1.5])
    sigma_1 = 1
    correct_1 = np.array([0, 0, 0, 1])
    endurance_1 = np.array([0, 0, 0, 0.5])
    t_1 = np.array([1, 1, 1, 2])
    utilities_expected_1 = np.array([1, 2, 3, 4.75])
    utilities_calculated_1 = calculate_utilities(correct_1, alpha_1, beta_1, sigma_1, endurance_1, t_1)
    npt.assert_array_equal(utilities_expected_1, utilities_calculated_1,
                       err_msg=f'utilities are {utilities_expected_1} but they should be {utilities_calculated_1}',
                       verbose=True, strict=False)
        
def test_vectorize_data():
    # Example Dataset
    example_data_individuals = pd.DataFrame(
        data=[[1, 1, 1, 1], [1, 3, 2, 2], [2, 1, 1, 2], [2, 2, 2, 1]],
        columns=["individual_id", "answer", "t", "question_id"],
    )

    example_data_questions = pd.DataFrame(
        data=[[1, 1], [2, 3]],
        columns=["question_id", "correct_answer"],
    )

    example_params_individuals = pd.DataFrame(
        data=[[1, 1, 0.1], [2, 3, 0.2]],
        columns=["individual_id", "beta", "k"],
    )

    example_params_questions = pd.DataFrame(
        data=[[1, 1, 1, 1, 1, 1], [1, 3, 5, 2, 4, 5]],
        columns=["question_id", "alpha1", "alpha2", "alpha3", "alpha4", "sigma"],
    )

    # Expected output 
    expected_correct = np.array([1, 0, 0, 0,
                                  0, 0, 1, 0,
                                  1, 0, 0, 0,
                                  0, 0, 1, 0])
    
    expected_k = np.array([0.1]*8 + [0.2]*8)
    
    expected_alpha = np.array([1, 1, 1, 1,
                                3, 5, 2, 4,
                                1, 1, 1, 1,
                                3, 5, 2, 4])
    
    expected_beta =  np.array([1]*8 + [3]*8)
    
    expected_sigma = np.array([1, 1, 1, 1,
                                5, 5, 5, 5,
                                1, 1, 1, 1,
                                5, 5, 5, 5,])
    
    expected_answer = np.array([1, 0, 0, 0,
                                 0, 0, 1, 0,
                                 0, 1, 0, 0,
                                 1, 0, 0, 0])
    
    expected_question_id =  np.array(2*(4*[1]+4*[2]))
    expected_individual_id = np.array(8*[1]+8*[2])
    expected_t =  np.array(4*[1]+8*[2]+4*[1])


    # Call the function
    example_data_individuals = example_data_individuals.sort_values(by=['individual_id', 'question_id'], ascending=True)
    output_df = vectorize_data(example_params_individuals, example_params_questions,
                               example_data_questions, example_data_individuals)
    
    # Extract actual output
    actual_correct = output_df['correct']
    actual_alpha = output_df['alpha']
    actual_beta = output_df['beta']
    actual_sigma = output_df['sigma']
    actual_t = output_df['t']
    actual_k = output_df['k']
    actual_answer = output_df['answer']
    actual_question_id = output_df['question_id']
    actual_individual_id = output_df['individual_id']



    # Test the result
    assert np.allclose(actual_k, expected_k), f"Expected {expected_k}, but got {actual_k}"
    assert np.allclose(actual_t, expected_t), f"Expected {expected_t}, but got {actual_t}"

    assert np.allclose(actual_correct, expected_correct), f"Expected {expected_correct}, but got {actual_correct}"
    assert np.allclose(actual_alpha, expected_alpha), f"Expected {expected_alpha}, but got {actual_alpha}"
    assert np.allclose(actual_beta, expected_beta), f"Expected {expected_beta}, but got {actual_beta}"
    assert np.allclose(actual_sigma, expected_sigma), f"Expected {expected_sigma}, but got {actual_sigma}"
    assert np.allclose(actual_answer, expected_answer), f"Expected {expected_answer}, but got {actual_answer}"
    assert np.array_equal(actual_question_id, expected_question_id), f"Expected {expected_question_id}, but got {actual_question_id}"
    assert np.array_equal(actual_individual_id, expected_individual_id), f"Expected {expected_individual_id}, but got {actual_individual_id}"
    assert np.array_equal(actual_t, expected_t), f"Expected {expected_t}, but got {actual_t}"

    
def test_transform_question_data():
    # Test input data
    test_answer = np.array([1, 2, 3, 1, 3, 4])
    
    # Expected output
    expected_correct = np.array([1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  1, 0, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1])
    
    # Call the function with test data
    actual_correct = transform_question_data(test_answer)

    # Assert that the actual output matches the expected output
    np.testing.assert_array_equal(
        actual_correct, expected_correct, 
        err_msg=f"Expected {expected_correct}, but got {actual_correct}"
    )

    
def test_scale_to_item_level():
    n_questions = 2
    data = np.array([1, 2])
    expected_output = np.array([1, 1, 1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2, 2, 2])
    actual_output = scale_to_item_level(n_questions, data)
    np.testing.assert_array_equal(
        actual_output, expected_output, 
        err_msg=f"Expected {expected_output}, but got {actual_output}"
    )
    
def test_calculate_choice_probabilities():
    # Example Vectorized data
    # Variables
    correct = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    k = np.array([0.1] * 8 + [0.2] * 8)
    alpha = np.array([1, 1, 1, 1, 3, 5, 2, 4, 1, 1, 1, 1, 3, 5, 2, 4])
    beta = np.array([1] * 8 + [3] * 8)
    sigma = np.array([1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5])
    answer = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    question_id = np.array(2 * (4 * [1] + 4 * [2]))
    individual_id = np.array(8 * [1] + 8 * [2])
    t = np.array(4 * [1] + 8 * [2] + 4 * [1])

    # Creating the DataFrame
    vectorized_dict = {
        "correct": correct,
        "k": k,
        "alpha": alpha,
        "beta": beta,
        "sigma": sigma,
        "answer": answer,
        "question_id": question_id,
        "individual_id": individual_id,
        "t": t,
        "answer": answer
    }
    
    #vectorized_df = vectorized_df.sort_values(by=['question_id', 'individual_id'], ascending=True)

    # Expected choice probabilities
    expected_choice_probs = np.array([
        0.475366886, 0.174877705, 0.174877705, 0.174877705,
        0.022612775, 0.167087063, 0.748832266, 0.061467896,
        0.78606844, 0.07131052,  0.07131052, 0.07131052,
        8.31521E-07, 6.14416E-06, 0.999990764, 2.26031E-06
    ])
    expected_choice_probs = expected_choice_probs[answer]
    
    # Calculate actual choice probabilities using the function
    actual_choice_probs = calculate_choice_probabilities(vectorized_dict)

    # Assert that the actual choice probabilities match the expected choice probabilities
    npt.assert_array_almost_equal(
        actual_choice_probs, expected_choice_probs, decimal=8,
        err_msg=f"Expected {expected_choice_probs}, but got {actual_choice_probs}"
    )

    print("test_calculate_choice_probabilities passed!")
    

def calculate_expected_means_question(alpha_cols, sigma_shape, sigma_scale):
    """
    Calculate expected means to test simulate_question_parameters.
    
    Parameters:
    - alpha_cols: Number of alpha columns.
    - sigma_shape: Shape parameter for the sigma distribution.
    - sigma_scale: Scale parameter for the sigma distribution.
    
    Returns:
    - pd.Series of expected means.
    """
    return pd.Series([0] * alpha_cols + [sigma_scale * sigma_shape])

def calculate_expected_vars_question(alpha_variance, alpha_cols, sigma_shape, sigma_scale):
    """
    Calculate expected variances to test simulate_question_parameters.
    
    Parameters:
    - alpha_variance: Variance for the alpha distribution.
    - alpha_cols: Number of alpha columns.
    - sigma_shape: Shape parameter for the sigma distribution.
    - sigma_scale: Scale parameter for the sigma distribution.
    
    Returns:
    - pd.Series of expected variances.
    """
    return pd.Series([alpha_variance] * alpha_cols + [(sigma_scale ** 2) * sigma_shape])


def calculate_expected_means_individuals(beta_shape, beta_scale, k_alpha, k_beta):
    """
    Calculate expected means test simulate_individual_parameters

    Parameters:
    - beta_shape: Shape parameter for the beta distribution.
    - beta_scale: Scale parameter for the beta distribution.
    - k_alpha: Shape parameter for the k distribution.
    - k_beta: Scale parameter for the k distribution.

    Returns:
    - pd.Series of expected means.
    """
    beta_mean = beta_shape * beta_scale
    k_mean = k_alpha / (k_alpha + k_beta)
    return pd.Series([beta_mean, k_mean], index=["beta", "k"])

def calculate_expected_vars_individuals(beta_shape, beta_scale, k_alpha, k_beta):
    """
    Calculate expected variances to test simulate_individual_parameters


    Parameters:
    - beta_shape: Shape parameter for the beta distribution.
    - beta_scale: Scale parameter for the beta distribution.
    - k_alpha: Shape parameter for the k distribution.
    - k_beta: Scale parameter for the k distribution.

    Returns:
    - pd.Series of expected variances.
    """
    beta_var = beta_shape * (beta_scale ** 2)
    k_var = (k_alpha * k_beta) / (((k_alpha + k_beta) ** 2) * (k_alpha + k_beta + 1))
    return pd.Series([beta_var, k_var], index=["beta", "k"])


def simulate_and_check(simulation_func, params, expected_means, expected_vars, test_case_number, decimal=1):
    """
    Simulate parameters using the specified function and check if the means and variances match expected values.
    
    Parameters:
    - simulation_func: The function to simulate parameters. It should return data and column names.
    - params: Dictionary of parameters for the simulation.
    - expected_means: Series of expected means.
    - expected_vars: Series of expected variances.
    - test_case_number: Identifier for the test case.
    - decimal: Decimal precision for the comparison.
    """
    # Execute the simulation function with the provided parameters
    draw_data, draw_columns = simulation_func(**params)
    
    # Convert the result into a DataFrame
    actual_simulation = pd.DataFrame(draw_data, columns=draw_columns)
    
    # Calculate the means and variances
    actual_means = actual_simulation.mean().iloc[1:]
    actual_variances = actual_simulation.var().iloc[1:]

    # Assert the means and variances against the expected values
    npt.assert_array_almost_equal(
        actual_means, expected_means, decimal=decimal,
        err_msg=f"Expected means {', '.join(expected_means.map(str))} but got {', '.join(actual_means.map(str))}, test case {test_case_number}"
    )

    npt.assert_array_almost_equal(
        actual_variances, expected_vars, decimal=decimal,
        err_msg=f"Expected variances {', '.join(expected_vars.map(str))} but got {', '.join(actual_variances.map(str))}, test case {test_case_number}"
    )


def test_simulate_question_parameters():
    n = 10**7
    
    # Test Case 1
    args1 = {
        "n": n,
        "alpha_variance": 1,
        "alpha_cols": 4,
        "sigma_shape": 2,
        "sigma_scale": 1
    }
    expected_means_1 = calculate_expected_means_question(args1["alpha_cols"], args1["sigma_shape"], args1["sigma_scale"])
    expected_vars_1 = calculate_expected_vars_question(args1["alpha_variance"], args1["alpha_cols"], args1["sigma_shape"], args1["sigma_scale"])
    simulate_and_check(simulate_question_parameters, args1, expected_means_1, expected_vars_1, "1")

    # Test Case 2
    args2 = {
        "n": n,
        "alpha_variance": 2,
        "alpha_cols": 3,
        "sigma_shape": 5,
        "sigma_scale": 2
    }
    expected_means_2 = calculate_expected_means_question(args2["alpha_cols"], args2["sigma_shape"], args2["sigma_scale"])
    expected_vars_2 = calculate_expected_vars_question(args2["alpha_variance"], args2["alpha_cols"], args2["sigma_shape"], args2["sigma_scale"])
    simulate_and_check(simulate_question_parameters, args2, expected_means_2, expected_vars_2, "2")

    # Test Case 3
    args3 = {
        "n": n,
        "alpha_variance": 0.5,
        "alpha_cols": 5,
        "sigma_shape": 3,
        "sigma_scale": 1.5
    }
    expected_means_3 = calculate_expected_means_question(args3["alpha_cols"], args3["sigma_shape"], args3["sigma_scale"])
    expected_vars_3 = calculate_expected_vars_question(args3["alpha_variance"], args3["alpha_cols"], args3["sigma_shape"], args3["sigma_scale"])
    simulate_and_check(simulate_question_parameters, args3, expected_means_3, expected_vars_3, "3")

    # Test Case 4
    args4 = {
        "n": n,
        "alpha_variance": 3,
        "alpha_cols": 2,
        "sigma_shape": 4,
        "sigma_scale": 0.5
    }
    expected_means_4 = calculate_expected_means_question(args4["alpha_cols"], args4["sigma_shape"], args4["sigma_scale"])
    expected_vars_4 = calculate_expected_vars_question(args4["alpha_variance"], args4["alpha_cols"], args4["sigma_shape"], args4["sigma_scale"])
    simulate_and_check(simulate_question_parameters, args4, expected_means_4, expected_vars_4, "4")

    print("All test cases for simulate_question_parameters passed!")
    
    
def test_simulate_individual_parameters():
    n = 10**7
    
    # Test Case 1
    args1 = {
        "n": n,
        "beta_shape": 2,
        "beta_scale": 1,
        "k_alpha": 2,
        "k_beta": 1
    }
    expected_means_1 = calculate_expected_means_individuals(args1["beta_shape"], args1["beta_scale"], args1["k_alpha"], args1["k_beta"])
    expected_vars_1 = calculate_expected_vars_individuals(args1["beta_shape"], args1["beta_scale"], args1["k_alpha"], args1["k_beta"])
    simulate_and_check(simulate_individual_parameters, args1, expected_means_1, expected_vars_1, "1")

    # Test Case 2
    args2 = {
        "n": n,
        "beta_shape": 5,
        "beta_scale": 2,
        "k_alpha": 3,
        "k_beta": 2
    }
    expected_means_2 = calculate_expected_means_individuals(args2["beta_shape"], args2["beta_scale"], args2["k_alpha"], args2["k_beta"])
    expected_vars_2 = calculate_expected_vars_individuals(args2["beta_shape"], args2["beta_scale"], args2["k_alpha"], args2["k_beta"])
    simulate_and_check(simulate_individual_parameters, args2, expected_means_2, expected_vars_2, "2")

    # Test Case 3
    args3 = {
        "n": n,
        "beta_shape": 1,
        "beta_scale": 3,
        "k_alpha": 4,
        "k_beta": 3
    }
    expected_means_3 = calculate_expected_means_individuals(args3["beta_shape"], args3["beta_scale"], args3["k_alpha"], args3["k_beta"])
    expected_vars_3 = calculate_expected_vars_individuals(args3["beta_shape"], args3["beta_scale"], args3["k_alpha"], args3["k_beta"])
    simulate_and_check(simulate_individual_parameters, args3, expected_means_3, expected_vars_3, "3")

    # Test Case 4
    args4 = {
        "n": n,
        "beta_shape": 3,
        "beta_scale": 0.5,
        "k_alpha": 5,
        "k_beta": 1
    }
    expected_means_4 = calculate_expected_means_individuals(args4["beta_shape"], args4["beta_scale"], args4["k_alpha"], args4["k_beta"])
    expected_vars_4 = calculate_expected_vars_individuals(args4["beta_shape"], args4["beta_scale"], args4["k_alpha"], args4["k_beta"])
    simulate_and_check(simulate_individual_parameters, args4, expected_means_4, expected_vars_4, "4")

    print("All test cases for simulate_individual_parameters passed!")