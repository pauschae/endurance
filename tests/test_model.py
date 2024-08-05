# -*- coding: utf-8 -*-
"""
Test for Decisions.py

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

import numpy as np
import numpy.testing as npt
from model.decisions import (calculate_utilities,
                             calculate_remaining_endurance,
                             vectorize_data, transform_question_data,
                             scale_to_item_level,
                             calculate_choice_probabilities,
                             simulate_question_parameters,
                             simulate_individual_parameters)
import pytest 
import pandas as pd

def test_utilities():
    alpha_1 = np.array([1, 2, 3, 4])
    beta_1 = np.array([0.5, 0.5, 1, 1.5])
    sigma_1 = 1
    correct_1 = np.array([0, 0, 0, 1])
    remaining_endurance_1 = np.array([1, 1, 1, 1])
    utilities_expected_1 = np.array([1, 2, 3, 5.5])
    utilities_calculated_1 = calculate_utilities(correct_1, alpha_1, beta_1, sigma_1, remaining_endurance_1)
    npt.assert_array_equal(utilities_expected_1, utilities_calculated_1,
                       err_msg=f'utilities are {utilities_expected_1} but they should be {utilities_calculated_1}',
                       verbose=True, strict=False)
    
def test_calculate_remaining_endurance():
    """
    Test the calculate_remaining_endurance function.

    This test checks if the calculate_remaining_endurance function correctly models
    how endurance declines over time given the parameters t and cognitive_endurance.
    """

    # Test case 1: Basic functionality
    t = 6
    cognitive_endurance = 0.1
    expected = 0.5
    assert calculate_remaining_endurance(t, np.array([cognitive_endurance])) == pytest.approx(expected), \
        f"Expected {expected}, but got {calculate_remaining_endurance(t, cognitive_endurance)}"

    # Test case 2: Endurance does not go below zero
    t = 21
    cognitive_endurance = 0.1
    expected = 0
    assert calculate_remaining_endurance(t, cognitive_endurance) == pytest.approx(expected), \
        f"Expected {expected}, but got {calculate_remaining_endurance(t, cognitive_endurance)}"

    # Test case 3: Zero time, endurance should be 1
    t = 1
    cognitive_endurance = 0.1
    expected = 1
    assert calculate_remaining_endurance(t, cognitive_endurance) == pytest.approx(expected), \
        f"Expected {expected}, but got {calculate_remaining_endurance(t, cognitive_endurance)}"

    # Test case 4: High cognitive endurance rate, endurance drops quickly
    t = 2
    cognitive_endurance = 0.9
    expected = 0.1
    assert calculate_remaining_endurance(t, cognitive_endurance) == pytest.approx(expected), \
        f"Expected {expected}, but got {calculate_remaining_endurance(t, cognitive_endurance)}"

    # Test case 5: Negative cognitive endurance rate, endurance increases
    t = 6
    cognitive_endurance = -0.1
    expected = 1.5
    assert calculate_remaining_endurance(t, cognitive_endurance) == pytest.approx(expected), \
        f"Expected {expected}, but got {calculate_remaining_endurance(t, cognitive_endurance)}"

    print("All tests passed!")
    
def test_calculate_remaining_endurance_vector():
    """
    Test the calculate_remaining_endurance function with vector inputs.

    This test checks if the calculate_remaining_endurance function correctly models
    how endurance declines over time when passing vectors (arrays) of time 
    and cognitive endurance values.
    """

    # Define test cases
    t = np.array([1, 2, 3, 4, 5, 6])
    cognitive_endurance = np.array([0.1, 0.2, 0.3, 0.1, 0.05, 0.15])

    # Expected output
    expected = np.array([1, 0.8, 0.4, 0.7, 0.8, 0.25])
    
    # Calculate actual output
    actual = calculate_remaining_endurance(t, cognitive_endurance)

    # Test the results
    np.testing.assert_array_almost_equal(
        actual, expected, decimal=6,
        err_msg=f"Expected {expected}, but got {actual}"
    )

    print("All vector tests passed!")

def test_vectorize_data():
    # Example Datasets
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
    expected_corrects = np.array([1, 0, 0, 0,
                                  0, 0, 1, 0,
                                  1, 0, 0, 0,
                                  0, 0, 1, 0])
    
    expected_ks = np.array([0.1]*8 + [0.2]*8)
    
    expected_alphas = np.array([1, 1, 1, 1,
                                3, 5, 2, 4,
                                1, 1, 1, 1,
                                3, 5, 2, 4])
    
    expected_betas =  np.array([1]*8 + [3]*8)
    
    expected_remaining_endurances = np.array([1, 1, 1, 1,
                                    0.9, 0.9, 0.9, 0.9,
                                    0.8, 0.8, 0.8, 0.8,
                                    1, 1, 1, 1])
    
    expected_sigmas = np.array([1, 1, 1, 1,
                                5, 5, 5, 5,
                                1, 1, 1, 1,
                                5, 5, 5, 5,])
    
    expected_answers = np.array([1, 0, 0, 0,
                                 0, 0, 1, 0,
                                 0, 1, 0, 0,
                                 1, 0, 0, 0])
    
    expected_question_ids =  np.array(2*(4*[1]+4*[2]))
    expected_individual_ids = np.array(8*[1]+8*[2])
    expected_ts =  np.array(4*[1]+8*[2]+4*[1])


    # Call the function
    example_data_individuals = example_data_individuals.sort_values(by=['individual_id', 'question_id'], ascending=True)
    output_df = vectorize_data(example_params_individuals, example_params_questions,
                               example_data_questions, example_data_individuals)
    
    # Extract actual outputs
    actual_corrects = output_df['corrects'].to_numpy()
    actual_alphas = output_df['alphas'].to_numpy()
    actual_betas = output_df['betas'].to_numpy()
    actual_sigmas = output_df['sigmas'].to_numpy()
    actual_ts = output_df['t'].to_numpy()
    actual_ks = output_df['k'].to_numpy()
    actual_remaining_endurances = output_df['remaining_endurances'].to_numpy()
    actual_answers = output_df['answers'].to_numpy()
    actual_question_ids = output_df['question_id'].to_numpy()
    actual_individual_ids = output_df['individual_id'].to_numpy()



    # Test the results
    assert np.allclose(actual_ks, expected_ks), f"Expected {expected_ks}, but got {actual_ks}"
    assert np.allclose(actual_ts, expected_ts), f"Expected {expected_ts}, but got {actual_ts}"

    assert np.allclose(actual_corrects, expected_corrects), f"Expected {expected_corrects}, but got {actual_corrects}"
    assert np.allclose(actual_alphas, expected_alphas), f"Expected {expected_alphas}, but got {actual_alphas}"
    assert np.allclose(actual_betas, expected_betas), f"Expected {expected_betas}, but got {actual_betas}"
    assert np.allclose(actual_sigmas, expected_sigmas), f"Expected {expected_sigmas}, but got {actual_sigmas}"
    assert np.allclose(actual_remaining_endurances, expected_remaining_endurances), f"Expected {expected_remaining_endurances}, but got {actual_remaining_endurances}"
    assert np.allclose(actual_answers, expected_answers), f"Expected {expected_answers}, but got {actual_answers}"
    assert np.array_equal(actual_question_ids, expected_question_ids), f"Expected {expected_question_ids}, but got {actual_question_ids}"
    assert np.array_equal(actual_individual_ids, expected_individual_ids), f"Expected {expected_individual_ids}, but got {actual_individual_ids}"
    assert np.array_equal(actual_ts, expected_ts), f"Expected {expected_ts}, but got {actual_ts}"

    
def test_transform_question_data():
    # Test input data
    test_answers = np.array([1, 2, 3, 1, 3, 4])
    
    # Expected output
    expected_corrects = np.array([1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  1, 0, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1])
    
    # Call the function with test data
    actual_corrects = transform_question_data(test_answers)

    # Assert that the actual output matches the expected output
    np.testing.assert_array_equal(
        actual_corrects, expected_corrects, 
        err_msg=f"Expected {expected_corrects}, but got {actual_corrects}"
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
    corrects = np.array([1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0])
    ks = np.array([0.1] * 8 + [0.2] * 8)
    alphas = np.array([1, 1, 1, 1, 3, 5, 2, 4, 1, 1, 1, 1, 3, 5, 2, 4])
    betas = np.array([1] * 8 + [3] * 8)
    remaining_endurances = np.array([1, 1, 1, 1, 0.9, 0.9, 0.9, 0.9, 0.8, 0.8,
                                     0.8, 0.8, 1, 1, 1, 1])
    sigmas = np.array([1, 1, 1, 1, 5, 5, 5, 5, 1, 1, 1, 1, 5, 5, 5, 5])
    answers = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    question_ids = np.array(2 * (4 * [1] + 4 * [2]))
    individual_ids = np.array(8 * [1] + 8 * [2])
    ts = np.array(4 * [1] + 8 * [2] + 4 * [1])

    # Creating the DataFrame
    vectorized_df = pd.DataFrame({
        "corrects": corrects,
        "ks": ks,
        "alphas": alphas,
        "betas": betas,
        "remaining_endurances": remaining_endurances,
        "sigmas": sigmas,
        "answers": answers,
        "question_ids": question_ids,
        "individual_ids": individual_ids,
        "ts": ts,
        "answers": answers
    })
    
    # Expected choice probabilities
    expected_choice_probs = np.array([
        0.475366886, 0.174877705, 0.174877705, 0.174877705,
        0.022612775, 0.167087063, 0.748832266, 0.061467896,
        0.78606844, 0.07131052,  0.07131052, 0.07131052,
        8.31521E-07, 6.14416E-06, 0.999990764, 2.26031E-06
    ])
    
    # Calculate actual choice probabilities using the function
    actual_df = calculate_choice_probabilities(vectorized_df)
    actual_choice_probs = actual_df['choice_prob'].values

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


def simulate_and_check(simulation_func, params, expected_means, expected_vars, test_case_number, decimal=2):
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