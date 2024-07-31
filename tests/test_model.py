# -*- coding: utf-8 -*-
"""
Test for Decisions.py

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

import numpy as np
import numpy.testing as npt
from model.decisions import calculate_utilities, calculate_remaining_endurance, vectorize_data
import pytest 
import pandas as pd

def test_utilities():
    alpha_1 = np.array([1, 2, 3, 4])
    beta_1 = np.array([0.5, 0.5, 1, 1.5])
    sigma_1 = 1
    correct_1 = np.array([0, 0, 0, 1])
    utilities_desired_1 = np.array([1, 2, 3, 5.5])
    utilities_calculated_1 = calculate_utilities(correct_1, alpha_1, beta_1, sigma_1)
    npt.assert_array_equal(utilities_desired_1, utilities_calculated_1,
                       err_msg=f'utilities are {utilities_desired_1} but they should be {utilities_calculated_1}',
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
    
    expected_endurances = np.array([1, 1, 1, 1,
                                    0.9, 0.9, 0.9, 0.9,
                                    0.8, 0.8, 0.8, 0.8,
                                    1, 1, 1, 1])
    
    expected_sigmas = np.array([1, 1, 1, 1,
                                5, 5, 5, 5,
                                1, 1, 1, 1,
                                5, 5, 5, 5,])
    
    expected_deltas = expected_sigmas*expected_endurances
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
    actual_deltas = output_df['deltas'].to_numpy()
    actual_sigmas = output_df['sigmas'].to_numpy()
    actual_ts = output_df['t'].to_numpy()
    actual_ks = output_df['k'].to_numpy()
    actual_endurances = output_df['endurances'].to_numpy()
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
    assert np.allclose(actual_endurances, expected_endurances), f"Expected {expected_endurances}, but got {actual_endurances}"
    assert np.allclose(actual_deltas, expected_deltas), f"Expected {expected_deltas}, but got {actual_deltas}"
    assert np.allclose(actual_answers, expected_answers), f"Expected {expected_answers}, but got {actual_answers}"
    assert np.array_equal(actual_question_ids, expected_question_ids), f"Expected {expected_question_ids}, but got {actual_question_ids}"
    assert np.array_equal(actual_individual_ids, expected_individual_ids), f"Expected {expected_individual_ids}, but got {actual_individual_ids}"
    assert np.array_equal(actual_ts, expected_ts), f"Expected {expected_ts}, but got {actual_ts}"

    print("All tests passed!")