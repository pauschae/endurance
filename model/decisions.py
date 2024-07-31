    # -*- coding: utf-8 -*-
"""
Decisions

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

import numpy as np
from scipy.special import softmax
import pandas as pd
from numba import jit, int64
from joblib import Parallel, delayed
import multiprocessing as mp


@jit(nopython=True)
def calculate_utilities(correct, alpha, beta, sigma):
    """
    Calculates utilities as a function of item parameters and remaining
    endurance.

    Parameters
    ----------
    correct : numpy array, shape (4,)
        Vector that contains a 1 for the correct answer
        and zeros otherwise.
    alpha : numpy array, shape (4,)
        Baseline propensity to answer with option i.
    beta : numpy array, shape (4,)
        Ability: for a well-rested individual 
        this gets added to the utility of the correct answer.
    sigma : numpy array, shape (4,)
        Utility difference between correct and incorrect answers.
        Declines over time to model cognitive exhaustion.

    Returns
    -------
    numpy array, shape (4,)
        A numeric vector with the utilities of each answer option.
    """
    utilities = alpha + correct * beta * sigma
    return utilities

@jit(nopython=True)
def calculate_remaining_endurance(t, cognitive_endurance):
    """
    Models how endurance declines over time.

    Parameters
    ----------
    t : int
        Time, how many questions have passed.
        start at 1
    cognitive_endurance : float
        Number that describes how fast endurance declines.

    Returns
    -------
    float
        The endurance level as a numeric scalar.
    """
        
    endurance = np.maximum(1 - cognitive_endurance * (t-1), 0)
    return endurance

@jit(nopython=True)
def transform_question_data(answers):
    """
    Transforms question data for vectorized calculations.

    Parameters
    ----------
    answers : pandas Series of correct answers by question.

    Returns
    -------
    numpy array
        The data transformed for vectorized calculations.
    """
    nrows = len(answers)
    # Create a zero matrix of shape (len(answers), 4)
    corrects = np.zeros((nrows, 4), dtype=int64)
    
    # Use numpy advanced indexing to set the correct positions to 1
    for i in range(nrows):
        corrects[i, answers[i] - 1] = 1
    
    out = corrects.flatten()    
    return out

@jit(nopython=True)
def scale_to_item_level(n_questions, data):
    """
    Transforms individual data to the scale of question items.

    Parameters
    ----------
    n_questions : int
        Number of questions.
    data : Series
        A numpy array with individual-level data.

    Returns
    -------
    numpy array
        A data series on question level that can be used for vectorized analysis.
    """
    scaled = np.repeat(data, n_questions * 4)
    return scaled


def vectorize_data(params_individuals, params_questions, data_questions, data_individuals):
    """
    Vectorizes the input data for a given set of individuals and questions.

    The function scales individual parameters to the item level, tiles the question
    parameters across individuals, computes endurances, transforms question data,
    and finally concatenates the results into a DataFrame.

    Parameters
    ----------
    params_individuals : DataFrame
        Sorted DataFrame by question_id and individual_id  containing individual-level parameters.
        - individual_id : int
            Unique identifier for each individual.
        - beta : float
            Individual ability parameters.
        - k : float
            Individual endurance parameters.

    params_questions : DataFrame
        DataFrame containing question-level parameters.
        - question_id : int
            Unique identifier for each question.
        - alpha1 : float
            Parameter alpha1 for each question. Measures propensity to choose
            option 1.
        - alpha2 : float
            Parameter alpha2 for each question.
        - alpha3 : float
            Parameter alpha3 for each question.
        - alpha4 : float
            Parameter alpha4 for each question.
        - sigma : float
            Measures the sensitivity of propensity for the correct answer to
            endurance-scaled ability.

    data_questions : DataFrame
        DataFrame containing the correct answers for the questions.
        - question_id : int
            Unique identifier for each question.
        - correct_answer : int
            The correct answer for each question.

    data_individuals : DataFrame
        DataFrame containing data related to individuals.
        - individual_id : int
            Unique identifier for each individual.
        - answer : int
            The answer provided by the individual.
        - t : int
            Time index.
        - question_id : int
            Question ID.

    Returns
    -------
    output_df : DataFrame
        A DataFrame containing the vectorized data (1 row per question answer),
        including the following columns:
        - corrects : 1 if the answer is correct.
        - alphas : Combined alpha parameters for each question.
        - betas : Scaled individual difficulty parameters.
        - deltas : Product of sigmas and endurances.
        - sigmas : Question sensitivty parameters.
        - endurances: endurance at current time t
        - answers: dummy indicating if this is the correct answer
        - The resulting dataset also includes the question_id, individual_id and t
        
        The dataset is sorted by individual ids and within individual by 
        question id.        
        
    """
    
    n_questions = len(params_questions)
    n_individuals = len(data_individuals) // n_questions
      
    # Individual Level 
    betas = scale_to_item_level(n_questions, params_individuals["beta"].to_numpy())
    ks = scale_to_item_level(n_questions, params_individuals["k"].to_numpy())
    answers = transform_question_data(data_individuals['answer'].to_numpy())

    # Question Level    
    alphas = params_questions[["alpha1", "alpha2", "alpha3", "alpha4"]].to_numpy().flatten()
    alphas = np.tile(alphas, n_individuals)
    sigmas = np.tile(scale_to_item_level(1, params_questions["sigma"].to_numpy()), n_individuals)

    # Time Varying
    ts = scale_to_item_level(1, data_individuals['t'].to_numpy())
    endurances = calculate_remaining_endurance(ts, ks)
    corrects = transform_question_data(data_questions['correct_answer'].to_numpy())
    corrects = np.tile(corrects, n_individuals)
    
    deltas = sigmas * endurances
    
    #Set ids
    #question_id = np.concatenate([4*[x] for x in data_individuals['question_id']])
    #individual_id = np.concatenate([4*[x] for x in data_individuals['individual_id']])
    question_id = np.repeat(data_individuals['question_id'], 4)
    individual_id = np.repeat(data_individuals['individual_id'], 4)


    output_df = pd.DataFrame({
        'corrects': corrects,
        'alphas': alphas,
        'betas': betas,
        'deltas': deltas,
        'sigmas': sigmas,
        'endurances': endurances,
        'answers': answers,
        't': ts,
        'k': ks,
        'question_id': question_id,
        'individual_id': individual_id
    })

    return output_df

def calculate_choice_probabilities(vectorized_df):
    """
    Calculate choice probabilities and filter DataFrame based on answers.

    This function computes the utility scores using the `utilities` function and stores them
    in the 'res_u' column of the input DataFrame. It then calculates the choice probabilities 
    using the softmax function on these utility scores, storing the result in the 'choice_prob'
    column. Finally, it filters the DataFrame to include only rows where the 'answers' column 
    is equal to 1.

    Parameters:
    vectorized_df (pd.DataFrame): A DataFrame containing the following columns:
        - 'corrects': (int) The number of the correct answer.
        - 'alphas': (float) Parameter alpha for the utility calculation.
        - 'betas': (float) Parameter beta for the utility calculation.
        - 'sigmas': (float) Parameter sigma for the utility calculation.
        - 'individual_id': (int) Identifier for individuals.
        - 'question_id': (int) Identifier for questions.
        - 'answers': (int) The number of the answer given by the test taker.

    Returns:
    pd.DataFrame: A filtered DataFrame containing only rows where the 'answers' column is 1. 
                  The DataFrame includes two additional columns:
        - 'res_u': The calculated utility scores.
        - 'choice_prob': The calculated choice probabilities using the softmax function.
    """
    vectorized_df['res_u'] = calculate_utilities(vectorized_df['corrects'].to_numpy(),
                                                 vectorized_df['alphas'].to_numpy(), 
                                                 vectorized_df['betas'].to_numpy(),
                                                 vectorized_df['sigmas'].to_numpy())
    
    
    #Recast The Utilities to and use the softmax in a vectorized way
    u_array = vectorized_df['res_u'].values.reshape((-1, 4))
    softmax_array = softmax(u_array, axis=1)
    flattened_softmax_array = softmax_array.flatten()
    softmax_series = pd.Series(flattened_softmax_array)


    vectorized_df['choice_prob'] = softmax_series
    return vectorized_df

@jit(nopython=True)
def simulate_question_parameters(n, alpha_mean=0, alpha_variance=0,
                                 alpha_cols=4, sigma_shape=2, sigma_scale=1):
    """
    Simulate a dataset with n rows.

    Parameters:
    n (int): Number of rows to simulate.
    m (float): Mean of the normal distribution for alphas.
    v (float): Variance of the normal distribution for alphas.
    alpha_cols (int): Number of alpha columns to generate.
    sigma_shape (float): Shape parameter for the Gamma distribution for sigma.
    sigma_scale (float): Scale parameter for the Gamma distribution for sigma.

    Returns:
    pd.DataFrame: Simulated DataFrame with columns "question_id", "alpha1", ..., "alphaN", "sigma".
    """
    # Generate question_id from 1 to n
    question_id = np.arange(1, n + 1)

    # Generate alphas from a normal distribution with mean m and variance v
    alphas = np.random.normal(alpha_mean, np.sqrt(alpha_variance), (n, alpha_cols))

    # Generate sigma from a Gamma distribution with specified shape and scale
    sigma = np.random.gamma(sigma_shape, sigma_scale, n)

    # Combine all columns into a DataFrame
    data = np.column_stack((question_id, alphas, sigma))
    columns = ["question_id"] + [f"alpha{i+1}" for i in range(alpha_cols)] + ["sigma"]

    return (data, columns)

@jit(nopython=True)
def simulate_individual_parameters(n, beta_shape=2, beta_scale=1, k_alpha=2, k_beta=1):
    """
    Simulate individual data with n rows.

    Parameters:
    n (int): Number of rows to simulate.
    beta_shape (float): Shape parameter for the Gamma distribution for beta.
    beta_scale (float): Scale parameter for the Gamma distribution for beta.
    k_alpha (float): Shape parameter for the Gamma distribution for k.
    k_beta (float): Scale parameter for the Gamma distribution for k.

    Returns:
    pd.DataFrame: Simulated DataFrame with columns "individual_id", "beta", "k".
    """
    # Generate individual_id from 1 to n
    individual_id = np.arange(1, n + 1)

    # Generate beta from a Gamma distribution with specified shape and scale
    beta = np.random.gamma(beta_shape, beta_scale, n)

    # Generate k from a Gamma distribution with specified shape and scale
    k = np.random.beta(k_alpha, k_beta, n)

    # Combine all columns into a DataFrame
    data = np.column_stack((individual_id, beta, k))
    columns = ["individual_id", "beta", "k"]

    return (data, columns)

def inner_loop_marginal_likelihood(params_df, data_questions, data_individuals):
    # Draw From the Distribution
    draw_data_questions, draw_columns_questions = simulate_question_parameters(
        data_questions.shape[0], 
        alpha_mean=params_df.loc["alpha_mean", "value"],
        alpha_variance=params_df.loc["alpha_variance", "value"],
        alpha_cols=4, 
        sigma_shape=params_df.loc["sigma_shape", "value"],
        sigma_scale=1
    )
    
    draw_params_questions = pd.DataFrame(draw_data_questions,
                                         columns=draw_columns_questions)

    draw_data_individuals, draw_columns_individuals = simulate_individual_parameters(
        data_individuals.individual_id.nunique(),
        beta_shape=params_df.loc["beta_shape", "value"],
        beta_scale=params_df.loc["beta_scale", "value"],
        k_alpha=params_df.loc["k_alpha", "value"],
        k_beta=params_df.loc["k_beta", "value"]
    )
    
    draw_params_individuals = pd.DataFrame(draw_data_individuals,
                                                 columns=draw_columns_individuals)


    vectorized_df = vectorize_data(
        draw_params_individuals,
        draw_params_questions,
        data_questions,
        data_individuals
    )
    
    new_choice_probabilities = calculate_choice_probabilities(vectorized_df).choice_prob.values
    
    return(new_choice_probabilities)


def calc_marginal_likelihood(params_df, data_questions, data_individuals, n_draws):
        mp.set_start_method('spawn', force=True)
        results = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(inner_loop_marginal_likelihood)(params_df, data_questions, data_individuals) for i in range(n_draws))
        stacked_props = np.vstack(results)
        contributions = np.mean(stacked_props, axis=0)
        contributions = np.log(contributions)
        return {"contributions": contributions, "value": contributions.sum()}

# def calc_marginal_likelihood(params_df, data_questions, data_individuals, n_draws):
#     """
#     Calculate the marginal likelihood by averaging choice probabilities over multiple draws.

#     This function performs the following steps:
#     1. Iteratively draws parameters for questions and individuals from their respective distributions.
#     2. Vectorizes the data using the drawn parameters.
#     3. Calculates choice probabilities for each draw.
#     4. Averages the choice probabilities over all draws to obtain the marginal likelihood.

#     Parameters:
#     params_df (pd.DataFrame): DataFrame containing the parameters for the distributions.
#     data_questions (pd.DataFrame): DataFrame containing the data for questions.
#     data_individuals (pd.DataFrame): DataFrame containing the data for individuals.
#     n_draws (int): Number of draws to perform for averaging.

#     Returns:
#     pd.Series: Averaged choice probabilities over all draws.
#     """
#     Parallel(backend='multiprocessing')
#     results = Parallel(n_jobs=-1)(delayed(inner_loop_marginal_likelihood)(params_df, data_questions, data_individuals) for i in range(n_draws))
#     stacked_props = np.vstack(results)
#     contributions = np.mean(stacked_props, axis=0)
    
#     # Calculate the likelihood
#     contributions = np.log(contributions)
#     return  {"contributions": contributions, "value": contributions.sum()}

def simulate_question_data(n_questions):
    """
    Create a DataFrame with n_questions rows where the first column is a running question_id
    starting at 1 and the second column is correct_answer, which is a draw from a discrete
    uniform distribution from 1 to 4.

    Parameters:
    n_questions (int): Number of questions to simulate.

    Returns:
    pd.DataFrame: A DataFrame with columns 'question_id' and 'correct_answer'.
    """
    # Generate question_id from 1 to n_questions
    question_id = np.arange(1, n_questions + 1)
    
    # Generate correct_answer from a discrete uniform distribution from 1 to 4
    correct_answer = np.random.randint(1, 5, size=n_questions)
    
    # Combine into a DataFrame
    df = pd.DataFrame({
        'question_id': question_id,
        'correct_answer': correct_answer
    })
    
    return df

def simulate_dgp(params_df, n_questions, n_individuals):
    """
    Simulate the data generating process (DGP) for questions and individuals.

    This function performs the following steps:
    1. Randomly draws parameters for questions and individuals.
    2. Loops through each individual and each question to simulate the individual-level data.
    3. Collects the results into a DataFrame.

    Parameters:
    params_df (pd.DataFrame): DataFrame containing the parameters for the distributions.
    n_questions (int): Number of questions to simulate.
    n_individuals (int): Number of individuals to simulate.

    Returns:
    tuple: A tuple containing:
        - draw_params_questions (pd.DataFrame): Simulated question parameters.
        - draw_params_individuals (pd.DataFrame): Simulated individual parameters.
        - simulated_indiviudal_data (pd.DataFrame): Simulated individual-level data.
        - simulated_question_data (pd.DataFrame): Simulated question data.

    """
    # Random Draws of Question Parameters
    (questions_params, question_params_columns) = simulate_question_parameters(
        n_questions, 
        alpha_mean=params_df.loc["alpha_mean", "value"],
        alpha_variance=params_df.loc["alpha_variance", "value"],
        alpha_cols=4, 
        sigma_shape=params_df.loc["sigma_shape", "value"],
        sigma_scale=1
    )
    
    draw_params_questions = pd.DataFrame(questions_params,
                                         columns=question_params_columns)


    # Add correct_answer to question parameters
    question_data = simulate_question_data(n_questions)
    draw_params_questions['correct_answer'] = question_data['correct_answer'].astype(int)

    # Random Draws of Individual Parameters
    (individual_params, individual_params_columns)  = simulate_individual_parameters(
        n_individuals,
        beta_shape=params_df.loc["beta_shape", "value"],
        beta_scale=params_df.loc["beta_scale", "value"],
        k_alpha=params_df.loc["k_alpha", "value"],
        k_beta=params_df.loc["k_beta", "value"]
    )
    
    draw_params_individuals = pd.DataFrame(individual_params,
                                           columns=individual_params_columns)
    # Initialize list to store DataFrames for each individual
    individual_dfs = []

    # Loop through each individual
    for individual_index, individual_row in draw_params_individuals.iterrows():
        
        #Assign times to questions randomly
        ts = np.arange(1, n_questions + 1)
        #np.random.shuffle(ts)
        
        # Creat empty containers for the other variables
        choices = []
        question_ids = []
        individual_ids = []
        remaining_endurance = []
        for question_index, question_row in draw_params_questions.iterrows():
            t = ts[question_index]
            corrects = transform_question_data(pd.Series(question_row.correct_answer.astype(int)).to_numpy())
            alphas = np.array([
                question_row.alpha1, 
                question_row.alpha2, 
                question_row.alpha3, 
                question_row.alpha4
            ])
            sigma = question_row.sigma

            k = individual_row.k
            beta = individual_row.beta
            new_remaining_endurance = calculate_remaining_endurance(t, k)
            remaining_endurance.append(new_remaining_endurance)
            
            util = calculate_utilities(np.array(corrects),np.array(alphas),
                                       np.array(beta),  np.array(sigma * new_remaining_endurance))
            p = softmax(util)

            assert np.isclose(sum(p), 1), "The choice probabilities must sum to 1"
            choice = np.random.choice([1, 2, 3, 4], p=p)
            choices.append(choice)

            question_ids.append(question_row.question_id)
            individual_ids.append(individual_row.individual_id)

        # Create a DataFrame for the current individual
        individual_df = pd.DataFrame({
            'individual_id': individual_ids,
            'question_id': question_ids,
            't': ts,
            'answer': choices,
            'remaining_endurance': remaining_endurance
        })

        # Append the individual DataFrame to the list
        individual_dfs.append(individual_df)

    # Concatenate all individual DataFrames into a single DataFrame
    simulated_data = pd.concat(individual_dfs, ignore_index=True)

    return draw_params_questions, draw_params_individuals, simulated_data