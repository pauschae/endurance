    # -*- coding: utf-8 -*-
"""
Decisions

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""
import numpy as np
from scipy.special import softmax, logsumexp
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

#@jit(nopython=True)
def calculate_utilities(correct, alpha, beta, sigma, cognitive_endurance, t):
    """
    Calculate utilities as a function of item parameters and remaining endurance.

    Parameters
    ----------
    correct : numpy array of shape (n, 1)
        Boolean array containing True for the correct answer and False otherwise.
    alpha : numpy array of shape (n, 1)
        Baseline propensity to answer with option i.
    beta : numpy array of shape (n, 1)
        Float values representing the ability for a well-rested individual.
    sigma : numpy array of shape (n, 1)
        Float values indicating the sensitivity of the question's net ability.
    cognitive_endurance : numpy array of shape (n, 1)
        Float values between 0 and 1 indicating the endurance parameter.
    t : numpy array of shape (n, 1)
        Integer values indicating the number of questions answered since the first question.

    Returns
    -------
    numpy array of shape (4*n,)
        A numeric vector with the utilities of each answer option.
    """
    correct_bonus = beta * sigma * np.maximum(1 - cognitive_endurance * (t-1), 0)
    utilities = alpha + correct_bonus*correct
    
    return utilities


#@jit(nopython=True)
def transform_question_data(answer):
    """
    Transforms question data into a boolean array for vectorized calculations.

    Parameters
    ----------
    answer : numpy array of correct answers by question. Array contains the
        item number i_q

    Returns
    -------
    numpy array
        The data transformed into a boolean array. This array has 4 times the
        original size and contains a True value for question i at position
        4*i+i_q-1
    """
    nrows = len(answer)
    num_options = 4  # Assuming there are 4 possible options per question
    
    # Create a boolean matrix initialized to False
    correct = np.zeros((nrows, num_options), dtype=bool)
    
    # Use numpy advanced indexing to set the correct positions to True
    for i in range(nrows):
        correct[i, answer[i] - 1] = True
    
    # Flatten the boolean array to match the original function's return shape
    out = correct.flatten()
    return out


#@jit(nopython=True)
def scale_to_item_level(n_questions, data):
    """
    Transform individual data to the scale of question items.

    Parameters
    ----------
    n_questions : int
        The number of questions.
    data : numpy.ndarray or pandas.Series
        An array or series with individual-level data.

    Returns
    -------
    numpy.ndarray
        A numpy array at the question level that can be used for vectorized analysis.
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
        DataFrame containing the correct answer for the questions.
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
    output_dict : Dictionary
        A Dictionary containing the vectorized data (1 row per question answer),
        including the following columns:
        - correct : 1 if the answer is correct.
        - alpha : Combined alpha parameters for each question.
        - beta : Scaled individual difficulty parameters.
        - sigma : Question sensitivty parameters.
        - remaining_endurances: endurance at current time t
        - answer: dummy indicating if this is the correct answer
        - The resulting dataset also includes the question_id, individual_id and t
        
        The dataset is sorted by individual ids and within individual by 
        question id.        
        
    """
    
    n_questions = len(params_questions)
    n_individuals = len(data_individuals) // n_questions
      
    # Individual Level 
    beta = scale_to_item_level(n_questions, params_individuals["beta"].to_numpy())
    k = scale_to_item_level(n_questions, params_individuals["k"].to_numpy())
    answer = transform_question_data(data_individuals['answer'].to_numpy())

    # Question Level    
    alpha = params_questions[["alpha1", "alpha2", "alpha3", "alpha4"]].to_numpy().flatten()
    alpha = np.tile(alpha, n_individuals)
    sigma = np.tile(scale_to_item_level(1, params_questions["sigma"].to_numpy()), n_individuals)

    
    # Time Varying
    t = scale_to_item_level(1, data_individuals['t'].to_numpy())
    correct = transform_question_data(data_questions['correct_answer'].to_numpy())
    correct = np.tile(correct, n_individuals)
    
    
    #Set ids
    question_id = np.repeat(data_individuals['question_id'], 4)
    individual_id = np.repeat(data_individuals['individual_id'], 4)


    output_dict = {
        'correct': correct,
        'alpha': alpha,
        'beta': beta,
        'sigma': sigma,
        'answer': answer,
        't': t,
        'k': k,
        'question_id': question_id,
        'individual_id': individual_id
    }
    
    return output_dict

def calculate_choice_probabilities(vectorized_dict):
    """
    Calculate choice probabilities and filter Output based on answer.

    This function computes the utility scores using the `utilities` function and stores them
    in the 'res_u' column of the input DataFrame. It then calculates the choice probabilities 
    using the softmax function on these utility scores, storing the result in the 'choice_prob'
    column. Finally, it filters the DataFrame to include only rows where the 'answer' column 
    is equal to 1.

    Parameters:
    vectorized_df (pd.DataFrame): A DataFrame containing the following columns:
        - 'correct': (int) dummy indicating if this is the correct answer.
        - 'alpha': (float) Parameter alpha for the utility calculation.
        - 'beta': (float) Parameter beta for the utility calculation.
        - 'sigma': (float) Parameter sigma for the utility calculation.
        - 'individual_id': (int) Identifier for individuals.
        - 'question_id': (int) Identifier for questions.
        - 'answer': (int) dummy indicating if this is the chosen answer

    Returns:
        A scaler with the overall probability to observe the entire choice
        profile.
    """
    vectorized_dict['res_u'] = calculate_utilities(correct=vectorized_dict['correct'],
                                                alpha=vectorized_dict['alpha'], 
                                                beta=vectorized_dict['beta'],
                                                sigma=vectorized_dict['sigma'],
                                                cognitive_endurance= vectorized_dict['k'],
                                                t=vectorized_dict['t'])
    
    assert isinstance(vectorized_dict['res_u'], np.ndarray), f"res_u must be a NumPy array, but got {type(vectorized_dict['res_u'])}"
    
    #Recast The Utilities to and use the softmax in a vectorized way
    u_array = vectorized_dict['res_u'].reshape((-1, 4))
    softmax_array = softmax(u_array, axis=1)
    flattened_softmax_array = softmax_array.flatten()
    
    #Filter to answers that were actually chosen
    answer = vectorized_dict["answer"]
    selected_answers = flattened_softmax_array[answer]
    
    #Multiply the choice probabilities to get the probability for the entire profile
    # Use logarithms to calculate the product in a numerically stable way
    log_product = np.sum(np.log(selected_answers))

    # Exponentiate the result to get the product
    out = log_product
    return out

#@jit(nopython=True)
def simulate_question_parameters(n, alpha_variance=1,
                                 alpha_cols=4, sigma_shape=2, sigma_scale=1):
    """
    Simulate a dataset with n rows.

    Parameters:
    n (int): Number of rows to simulate.
    alpha_variance (float): Variance of the normal distribution for alpha.
    alpha_cols (int): Number of alpha columns to generate.
    sigma_shape (float): Shape parameter for the Gamma distribution for sigma.
    sigma_scale (float): Scale parameter for the Gamma distribution for sigma.

    Returns:
    pd.DataFrame: Simulated DataFrame with columns "question_id", "alpha1", ..., "alphaN", "sigma".
    """
    # Generate question_id from 1 to n
    question_id = np.arange(1, n + 1)

    # Generate alpha from a normal distribution with mean m and variance v
    alpha = np.random.normal(0, np.sqrt(alpha_variance), (n, alpha_cols))
    
    # Generate sigma from a Gamma distribution with specified shape and scale
    sigma = np.random.gamma(sigma_shape, sigma_scale, n)

    # Combine all columns into a DataFrame
    data = np.column_stack((question_id, alpha, sigma))
    columns = ["question_id"] + [f"alpha{i+1}" for i in range(alpha_cols)] + ["sigma"]

    return (data, columns)

#@jit(nopython=True)
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

# def calc_marginal_likelihood(params_df, data_questions, data_individuals, n_draws):
#     """
#     Calculate the marginal likelihood using parallel processing.

#     Parameters:
#     params_df (pd.DataFrame): DataFrame with distribution parameters.
#     data_questions (pd.DataFrame): DataFrame with question data.
#     data_individuals (pd.DataFrame): DataFrame with individual data.
#     n_draws (int): Number of draws for simulation.

#     Returns:
#     dict: Contains contributions and the calculated likelihood value.
#     """
#     print(f"{n_draws} draws")
    
#     # Create parallel lists for each argument
#     params_parallel = [params_df] * n_draws
#     data_questions_parallel = [data_questions] * n_draws
#     data_individuals_parallel = [data_individuals] * n_draws
    
#     # Use ProcessPoolExecutor for parallel processing
#     with ProcessPoolExecutor() as executor:
#         results = list(
#             executor.map(
#                 inner_loop_marginal_likelihood,
#                 params_parallel,
#                 data_questions_parallel,
#                 data_individuals_parallel
#                 )
#             )
#     # Stack the probabilities from results
#     stacked_props = np.vstack(results)
#     contributions = np.mean(stacked_props, axis=0)
#     contributions = np.log(contributions)
#     value = contributions.sum()

#     print(f"Value of the likelihood: {value}")
#     return {"contributions": contributions, "value": value}

def calc_marginal_likelihood(params_df, data_questions, data_individuals, n_draws):
    print(f"{n_draws} draws")

    results = []
    for _ in range(n_draws):
        probabilities = inner_loop_marginal_likelihood(params_df, data_questions, data_individuals)
        results.append(probabilities)

    stacked_props = np.vstack(results)
    value = logsumexp(a=stacked_props, axis=0, b = 1/n_draws)
    
    print(f"Value of the likelihood: {value}")
    return {"value": value}


def inner_loop_marginal_likelihood(params_df, data_questions, data_individuals):
    """
    A single iteration of calculating marginal likelihood.

    Parameters:
    params_df (pd.DataFrame): DataFrame with distribution parameters.
    data_questions (pd.DataFrame): DataFrame with question data.
    data_individuals (pd.DataFrame): DataFrame with individual data.

    Returns:
    np.array: Choice probabilities for the iteration.
    """
    draw_data_questions, draw_columns_questions = simulate_question_parameters(
        data_questions.shape[0], 
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

    vectorized_dict = vectorize_data(
        draw_params_individuals,
        draw_params_questions,
        data_questions,
        data_individuals
    )

    new_choice_probabilities = calculate_choice_probabilities(vectorized_dict)
    return new_choice_probabilities

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
        t = np.arange(1, n_questions + 1)
        #np.random.shuffle(t)
        
        # Creat empty containers for the other variables
        choices = []
        question_ids = []
        individual_ids = []
        for question_index, question_row in draw_params_questions.iterrows():
            question_t = t[question_index]
            correct = transform_question_data(pd.Series(question_row.correct_answer.astype(int)).to_numpy())
            alpha = np.array([
                question_row.alpha1, 
                question_row.alpha2, 
                question_row.alpha3, 
                question_row.alpha4
            ])
            sigma = question_row.sigma

            k = individual_row.k
            beta = individual_row.beta
            
            util = calculate_utilities(correct = np.array(correct),
                                       alpha = np.array(alpha),
                                       beta = np.array(beta),
                                       sigma = np.array(sigma),
                                       cognitive_endurance = np.array(k),
                                       t = np.array(question_t))
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
            't': t,
            'answer': choices,
            'cognitive_endurance': k
        })

        # Append the individual DataFrame to the list
        individual_dfs.append(individual_df)

    # Concatenate all individual DataFrames into a single DataFrame
    simulated_data = pd.concat(individual_dfs, ignore_index=True)

    return draw_params_questions, draw_params_individuals, simulated_data