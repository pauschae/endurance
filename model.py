# -*- coding: utf-8 -*-
"""
Model

This file contains all parts of the model that we need to calculate
choic probabilities.

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

import numpy as np

def utilities(alpha, beta, correct, sigma):
    """
    

    Parameters
    ----------
    alpha : numeric 4 times 1 vector 
        baseline propensity answer with option i.
    beta : numeric 4 times 1 vector
        ability: for a well rested individual 
        this gets added to the utility of the correct answer.
    correct : numeric 4 times 1 vector
        vector that contains a 1 for the correct answer
        and zeros otherwise.
    sigma : numeric 4 times 1 vector
        utility difference between correct and incorrect answers.
        Declines over time to model cognitive exhaustion

    Returns
    -------
    A numeric 4 times 1 vector with the utilities of 
    each answer option.

    """
    
    utilities = alpha + correct*beta*sigma
    
    return utilities

