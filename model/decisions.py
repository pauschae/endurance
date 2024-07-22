# -*- coding: utf-8 -*-
"""
Decisions

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

def utilities(alpha, beta, correct, sigma):
    """
    

    Parameters
    ----------
    alpha : numpy array 4 times 1  
        baseline propensity answer with option i.
    beta : numpy array 4 times 1 
        ability: for a well rested individual 
        this gets added to the utility of the correct answer.
    correct : numpy array 4 times 1 
        vector that contains a 1 for the correct answer
        and zeros otherwise.
    sigma : numpy array 4 times 1 
        utility difference between correct and incorrect answers.
        Declines over time to model cognitive exhaustion

    Returns
    -------
    A numeric 4 times 1 vector with the utilities of 
    each answer option.

    """
    
    utilities = alpha + correct*beta*sigma
    
    return utilities

