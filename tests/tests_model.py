# -*- coding: utf-8 -*-
"""
Decisions

Created on Mon Oct 23 16:13:21 2023

@author: pis2
"""

import numpy as np
from model.decisions import utilities

def test_utilities():
    alpha_1 = np.array([1, 2, 3, 4])
    beta_1 = np.array([0.5, 0.5, 1, 1.5])
    sigma_1 = 1
    correct_1 = np.array([0, 0, 0, 1])
    utilities_1 = np.array([1, 2, 3, 5.5])
    
    assert utilities_1 == utilities(alpha_1, beta_1, correct_1, sigma_1)