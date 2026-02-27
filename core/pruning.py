# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 21:35:29 2026

@author: konra
"""

from numba import njit

@njit
def prune(
        depth, 
        k, 
        current_min, 
        current_max
):
    return False