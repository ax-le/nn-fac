# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:01:35 2020

@author: amarmore
"""

class ArgumentException(BaseException): pass
class InvalidRanksException(ArgumentException): pass
class CustomNotEngouhFactors(ArgumentException): pass
class CustomNotValidFactors(ArgumentException): pass
class CustomNotValidCore(ArgumentException): pass
class InvalidInitializationType(ArgumentException): pass

class OptimException(BaseException): pass
class ZeroColumnWhenUnautorized(OptimException): pass