#!/usr/bin/env python
# -*- coding: utf-8 -*-
##################################### Help ####################################
"""Simple Naive Bayes implementation
Usage:
  naiveBayes.py
  naiveBayes.py --help
Options:
  -h --help                   Show this screen.
"""


################################### Imports ###################################
from docopt import docopt
import numpy as np
import pandas as pd
import pickle, random

################################## Functions ##################################

################################### Classes ###################################		


##################################### Main ####################################
# Seed for reproductibility
random.seed(1)

# Data loading
data = pickle.load(open("/home/koala/Documents/Scripts/naiveBayes/T.ft", "r"))
nbFt = len(data[0]) - 1

# Counting the probabilities for each column for each class
evidence = [pd.DataFrame()]*nbFt
for i in xrange(nbFt):
	evidence[i] = pd.crosstab(index = data[:,-1], columns = data[:,i], margins=True)

# e.g feature 1, proba of each class for the value 0
print evidence[1][0]
# e.g feature 2, proba of class 2 knowing value = 1
print evidence[2][1][2]/float(evidence[2][1]["All"])

# Counting the overall probabilities (prior)

# Prediction
# Multiplying the probabilities for each column of the input
# Multiplying by the overall probabilities
# Outputing the most likely
