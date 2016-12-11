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
testData = np.array([[2,0,1,0,1,0,2,2],[0,1,2,1,2,1,0,1],[1,0,1,1,1,1,1,2]])
nbFt = len(data[0]) - 1
nbTr = len(testData)

# Counting the probabilities for each column for each class
likelihood = [pd.DataFrame()]*nbFt
for i in xrange(nbFt):
	likelihood[i] = pd.crosstab(index = data[:,-1], columns = data[:,i], margins=True)

# e.g feature 1, proba of each class for the value 0
# likelihood[1][0]
# e.g feature 2, proba of value = 1 knowing class = 2
# likelihood[2][1][2]/float(likelihood[2]["All"][2])

# Counting the overall probabilities (prior)
prior = pd.crosstab(index = data[:,-1], columns="count", margins=True)["count"]
nbClasses = len(prior)-1

# e.g prior of class 1
# prior[1]/float(prior["All"])

# Prediction
prediction = [0]*nbTr
for i in xrange(nbTr):
	# Multiplying by the overall probabilities
	probas = prior[:nbClasses]/float(prior[-1])
	# Multiplying the probabilities for each column of the input
	for j in xrange(nbFt):
		probas = probas * likelihood[j][testData[i,j]][:-1]/likelihood[j]["All"][:-1]
	# Outputing the most likely
	prediction[i] = list(probas[probas == max(probas)].index)[0]
