import numpy as np
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
from .NN import * 
from .AutoEncoder import *
from .validationEvaluation import *
from .preprocessing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import stats
from Bio import SeqIO

#first importing all of the data sets we will need to use
testData = pd.read_csv("./data/rap1-lieb-test.txt", sep = "\t", names=['seq'])
#print(testData)
positives = pd.read_csv("./data/rap1-lieb-positives.txt", sep = "\t", names = ['seq'])
print(positives)
positives['resp'] = [1] * len(positives) #true positives

#parsing the upstream negative fasta
negatives = parseFasta("./data/yeast-upstream-1k-negative.fa")
#print(negatives)
scoreAlignment(positives, negatives)

negatives = negativeSplit(negatives)

###################come back to this .. 
#upsteam = 


#going to split the dataset into 70% for training and 30% for testing
#as suggested by the repo there are 500 iterations at an alpha of 0.0001, 6 hidden neurons

x, y = dataPreparation(neg=negatives, pos=positives, nNeg = 137, nPos = 137) #sample our data
XTrain, XTest, yTrain, yTest

