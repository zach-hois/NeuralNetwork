import numpy as np
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
from .NN import * 
from .AutoEncoder import *
from .validationEvaluation import *
from preprocessing import *
from sklearn import metrics

#first importing all of the data sets we will need to use
testData = pd.read_csv("./data/rap1-lieb-test.txt", sep = "\t", names=['seq'])
#print(testData)
positives = pd.read_csv("./data/rap1-lieb-positives.txt", sep = "\t", names = ['seq'])
#print(positiveData)
positives['resp'] = [1] * len(positives) #true positives

###################come back to this .. 
#upsteam = 


#going to split the dataset into 70% for training and 30% for testing

