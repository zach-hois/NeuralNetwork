import numpy as np
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
#from .NN import * 
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
XTrain, XTest, yTrain, yTest = train_test_split(x, y, test_size = 0.3, random_state = 1231) #sklearn, train test
yTrain = np.reshape(yTrain, (-1, 1))
yTest = np.reshape(yTest, (-1, 1)) #reshape them

base, baseW1, baseW2, = NeuralNetwork(x=XTrain, y=yTrain, hidden=6, iterations = 500, alpha = 0.001) #make the network
layer1, basePredictions = forward(XTest, baseW1, baseW2)

fpr, tpr, thresholds = metrics.roc_curve(yTrain[:,0], base[:,0], pos_label = 1) #y true, y score, and pos label. baseline AUROC
ROCAUC = metrics.auc(fpr, tpr)
accuracy = metrics.accuracy_score(yTrain[:,0], np.round(basePredictions[:,0])) #baseline accuracy

#test statistics
testFPR, testTPR, testThresholds = metrics.roc_curve(yTest[:,0], basePredictions[:,0], pos_label = 1) #y true, y score, and pos label. baseline AUROC
testROCAUC = metrics.auc(testFPR, testTPR)
testAccuracy = metrics.accuracy_score(yTest[:,0], np.round(basePredictions[:,0]))

plt.figure() #plot the ROC curve
plt.plot(fpr, tpr, color = "blue", lw = 2, label = 'Training ROC = %0.2f, Training Accuracy = %0.2f' % (ROCAUC, accuracy))
plt.plot(fpr, tpr, color = "red", lw = 2, label = 'Testing ROC = %0.2f, Testing Accuracy = %0.2f' % (testROCAUC, testAccuracy))
plt.plot([0,1], [0,1], color = "black", lw=2, linestyle = '--', label = 'Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training and Test ROC')
plt.legend(loc="lower right")
plt.show()


