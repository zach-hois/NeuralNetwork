import numpy as np
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
#from .NN import * 
from .AutoEncoder import *
from .validationEvaluation import *
#from .preprocessing import *
from .processing import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy import stats
from Bio import SeqIO

#first importing all of the data sets we will need to use
testData = pd.read_csv("./data/rap1-lieb-test.txt", sep = "\t", names=['seq'])
#print(testData)
positives = pd.read_csv("./data/rap1-lieb-positives.txt", sep = "\t", names = ['seq'])
pos2 = list(positives['seq'])

positives['resp'] = [1] * len(positives) #true positives

neg_list = []
negs = generate_negative_samples("./data/yeast-upstream-1k-negative.fa",17,137,pos2) #takes in the positive liust and removes them
for x in negs:
	# print(x)
	neg_list.append(x)

negatives2 = pd.DataFrame(neg_list,columns=['seq'])
negatives2['resp'] = [0] * len(negatives2)

#print(negatives2)
"""                seq  resp #example of the negative data 
0    TGTCGATATGAGGG     0
1    TAATATACTCTAAA     0
2    AAAACTTTTTTAAA     0
3    CATTTTGGTTACCT     0
4    CCGCTCGCTTCCAA     0
..              ...   ...
133  TCTCGTCCTCGACT     0
134  TCCACTGCTATGAT     0
135  TAAGAAATAAGAAA     0
136  GGCGAGACACTCAC     0
137  GTAGGTGGTTCCTC     0
"""

#going to split the dataset into 70% for training and 30% for testing
#as suggested by the repo there are 500 iterations at an alpha of 0.0001, 6 hidden neurons

x, y = dataPreparation(neg=negatives2, pos=positives, nNeg = 137, nPos = 137) #sample our data
XTrain, XTest, yTrain, yTest = train_test_split(x, y, test_size = 0.33, random_state = 42) #sklearn, train test
yTrain = np.reshape(yTrain, (-1, 1))
yTest = np.reshape(yTest, (-1, 1)) #reshape them

#print((XTrain))
"""
     0_A  0_C  0_G  0_T  1_A  1_C  ...  15_G  15_T  16_A  16_C  16_G  16_T
250    1    0    0    0    0    0  ...     0     1     0     1     0     0
78     0    0    0    1    1    0  ...     0     0     0     0     0     0
185    0    0    1    0    0    1  ...     0     0     0     1     0     0
266    0    0    1    0    0    1  ...     0     0     0     0     0     1
234    1    0    0    0    0    1  ...     0     0     0     0     0     1
..   ...  ...  ...  ...  ...  ...  ...   ...   ...   ...   ...   ...   ...
188    1    0    0    0    0    1  ...     0     0     0     1     0     0
71     1    0    0    0    1    0  ...     0     0     0     0     0     0
106    1    0    0    0    1    0  ...     0     0     0     0     0     0
270    0    0    0    1    0    0  ...     0     0     0     0     0     1
102    0    0    1    0    1    0  ...     0     0     0     0     0     0

Shown above is an example of the input data for the XTraining set. It uses onehot encoding, 
assigning a probability of each base pair appearing at each point in the 17 nucleotide sequence
for the positive and negative sets
"""

base, baseW1, baseW2, = NeuralNetwork(x=XTrain, y=yTrain, hidden=6, iterations = 500, a = 0.001) #make the network
layer1, basePredictions = forward(XTest, baseW1, baseW2)


fpr, tpr, thresholds = metrics.roc_curve(yTrain[:,0], base[:,0], pos_label = 1) #y true, y score, and pos label. baseline AUROC
ROCAUC = metrics.auc(fpr, tpr)
accuracy = metrics.accuracy_score(yTrain[:,0], np.round(base[:,0])) #baseline accuracy

#test statistics
testFPR, testTPR, testThresholds = metrics.roc_curve(yTest[:,0], basePredictions[:,0], pos_label = 1) #y true, y score, and pos label. baseline AUROC
testROCAUC = metrics.auc(testFPR, testTPR)
testAccuracy = metrics.accuracy_score(yTest[:,0], np.round(basePredictions[:,0]))

plt.figure() #plot the ROC curve
plt.plot(fpr, tpr, color = "blue", lw = 2, label = 'Training ROC = %0.2f, Training Accuracy = %0.2f' % (ROCAUC, accuracy))
plt.plot(testFPR, testTPR, color = "red", lw = 2, label = 'Testing ROC = %0.2f, Testing Accuracy = %0.2f' % (testROCAUC, testAccuracy))
plt.plot([0,1], [0,1], color = "black", lw=2, linestyle = '--', label = 'Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Training and Test ROC')
plt.legend(loc="lower right")
#plt.show() #screenshot it

#next i will graph the k folds cross validation technique performance 
#the positives are samples with replacement but only 32 are ultimately used
#this is compared againse 3200 negatives, this shows how well the model holds to generalization (100:1 ratio)

kfoldsPOS = positives.sample(3200, replace = True)
"""
outputDict = kfold(negatives2, kfoldsPOS, k = 10, nNeg = 3200, nPos = 32, iterations = 1000, alpha = 0.0001)
colors = ("blue", "green", "red", "cyan", "magenta", "yellow", "black", "white", "violet", "chartreuse") # 10 curves

plt.figure() #plot the ROC curve

for i, c in zip(range(k), colors):
	plt.plot(outputDict['fpr'][i], outputDict['tpr'][i], color=c, lw = 2, 
		label = '{0} AUROC = {1:0.2f}, Accuracy = {1:0.2f}' ''.format(i, outputDict['auc'][i], outputDict[accuracy][i]))

plt.plot([0,1], [0,1], color = "black", lw=2, linestyle = '--', label = 'Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KFolds Cross Validation AUROC 100:1 Negative to Positive')
plt.legend(loc="lower right")
plt.show() #screenshot it
"""

#Next use ensembling methods to reduce variance
# We could take the average unseen data predictions for each k folds model and average these prediction 
# Or Take the average weights of the k folds cross validation and make a "best" weight
"""
trainNeg = negatives2.sample(n=5000)
trainPos = positives.iloc[1:101,:]
trainPos = trainPos.sample(n=5000, replace = True)
testNeg = negatives2.sample(n=3200, replace=True) #negatives can be resampled because they dont need to be specific
testPos = positives.iloc[101:138,:] #keeping out some of the data for testing


ensDict = kfold(trainNeg, trainPos, k = 10, nNeg = len(trainNeg), nPos = len(trainPos), iterations = 1000, alpha = 0.0001)

testData = pd.concat([testNeg, testPos], axis = 0)
testData = testdata.sample(frac=1)
testDatax = testData['seq'].apply(lambda x: pd.Series(list(x)))
testDatax = pd.get_dummies(pd.DataFrame(testData))

Emodel = ensembleM(ensDict, testDatax)
Eweight = ensembleW(ensDict, testDatax)
finalPredictions = pd.DataFrame({'seq': testData['seq'], 'EModel':Emodel['ensemble'].values, 
					"EWeight":Eweight[:,0], 'true': testData['resp'].values})

#emodel plotting 
fpr, tpr, thresholds = metrics.roc_curve(finalPredictions['true'], finalPredictions['EModel'], pos_label=1)
ROCAUC = metrics.auc(fpr, tpr)
accuracy = metrics.accuracy_score(finalPredictions['true'], np.round(finalPredictions['EModel']))
#Wmodel plotting
testfpr, testtpr, testthresholds = metrics.roc_curve(finalPredictions['true'], finalPredictions['EWeight'], pos_label=1)
testROCAUC = metrics.auc(wfpr, wtpr)
testAccuracy = metrics.accuracy_score(finalPredictions['true'], np.round(finalPredictions['EWeight']))

plt.figure() #plot the ROC curve

for i, c in zip(range(k), colors):
	plt.plot(outputDict['fpr'][i], outputDict['tpr'][i], color=c, lw = 2, 
		label = '{0} AUROC = {1:0.2f}, Accuracy = {1:0.2f}' ''.format(i, outputDict['auc'][i], outputDict[accuracy][i]))
plt.plot(fpr, tpr, color = "blue", lw = 2, label = 'EModel AUCROC = %0.2f, EModel Accuracy = %0.2f' % (ROCAUC, accuracy))
plt.plot(testFPR, testTPR, color = "red", lw = 2, label = 'EWeight AUCROC = %0.2f, EWeight Accuracy = %0.2f' % (testROCAUC, testAccuracy))

plt.plot([0,1], [0,1], color = "black", lw=2, linestyle = '--', label = 'Baseline')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Ensembles')
plt.legend(loc="lower right")
plt.show() #screenshot it
"""



# final predictions model, repeat what was done above
predictionPOS = positives.sample(10000, replace = True)
predictDict = kfold(negatives2, predictionPOS, k = 10, nNeg = 10000, nPos = 10000, iterations = 10000, alpha = 0.0003)
testDatax = testData['seq'].apply(lambda x: pd.Series(list(x)))
testDatax = pd.get_dummies(pd.DataFrame(testData))
Emodel = ensembleM(predictDict, testDatax)
finalPredictions = pd.DataFrame({'seq': testData['seq'], 'EModel':Emodel['ensemble'].values})
finalPredictions = finalPredictions[['seq', 'mod']]
finalPredictions.to_csv("./HOISINGTONPredictions.txt", sep = "\t", index=False, header=False)





