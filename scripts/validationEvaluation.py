import numpy as np
import pandas as pd
from .AutoEncoder import *
from sklearn import metrics

def dataPreparation(neg, pos, nNeg, nPos):
	#combining the negative and positive sequences and oneHot encode the features

	trainNeg = neg.sample(n = nNeg)#, randomState = 100)
	pos = pos.sample(n = nPos)#, randomState = 100) #sample the data

	#data combination
	X = pd.concat([trainNeg, pos], ignore_index = True) #independent variable
	y = X['resp'].values #this takes out response variable that was previously taken in "preprocessing.py"
	X = X.drop(['resp'], axis=1) #drop the training
	X = X['seq'].apply(lambda x: pd.Series(list(x))) #split the characters into columns
	X = pd.get_dummies(pd.DataFrame(X))

	return X, y #return the variables

def split(neg, pos, k, nNeg, nPos): #simple data split function 
	splitNeg = [0] * k
	splitPos = [0] * k
	for i in range(k):
		splitNeg[i] = neg.sample(n=nPos, replace=True) #data grab
		splitPos[i] = pos.sample(n=nPos)
	return splitNeg, splitPos

"""
K FOLDS CROSS VALIDATION FUNCTION
A K folds cross validation function helps us estimate how accurately a predictive model will preform in practics
It is given a dataset of known data where we train, then tested on never before seen data
this is the "cross validation"
"""

def kfold(neg, pos, k, nNeg, nPos, iterations, alpha): #begin the function!
    fpr = [0] * k #false positive rate
    tpr = [0] * k #true pos
    auc = [0] * k #area under the curve
    weights = [0, 0] * k #weights
    mse = [0] * k
    data = [0] * k
    accu = [0] * k
    outDict = {'fpr': fpr, 'tpr': tpr, 'auc': auc,
               'weights': weights, 'mse': mse,
               'accu': accu, 'data': data} #this will be out output dictionary
    splitsNeg, splitsPos = split(neg, pos, k, nNeg=nNeg, nPos=nPos) #use the function we just wrote to store the splits
    for i in range(k): #remove one group from the training set for validation
        print(i)
        arr = np.array(range(k)) #everything except i array

        mask = np.ones(arr.shape, dtype=bool)
        mask[i] = 0
        noti = arr[mask] #maskoff

        trainNeg = pd.DataFrame(columns=['seq', 'resp']) #combine the data
        trainPos = pd.DataFrame(columns=['seq', 'resp']) #sequence and response

        for x in noti: #oversample the positive data
            trainNeg = pd.concat([trainNeg, splitsNeg[x]], ignore_index=True)
            trainPos = pd.concat([trainPos, splitsPos[x]], ignore_index=True)

        XTrain = pd.concat([trainNeg, trainPos], ignore_index=True) #combining the true positives and true negatives for training
        XTrain = XTrain.sample(frac=1).reset_index(drop=True)
        yTrain = XTrain['resp'].values.astype(np.uint64) #take the response variable
        yTrain = np.reshape(yTrain, (-1, 1))
        XTrain = XTrain.drop(['resp'], axis=1) #drop these from the training set

        XTrain = XTrain['seq'].apply(lambda x: pd.Series(list(x))) #splitting the characters into columns
        XTrain = pd.get_dummies(pd.DataFrame(XTrain))

        XTest = pd.concat([splitsNeg[i], splitsPos[i]], ignore_index=True) #combine these again for testing
        XTest = XTest.sample(frac=1).reset_index(drop=True)
        data = XTest #storage
        yTest = XTest['resp'].values.astype(np.uint64)
        yTest = np.reshape(yTest, (-1, 1))
        XTest = XTest.drop(['resp'], axis=1)
        XTest = XTest['seq'].apply(lambda x: pd.Series(list(x)))
        XTest = pd.get_dummies(pd.DataFrame(XTest)) #same sequence of commands as before

        nn, weight1, weight2 = NeuralNetwork(x=XTrain, y=yTrain, hidden=6, iterations=iterations, a=alpha) #begin our training, 6 hidden neurons

        layer1, preds = forward(XTest, weight1, weight2) #test and predict never before seen test set

        data['preds'] = preds[:, 0] #store these predictions in the data set we previously made
        data['se'] = (data['resp'].values - data['preds'].values)**2
        mse = np.mean(data['se']) #mean square error calculation
        weights = [weight1, weight2] #store weights

        fpr, tpr, thresholds = metrics.roc_curve(yTest[:, 0], preds[:, 0], pos_label=1) #creation of the ROC curve
        rocAUC = metrics.auc(fpr, tpr) #calculate AUC from the false positive and true positive rate
        outDict['fpr'][i] = fpr #store all of the collected values with our previously initialized output dictionary 
        outDict['tpr'][i] = tpr
        outDict['auc'][i] = rocAUC
        outDict['weights'][i] = weights
        outDict['mse'][i] = mse
        outDict['data'][i] = data
        outDict['accu'][i] = metrics.accuracy_score(yTest[:, 0], np.round(preds[:, 0]))
    return outDict #return the output 

#search for the optimal parameters next
def optimalParameters(XTrain, yTrain, iterations, a):
	auROC = np.zeros((len(iterations), len(a)))
	for i in range(len(iterations)):
		for alpha in range(len(a)):
			y = np.where(a == a[alpha])[0][0] #we are saving these 
			vvs, weight1, weight2 = NeuralNetwork(x=XTrain, y=yTrain, hidden=6, iterations=iterations[i], a=a[alpha]) #run the NN and save the results
			fpr, tpr, thresholds = metrics.roc_curve(yTrain[:,0], vvs[:,0], pos_label=1)
			auROC[i,alpha] = metrics.auc(fpr, tpr) #calculare roc with fpr and tpr
	return auROC

#finally we will use the previously computed model dictionary and test data
# to ensemble mdoel and then appropriately weight
def ensembleM(modelDict, testData):
    predDF = pd.DataFrame() #initialize the prediction data frame
    for i in range(len(modelDict['data'])):
        layer1, pred = forward(testData, weight1 = modelDict['weights'][i][0], weight2=modelDict['weights'][i][1])
        predDF = pd.concat([predDF, pd.DataFrame({i: pred[:,0]})], axis=1) #add to DF
    predDF['ensemble'] = predDF.mean(axis=1)
    return predDF

#last required part, weights for this
def ensembleW(modelDict, testData):
	weight1 = modelDict['weights'][0][0]
	weight2 = modelDict['weights'][0][1]
	for i in range(1, len(modelDict['data'])):
		weight1 += modelDict['weights'][i][0] #add the weights
		weight2 += modelDict['weights'][i][1] #add the weights
	weight1 = weight1 / len(modelDict['data'])
	weight2 = weight2 / len(modelDict['data']) #normalize
	layer1, pred = forward(testData, weight1=weight1, weight2=weight2)
	return pred














def parseFasta(filename): #This is used to take just the sequence from the fasta document
    seq = ""
    with open(filename) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq += line.strip()
    return seq

def pairs(filename):
    with open(filename) as fh:
        for line in fh:
            line = line.strip().split()
            yield line[0], line[1]    


