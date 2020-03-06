import numpy as np
import matplotlib.pyplot as plt # to plot error during training
import pandas as pd
from .NN import * 
from .AutoEncoder import *

testData = pd.read_csv("./data/rap1-lieb-test.txt", sep = "\t", names=['seq'])
print(testData)
positiveData = pd.read_csv("./data/rap1-lieb-positives.txt", sep = "\t", names = ['seq'])
print(positiveData)

"""
NN = [[]]
NN = NeuralNetwork(NN, x,y) #create the neural net
NN.train() #train it for 25k iterations

test1 = np.array()
test2 = np.array()

print(NN.predict(test1), '- Correct: ', test1[0][0])
print(NN.predict(test2), '- Correct: ', test2[0][0])

plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()
"""

