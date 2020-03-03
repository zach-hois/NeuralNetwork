import numpy as np
import matplotlib.pyplot as plt # to plot error during training
from NN import NeuralNetwork, oneHot

#this is the training set for the 8 x 3 x 8 
x = np.array([[1,0,0,0,0,0,0,0], #input
			 [0,1,0,0,0,0,0,0],
			 [0,0,1,0,0,0,0,0],
			 [0,0,0,1,0,0,0,0],
			 [0,0,0,0,1,0,0,0],
			 [0,0,0,0,0,1,0,0],
			 [0,0,0,0,0,0,1,0],
			 [0,0,0,0,0,0,0,1]])
y = np.array([[1,0,0,0,0,0,0,0], #output
			 [0,1,0,0,0,0,0,0],
			 [0,0,1,0,0,0,0,0],
			 [0,0,0,1,0,0,0,0],
			 [0,0,0,0,1,0,0,0],
			 [0,0,0,0,0,1,0,0],
			 [0,0,0,0,0,0,1,0],
			 [0,0,0,0,0,0,0,1]])

NN = NeuralNetwork(x,y) #create the neural net
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