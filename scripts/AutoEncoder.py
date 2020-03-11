import numpy as np
import matplotlib.pyplot as plt # to plot error during training

def architecture(inputSize = 8, compressionSize = 3, outputSize = 8):
	# first making the neural net architecture the 8 x 3 x 8 as requires
	# will assign weights to the first and second layer and output them
	weight1 = np.random.randn(inputSize, compressionSize) * np.sqrt(2 / inputSize)
	weight2 = np.random.randn(compressionSize, outputSize) * np.sqrt(2 / compressionSize)
	return weight1, weight2 #randomly make weights and return them to be used further

def sigmoid(x, deriv = False): #the activation function 
    if deriv == True:
    	return x * (1-x)
    return 1 / (1 + np.exp(-x))

def forward(x, weight1, weight2):
	#forward propogation
	#takes sequence data as an input along with random weights
	#returns transformed values (second is predictions from the model)
	layer1 = sigmoid(np.dot(x, weight1)) 
	layer2 = sigmoid(np.dot(layer1, weight2)) #the output
	return layer1, layer2

def backprop(x, y, layer1, layer2, weight1, weight2, a):
	#go backwards through the network to update the weights
	#x and y are the independent and dependent variables respectively
	#a is our learning rate
	dLoss = 2 * (y-layer2) #derivative of loss Mean Squared Error 
	dActivation = sigmoid(layer2, deriv = True) #activation derivative

	deltaWeight2 = np.dot(layer1.T, (dLoss * dActivation)) #chain rule

	#backwards 1 layer
	dHidden = np.dot((dLoss * dActivation), weight2.T)
	deltaWeight1 = np.dot(x.T, (dHidden * sigmoid(layer1, deriv = True))) #chain rule again

	#update the weights with the dLoss
	weight1 += a * deltaWeight1
	weight1 += a * deltaWeight2

	return weight1, weight2 #return the new weights

def NeuralNetwork(x, y, hidden, iterations=100, a=1):
	#here we will initialize and train the NN
	#x and y are our inputs, hidden is the number of neurons in the inner layers
	weight1, weight2 = architecture(inputSize = x.shape[1], compressionSize=hidden, outputSize=y.shape[1])
	i = 0
	while i < iterations:
		layer1, layer2 = forward(x=x, weight1=weight1, weight2=weight2)
		weight1, weight2 = backprop(x=x, y=y, l1=l1, l2=l2, weight1=weight1, weight2 = weight2, a=a)
		i +=1 
	return l2, w1, w2