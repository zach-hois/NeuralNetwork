import numpy as np
import matplotlib.pyplot as plt # to plot error during training

def oneHot(seq): #encodes a onehot array of the input sequence and flattens
	encode = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
	sequenceID = {"A": 0, "C": 1, "G": 2, "T": 3} 
	return encode[[sequenceID[i] for i in seq], :].flatten() #change letters to #


class NeuralNetwork:
    def __init__(self, inputs, outputs, setup=[[68,25,"sigmoid",0],[25,1,"sigmoid",0]],lr=.05,seed=1,error_rate=0,bias=1,iter=500,lamba=.00001,simple=0):
    	self.inputs = inputs
    	self.outputs = outputs

    	#first weights dont matter just an initialization
    	self.weights = np.array([[.50], [.50], [.50], [.50]])
    	self.error_history = [] #initializations
    	self.epoch_list = []

    #def make_weights(self):
    def sigmoid(self, x, deriv = False): #the activation function 
    	if deriv == True:
    		return x * (1-x)
    	return 1 / (1 + np.exp(-x))

    def feedforward(self): #push the data through the neurons
    	self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    def backprop(self): #go backwards through the network to update the weights
    	self.error = self.outputs - self.hidden
    	delta = self.error * self.sigmoid(self.hidden, deriv = True)
    	self.weights += np.dot(self.inputs.T, delta)

    def train(self, epochs=25000): #fit the net for 25000 iterations
    	for epoch in range(epochs):
    		self.feedforward() #feed forward and produce an output
    		self.backprop() #take this output, update the weights
    		self.error_history.append(np.average(np.abs(self.error))) #keep track of error history
    		self.epoch_list.append(epoch) #add the iteration to the list

    def predict(self, newInput): #create a prediction of the never before seen data
    	prediction = self.sigmoid(np.dot(newInput, self.weights))
    	return prediction


#def activation(x):
    
