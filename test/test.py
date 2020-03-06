from scripts import NN
from scripts import AutoEncoder
import numpy as np


def test_encoder():
	test = np.zeros((2,2)) #initialize a test matrix and random weights
	weight1 = np.ones((2,2))
	weight2 = np.ones((2))

	assert AutoEncoder.sigmoid(0, deriv = False) == 0.5 #check my math on these
	assert AutoEncoder.sigmoid(0, deriv = True) == 0

	layer1, layer2 = AutoEncoder.forward(test, weight1, weight2) #layers we expect

	assert layer2.all() == np.full((2,2), 0.5).all() #do they line up? #this tests the "forward" function

	#next to test backprop function (arguably most important)

	testY = np.zeros((2,2)) #output
	testLayer1 = np.full((2,2), 0.5)
	testLayer2 = np.full((2,2), 0.8234)
	testWeight1 = np.ones((2,2))
	testWeight2 = np.ones((2,2))

	outputW1, outputW2 = AutoEncoder.backprop(x=test, y=testY, layer1=testLayer1, layer2=testLayer2, weight1=testWeight1, weight2=testWeight2, a=1)

	assert outputW1.all() == np.ones((2,2)).all() #True? same as before?

#def test_encoder_relu():


#def test_one_d_ouput():
