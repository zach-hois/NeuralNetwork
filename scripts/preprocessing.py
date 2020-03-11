import numpy as np
import pandas as pd
from Bio import SeqIO, pairwise2 #we will use this for easy seq alignment

def scoreAlignment(positive, negative): #we will find the scores that are closely aligned to true negatives
	output = np.zeros((len(negative), len(positive))) #initialize our output matrix
	for n in range(output.shape[0]):
		for p in range(output.shape[1]): #run this through the entire matrix
			align = pairwise2.align.localms(positive.loc[p, 'seq'], negative[n].seq, 5, -4, -3, -0.1)
			rawScore = 0 #initialize
			for score in align:
				rawScore += score[2] #add the new score
			output[n, p] = rawScore #assign to matrix
	return output 

def negativeSplit(negative): #this will make an array of the negative sequences
	output = pd.DataFrame()
	for n in range(len(negative)):
		#split them all into strings that overlap
		length = 16
		sequences = list()
		for i in range(length, len(negative[n].seq)):
			seq = negative[n].seq[i - length: i + 1] #select the sequence
			sequences.append(str(seq))#keep them here
		ID = [n] * len(sequences) #sequence ID initialization
		seqID = [negative[n].id] * len(sequences)
		resp = [0] * len(sequences) #our response variable
		dfIteration = pd.DataFrame({
			"ID": ID,
			"seqID": seqID,
			"sequences":sequences,
			"resp": resp}) #put all of these pieces of data into one frame
		output = output.append(dfIteration) #and append to the dataframe for output
	return output.reset_index(drop = True) #return out output	