import numpy as np
import random 
import numpy as np
import pandas as pd
from Bio import SeqIO, pairwise2 #we will use this for easy seq alignment

def parse_fasta(filename): #this is what i will use to take my negatives from the file
	sequences = {}
	with open(filename) as fh:
		seq = ""
		for line in fh:
			if line.startswith(">"):
			
				if len(seq) > 0:
					sequences[header] = seq
					seq = ""
				header = line.strip()
			else:
				seq += line.strip().upper()
	return sequences


def generate_negative_samples(filename,kmer_length,number_of_samples,positive_list): #make the list of negative samples, 17 nucleotides each
	seqs = parse_fasta(filename)
	x = 0
	while x <= number_of_samples:
		a_key = random.choice(list(seqs))
		temp_seq = seqs[a_key]
		start = random.randrange(0,len(temp_seq)-kmer_length)
		out_string = temp_seq[start:start+14]
		if out_string in positive_list:
			continue
		x += 1
		yield temp_seq[start:start+14]

if __name__ == "__main__":
	
	# print(one_hot_encoding("ATCCN"))

	print(parse_fasta("../data/yeast-upstream-1k-negative.fa"))

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

def negativeSplit2(negative): #this will make an array of the negative sequences
	output = pd.DataFrame() #troubleshooting

	for n in range(len(negative)):
		#split them all into strings that overlap
		length = 17
		sequences = negative
		
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
