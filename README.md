[![Build
Status](https://travis-ci.org/zach-hois/NeuralNetwork.svg?branch=master)](https://travis-ci.org/zach-hois/NeuralNetwork)

Example python project with testing.

## usage

To use the package, first make a new conda environment and activate it

```
conda create -n exampleenv python=3
source activate exampleenv
```

then run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `example/__main__.py`) can be run as follows

```
python -m scripts
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.


## 3/2/2020 in class notes
task: have a fully connected neural network predict the outcome of a task

# 1. Data
	1. Reverse complementation
	2. Substring
	3. Check negative strings for upstring positives
# 2. Representation
	1. Vector for each sequence that is 4 x Seq Length. (one hot, codons, 'word2vec')

Points basis is to write an auto-encoder and talk through training and testing 

# 3. Auto-encoder
	1. You have an image, and the input is the same as the output, use a hidden bottleneck that has minimal loss. Reconstruct the output in the same image as the input. (e.g. dimensionality reduction) 8 input neurons, 3 bottleneck, 8 output

Show effectiveness on (minimize the difference between input and output):
10000000
01000000
00100000
00010000
00001000
00000100
00000010
00000001

# 4. Evaluation

1) How will I evaluate my training function? (loss function)
2) What is my dataset made of? Is my data set balanced?
3) What am I holding out? What am I keeping in? How do I pick my model?

K folds verification



