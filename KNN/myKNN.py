import numpy as np
import csv
import random
import math
import statistics
from pandas import Series

def matrixDistance(x1, x2):
	l, d = x1.shape
	sum = 0
	for i in range(l):
		for j in range(d):
			if x1[i][j]==0 and x2[i][j]==0:
				continue;
			sum +=  (x1[i][j]-x2[i][j])**2 
	return math.sqrt(sum)
	
	
def bubbleSort(y, len):
	for i in range(len-1):
		sorted = False
		for j in range(len-1):
			if list(y[j].keys())[0] > list(y[j+1].keys())[0]:
				tempy = y[j]
				y[j] = y[j+1]
				y[j+1] = tempy
				sorted = True
		if sorted == False:
			break	
	return ;


def identify(x, k):  # k means kNN
	sum = 0
	min = 999999999  # just a very big number
	target = 0 
	for i in range(length):
		sum += matrixDistance(x, X[i])
		if (i+1) % k == 0:
			if sum < min:
				min = sum
				target = Y[i-5]
			sum = 0
	return target


def KNN(x, k):
	ls = []   # k elements
	for i in range(99999999, 99999999+k):
		ls.append({i:-1})
	for i in range(length):
		distance = matrixDistance(x, X[i])
		if distance < list(ls[k-1].keys())[0]:
			ls[k-1] = {distance: Y[i]}
			bubbleSort(ls, len(ls))
	output = []
	for ele in ls:
		output.append(list(ele.values())[0])
	return max(output, key=output.count)    # get the mode
	
X = []   # 1000 for dataset capacity
Y = []
X_test = []
Y_test = []
with open('train.csv') as f:
	reader = csv.reader(f)
	i = 0
	for row in reader:
		if reader.line_num == 1:
			continue;
		if i < 9999 :
			Y.append(int(row[0]))
			X.append(np.array(row[1:], dtype=int).reshape(28,28))
			i += 1
		if i >= 9999 :
			Y_test.append(int(row[0]))
			X_test.append(np.array(row[1:], dtype=int).reshape(28,28))
			i += 1
		if i > 11000 :
			break;

length = len(Y)

for i in range(50,53):
	print("predict: ", KNN(X_test[i],30), "actual: ", Y_test[i])


'''quickSort(Y, X, 0, length-1)
count = 0
for i in range(0,200):
	if identify(X_test[i],10) == Y_test[i]:
		count += 1
print(count/200) '''






