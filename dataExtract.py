import numpy as np
import csv
import random
from pandas import DataFrame

# 0~9 test data, 200 each
'''
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []
x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]
with open('train.csv') as f:
	reader = csv.reader(f)
	count=np.zeros(10)
	for row in reader:
		if reader.line_num == 1:
			continue
		label = int(row[0])
		count[label] += 1
		if count[int(row[0])] > 200 :
			continue
		x[label].append(np.array(row[1:], dtype=int))  # no label

for i in range(0, 10):
	df = DataFrame(x[i])
	df.to_csv(str(i)+".csv")   # name shows label
'''	


# all test data, 2000 total
'''
x_test = []
with open('train.csv') as f:
	reader = csv.reader(f)
	count=np.zeros(10)
	for row in reader:
		if reader.line_num == 1:
			continue
		label = int(row[0])
		count[label] += 1
		if count[int(row[0])] > 200 :
			continue
		x_test.append(np.array(row[0:], dtype=int))  # column[1] is label 

df = DataFrame(x_test)
df.to_csv('testData.csv')
'''

# all training data, 1000 total, 1000 each
count = np.zeros(10)
x_train = []
with open('train.csv') as f:
	reader = csv.reader(f)
	count=np.zeros(10)
	for row in reader:
		if reader.line_num < 10000:
			continue
		label = int(row[0])
		count[label] += 1
		if count[int(row[0])] > 1000 :
			continue
		x_train.append(np.array(row[0:], dtype=int))  # column[1] is label 

df = DataFrame(x_train)
df.to_csv('trainData.csv')
