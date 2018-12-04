import numpy as np
import csv
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def accuracy(k):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(x_train, y_train)
	return clf.score(x_test, y_test)

def accuracy2(x_digit, label):
	clf = KNeighborsClassifier(n_neighbors=4)
	clf.fit(x_train, y_train)
	pred = clf.predict(x_digit)
	count = len(x_digit)
	right = 0
	for y_pre in pred : 
		if y_pre == label:
			right += 1
	return right/count

def loadOneDigit(name):
	x = []
	label = int(name[0])
	with open(name) as f:
		reader = csv.reader(f)
		for row in reader:
			if reader.line_num == 1:
				continue
			x.append(np.array(row[1:], dtype=int))
	return x, label
	
def loadData(name):
	x = []
	y = []
	with open(name) as f:
		reader = csv.reader(f)
		for row in reader:
			if reader.line_num == 1:
				continue
			x.append(np.array(row[2:], dtype=int))
			y.append(int(row[1]))
	return x, y

x_train, y_train = loadData("trainData.csv")
x_test, y_test = loadData("testData.csv")


# draw k-distribution diagram
'''
k = np.arange(1, 50)
myAccuracy = []
for i in range(1, 50):
	myAccuracy.append(accuracy(i))
plt.plot(k, myAccuracy, marker='o') # best, k=4, 95.6%
plt.xlabel('num of K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
'''


# draw digit-disrtibution diagram
num = np.arange(0, 10)
myAccuracy2 = []
for i in range(0, 10): 
	x , label = loadOneDigit(str(i)+".csv")
	myAccuracy2.append(accuracy2(x, label))
plt.bar(num, myAccuracy2)
plt.ylim(0.8, 1) 
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.show()

