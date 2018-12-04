import numpy as np
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def accuracy(k):
	clf = KNeighborsClassifier(n_neighbors=k)
	clf.fit(x_train, y_train)
	return clf.score(x_test, y_test)

def accuracy2():
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(x_train, y_train)
	count=np.zeros(10)
	right=np.zeros(10)
	pred = clf.predict(x_test)
	for y_pre, y in zip(pred, y_test): 
		count[y] += 1
		if y_pre == y:
			right[y] += 1
	return [right[0]/count[0], right[1]/count[1], right[2]/count[2], 
			right[3]/count[3], right[4]/count[4], right[5]/count[5],
			right[6]/count[6], right[7]/count[7],
			right[8]/count[8], right[9]/count[9] ]


digits = load_digits()
X = digits.data
Y = digits.target
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


#print(len(Y))
k = np.arange(1, 50)
myAccuracy = []
for i in range(1, 50):
	myAccuracy.append(accuracy(i))
plt.plot(k, myAccuracy, marker='o') # best, k=3
plt.xlabel('num of K')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
'''

num = np.arange(0, 10)
myAccuracy2 = accuracy2()
plt.bar(num, myAccuracy2)
plt.ylim(0.8, 1) 
plt.xlabel('Digit')
plt.ylabel('Accuracy')
plt.show()
'''
