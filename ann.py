import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

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

def accuracyOneLayer(hidden):
	clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(hidden))
	clf.fit(x_train,y_train)
	return clf.score(x_test, y_test)

# principle lower > upper
def accuracyTwoLayer(lower, upper):
	clf = MLPClassifier(activation='logistic', hidden_layer_sizes=(lower, upper))
	clf.fit(x_train,y_train)
	return clf.score(x_test, y_test)


x_train, y_train = loadData("trainData.csv")
x_test, y_test = loadData("testData.csv")

'''
acc1 = []
for i in range(5, 100):
	acc1.append(accuracyOneLayer(i))
nodes = np.arange(5, 100)
plt.plot(nodes, acc1, marker='o')
plt.xlabel('num of hidden nodes')  # 49 is ok 
plt.ylabel('Accuracy')
plt.title('One-hidden-layer ANN')
plt.grid(True)
plt.show()
'''

'''
acc2 = []
for i in range(5, 100):
	acc2.append(accuracyTwoLayer(i, int(0.333*i)))
nodes = np.arange(5, 100)
plt.plot(nodes, acc2, marker='o')
plt.xlabel('num of lower-layer hidden nodes (upper-layer is its 1/3)')
plt.ylabel('Accuracy')
plt.title('Two-hidden-layer ANN')
plt.grid(True)
plt.show()
'''

mlp = MLPClassifier(activation='logistic', hidden_layer_sizes=(49))
param_grid = {'solver': ['sgd', 'adam'],
              'alpha': [0.0001, 0.001, 0.01],   
              'learning_rate': ['constant', 'adaptive'],             
              'learning_rate_init': [0.0001, 0.001, 0.1 ], 
              'tol': [0.0001, 0.001, 0.005],
              'momentum': [0.9, 0.8, 0.7, 0.6]
             }
gs = GridSearchCV(mlp, param_grid, cv=10)
gs.fit(x_train, y_train)
#gs.grid_scores_   # is a 2-d list: [0]-param [1]-mean

maxMean = 0
param = {}
for ele in gs.grid_scores_:
    if ele[1] > maxMean:
        maxMean = ele[1]
        param = ele[0]
print(maxMean, param)


