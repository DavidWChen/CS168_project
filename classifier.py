from sklearn import svm, neighbors, neural_network, multiclass
import numpy as np
from scipy.io import loadmat
from mnist import MNIST


def loadSpiral():
	data = loadmat('twoSpirals.mat')
	dataX = data['X']
	dataY = data['Y']
	with open('./mdata.txt', 'r') as f:
		modelX=np.loadtxt(f)
	with open('./mlabel.txt', 'r') as g:
		modelY=np.loadtxt(g)
	return modelX, modelY, dataX, dataY

def loadMNIST():
	with open('./mndata.txt', 'r') as f:
		modelX=np.loadtxt(f)
	with open('./mnlabel.txt', 'r') as g:
		modelY=np.loadtxt(g)
	mndata = MNIST('./mnist_data')
	images, labels = mndata.load_testing()
	return modelX, modelY, images, labels

def loadEMNIST():
	with open('./emndata.txt', 'r') as f:
		modelX=np.loadtxt(f)
	with open('./emnlabel.txt', 'r') as g:
		modelY=np.loadtxt(g)
	mndata = MNIST('./emnist_data')
	mndata.select_emnist('digits')
	images, labels = mndata.load_testing()
	return modelX, modelY, images, labels

def testClf(X,Y, testX, testY, clf):
	
	testY = testY.tolist()
	clf.fit(X,Y)
	result = clf.predict(testX)

	print(clf.score(testX,testY))


if __name__ =='__main__':
	# #Spiral
	# modelX, modelY, dataX, dataY = loadSpiral()
	# testClf(modelX, modelY, dataX, dataY, svm.SVC(gamma='scale'))
	# testClf(modelX, modelY, dataX, dataY, neighbors.KNeighborsClassifier(n_neighbors=1))
	# testClf(modelX, modelY, dataX, dataY, neural_network.MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(2), random_state=1))
	

	# #NMIST
	# modelX, modelY, images, labels = loadMNIST()
	# testClf(modelX, modelY, images, labels, svm.SVC(gamma='scale'))
	# testClf(modelX, modelY, images, labels, neighbors.KNeighborsClassifier(n_neighbors=1))
	# testClf(modelX, modelY, images, labels, neural_network.MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(2), random_state=1))
	# testClf(modelX, modelY, images, labels, multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=0)))	

	#ENMIST
	modelX, modelY, images, labels = loadEMNIST()
	testClf(modelX, modelY, images, labels, svm.SVC(gamma='scale'))
	testClf(modelX, modelY, images, labels, neighbors.KNeighborsClassifier(n_neighbors=1))
	testClf(modelX, modelY, images, labels, neural_network.MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(2), random_state=1))
	testClf(modelX, modelY, images, labels, multiclass.OneVsRestClassifier(svm.LinearSVC(random_state=0)))	


