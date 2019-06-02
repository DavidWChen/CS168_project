from scipy.io import loadmat
import incrementalLearn
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST



def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def plotSpiral(dataX, dataY, model):
	iy0 = indices(dataY, lambda x: x == 0)
	iy1 = indices(dataY, lambda x: x == 1)

	ix0 = indices(model.Y, lambda x: x == 0)
	ix1 = indices(model.Y, lambda x: x == 1)

	dataX0 = dataX[iy0]
	dataX1 = dataX[iy1]
	modelX0 = model.X[ix0]
	modelX1 = model.X[ix1]

	plt.scatter(dataX1[:,0],dataX1[:,1], color='green')
	plt.scatter(modelX1[:,0],modelX1[:,1], marker='x', color='blue')

	plt.scatter(dataX0[:,0],dataX0[:,1], color='orange')
	plt.scatter(modelX0[:,0],modelX0[:,1], marker='x', color='red')

	plt.show()

def mainSpiral():
	data = loadmat('twoSpirals.mat')
	dataX = data['X']
	dataY = data['Y']

	model = incrementalLearn.incrementalLearn(dataX, dataY, {})
	plotSpiral(dataX, dataY, model)
	# np.savetxt('mdata.txt',model.X,fmt='%.2f')
	# np.savetxt('mlabel.txt',model.Y,fmt='%.2f')
	

def mainMNIST():
	mndata = MNIST('./mnist_data')
	images, labels = mndata.load_training()
	images = np.array(images)

	model = incrementalLearn.incrementalLearn(images, labels, {})
	# np.savetxt('mndata.txt',model.X,fmt='%.2f')
	# np.savetxt('mnlabel.txt',model.Y,fmt='%.2f')


if __name__ =='__main__':
	# mainSpiral()
	mainMNIST()





