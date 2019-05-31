# from mnist import MNIST
# mndata = MNIST('./mnist_data')
# for images, labels in mndata.load_training_in_batches(500):
# 	print(labels[0])


from scipy.io import loadmat
import incrementalLearn

data = loadmat('twoSpirals.mat')
dataX = data['X']
dataY = data['Y']
model = incrementalLearn.incrementalLearn(dataX, dataY, {})

print(model.X, model.Y)


