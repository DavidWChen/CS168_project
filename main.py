from mnist import MNIST
mndata = MNIST('./mnist_data')
for images, labels in mndata.load_training_in_batches(500):
	print(labels[0])
