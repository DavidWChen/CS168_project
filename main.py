# from mnist import MNIST
# mndata = MNIST('./mnist_data')
# for images, labels in mndata.load_training_in_batches(500):
# 	print(labels[0])


from scipy.io import loadmat
import incrementalLearn
import matplotlib.pyplot as plt

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

data = loadmat('twoSpirals.mat')
dataX = data['X']
dataY = data['Y']
model = incrementalLearn.incrementalLearn(dataX, dataY, {})

# print(model.X, model.Y)
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



# function testSpiral2()

# load('./twoSpirals.mat');

# % not displaying the label Y 
# plot(X(:,1), X(:,2), '.');  hold on;

# % displaying the label Y 
# %for i=1:length(X)
# %    plot(X(i,1), X(i,2), '.', 'Color', [Y(i)./2  Y(i)./2 1]); hold on;
# %end

# options.t = 1;

# idx = randperm(numel(Y));
# X = X(idx,:);
# Y = Y(idx,:);

# [model] = incrementalLearn(X, Y, options);

# for i=1:25
#     plot(model.X(i,1), model.X(i,2), 'rx'); hold on;
# end