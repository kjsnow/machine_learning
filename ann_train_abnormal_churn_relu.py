import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from process_abnormal_churn import get_data

# Best results so far:
# M = 5
# train on last 2000
# test on first 8k (10k sample)
# tanh/softmax (try relu)
# learning rate = .0001
# 1000 iterations
# Final train classification rate = .9385
# Final test classification rate = .946



def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

# X, Y = get_data()


# #X, Y = shuffle(X, Y)
# Y = Y.astype(np.int32)


# M = 10 # hidden nodes
# D = X.shape[1] # number of features
# K = len(set(Y)) # number of labels

# batch_size = 1000

# # train on the last 100 records, test the rest
# Xtrain = X[:-2000]
# Ytrain = Y[:-2000]
# Ytrain_ind = y2indicator(Ytrain, K)
# # test
# Xtest = X[-2000:]
# Ytest = Y[-2000:]
# Ytest_ind = y2indicator(Ytest, K)

# #W1 = np.random.randn(D, M)
# #b1 = np.zeros(M)
# #W2 = np.random.randn(M,K)
# #b2 = np.zeros(K)

# def softmax(a):
# 	expA = np.exp(a)
# 	return expA / expA.sum(axis=1, keepdims=True)

# def ReLU(a):
# 	return np.maximum(a,0,a)

# def forward(X, W1, b1, W2, b2):
# 	#Z = np.tanh(X.dot(W1) + b1)
# 	#return softmax(Z.dot(W2) + b2), Z
# 	Z = tf.matmul(X, W1)
# 	return tf.nn.relu(Z, b1), Z

# 	#return ReLU(Z.dot(W2) + b2), Z

# def predict(P_Y_given_X):
# 	return np.argmax(P_Y_given_X, axis=1)

# def classification_rate(Y,P):
# 	return np.mean(Y==P)

# def cross_entropy(T, pY):
# 	return -np.mean(T*np.log(pY))

# train_costs = []
# test_costs = []
# learning_rate = .00001
# for i in xrange(1000):
# 	pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
# 	pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

# 	ctrain = cross_entropy(Ytrain_ind, pYtrain) # if regression, cost function is squared error instead!
# 	ctest = cross_entropy(Ytest_ind, pYtest)
# 	train_costs.append(ctrain)
# 	test_costs.append(ctest)
	
# 	W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
# 	b2 -= learning_rate*(pYtrain - Ytrain_ind).sum()
# 	dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
# 	W1 -= learning_rate*Xtrain.T.dot(dZ)
# 	b1 -= learning_rate*dZ.sum(axis=0)
# 	if i % 100 == 0:
# 		print i, ctrain, ctest

# print "Final train classification_rate:", classification_rate(Ytrain, predict(pYtrain))
# print "Final test classification_rate:", classification_rate(Ytest, predict(pYtest))

# legend1, = plt.plot(train_costs, label='train cost')
# legend2, = plt.plot(test_costs, label='test_costs')
# plt.legend([legend1, legend2])
# plt.show() 


#################

# Tensor Flow ReLU

# np.random.seed(12)
# num_observations = 5000

# x1 = np.random.multivariate_normal([0, 0], [[2, .75],[.75, 2]], num_observations)
# x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)
# x3 = np.random.multivariate_normal([2, 8], [[0, .75],[.75, 0]], num_observations)

# simulated_separableish_features = np.vstack((x1, x2, x3)).astype(np.float32)
# simulated_labels = np.hstack((np.zeros(num_observations),
# 				np.ones(num_observations), np.ones(num_observations) + 1))

# labels_onehot = np.zeros((simulated_labels.shape[0], 3)).astype(int)
# labels_onehot[np.arange(len(simulated_labels)), simulated_labels.astype(int)] = 1

# Xtrain2, Xtest2, \
# Ytrain2, Ytest2 = train_test_split(
#     simulated_separableish_features, labels_onehot, test_size = .1, random_state = 12)

# print Xtrain2.view()
# print Xtest2.view()
# print Ytrain2.view()
# print Ytest2.view()


# Pull data
X, Y = get_data()

Y = Y.astype(np.int32)

X[:,3] = (X[:,3] - X[:,3].mean()) / X[:,3].std()
X[:,4] = (X[:,4] - X[:,4].mean()) / X[:,4].std()

# train on the last 100 records, test the rest
#Xtrain = X[:-2000]
#Ytrain = Y[:-2000]
#Ytrain_ind = y2indicator(Ytrain, K)
# test
#Xtest = X[-2000:]
#Ytest = Y[-2000:]
#Ytest_ind = y2indicator(Ytest, K)

labels_onehot = np.zeros((Y.shape[0], 2)).astype(int)
labels_onehot[np.arange(len(Y)), Y.astype(int)] = 1


Xtrain, Xtest, \
Ytrain, Ytest = train_test_split(
    X, labels_onehot, test_size = .2, random_state = 12)



print Xtrain.view()
print Xtest.view()
print Ytrain.view()
print Ytest.view()

################################################

M = 10 # hidden nodes
D = Xtrain.shape[1] # number of features
#K = len(set(Y)) # number of labels
K = Ytrain.shape[1]
learning_rate = .0001

batch_size = 1000

graph = tf.Graph()
with graph.as_default():
	
	tf_train_dataset = tf.placeholder(tf.float32, shape = [None, D])
	tf_train_labels = tf.placeholder(tf.float32, shape =[None, K])
	tf_test_dataset = tf.constant(Xtest)

	W1 = tf.Variable(tf.truncated_normal([D, M]))
	b1 = tf.Variable(tf.zeros([M]))

	W2 = tf.Variable(tf.truncated_normal([M, K]))
	b2 = tf.Variable(tf.zeros([K]))

	# Three layer network
	def three_layer_network(data):
		data = tf.cast(data, tf.float32)
		input_layer = tf.matmul(data, W1)
		hidden = tf.nn.relu(input_layer + b1)
		output_layer = tf.matmul(hidden, W2) + b2
		return output_layer

	# Model Scores
	model_scores = three_layer_network(tf_train_dataset)

	# Loss
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(model_scores, tf_train_labels))

	# Optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# Predictions
	train_prediction = tf.nn.softmax(model_scores)
	test_prediction = tf.nn.softmax(three_layer_network(tf_test_dataset))	

# combo of predict and classification functions?
def accuracy(predictions, labels):
	preds_correct_boolean = np.argmax(predictions, 1) == np.argmax(labels, 1)
	correct_predictions = np.sum(preds_correct_boolean)
	accuracy = 100.0 * correct_predictions / predictions.shape[0]
	return accuracy

num_steps = 50000

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	for i in range(num_steps):
		offset = (i * batch_size) % (Ytrain.shape[0] - batch_size)
		minibatch_data = Xtrain[offset:(offset + batch_size),:]
		minibatch_labels = Ytrain[offset:(offset + batch_size)]

		feed_dict = {tf_train_dataset : minibatch_data, tf_train_labels : minibatch_labels}

		_, l, predictions = session.run(
			[optimizer, loss, train_prediction], feed_dict = feed_dict
			)

		if i % 100 == 0:
			print 'Minibatch loss at iteration {0}: {1}'.format(i, l)

	print 'Test accuracy: {0}%'.format(accuracy(test_prediction.eval(), Ytest))

