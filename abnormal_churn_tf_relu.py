import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from process_abnormal_churn import get_data



def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind


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

# Best hyperparams so far:
#M = 20
#learning_rate = .01
#batch_size = 1000
#num_steps = 10000
# Test accuracy: 97.45%

M = 20 # hidden nodes
D = Xtrain.shape[1] # number of features
#K = len(set(Y)) # number of labels
K = Ytrain.shape[1]
learning_rate = .01

batch_size = 500

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

num_steps = 10000

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

