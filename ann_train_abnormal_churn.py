import numpy as np
import numpy.lib.recfunctions as nlr
import matplotlib.pyplot as plt
import datetime

from sklearn.utils import shuffle
from process_abnormal_churn import get_data

# Best results so far:
# M = 10
# train on first 8k
# test on last 2k (10k sample)
# tanh/softmax (try relu)
# learning rate = .00001
# 1000 iterations
# Final train classification rate = .966
# Final test classification rate = .971



def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

X, Y = get_data()


X, Y = shuffle(X, Y, random_state=0)
Y = Y.astype(np.int32)


M = 10
D = X.shape[1]
K = len(set(Y))


# train on the last 100 records, test the rest
Xtrain = X[:-2000]
Ytrain = Y[:-2000]
Ytrain_ind = y2indicator(Ytrain, K)
# test
Xtest = X[-2000:]
Ytest = Y[-2000:]
Ytest_ind = y2indicator(Ytest, K)

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def ReLU(a):
	return np.maximum(a,0,a)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2), Z
	#return ReLU(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y,P):
	return np.mean(Y==P)

def cross_entropy(T, pY):
	return -np.mean(T*np.log(pY))

train_costs = []
test_costs = []
learning_rate = .00001
for i in xrange(1000):
	pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
	pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

	ctrain = cross_entropy(Ytrain_ind, pYtrain) # if regression, cost function is squared error instead!
	ctest = cross_entropy(Ytest_ind, pYtest)
	train_costs.append(ctrain)
	test_costs.append(ctest)
	
	W2 -= learning_rate*Ztrain.T.dot(pYtrain - Ytrain_ind)
	b2 -= learning_rate*(pYtrain - Ytrain_ind).sum()
	dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
	W1 -= learning_rate*Xtrain.T.dot(dZ)
	b1 -= learning_rate*dZ.sum(axis=0)
	if i % 100 == 0:
		print i, ctrain, ctest

train_predict_results = predict(pYtrain)
test_predict_results = predict(pYtest)
print "Final train classification_rate:", classification_rate(Ytrain, train_predict_results)
print "Final test classification_rate:", classification_rate(Ytest, test_predict_results)

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test_costs')
plt.legend([legend1, legend2])
plt.show() 

print Xtrain.view()
print Xtest.view()

print Xtrain.shape
print Ytrain.shape
print train_predict_results.shape

# Save results
train_results = np.column_stack([Xtrain, Ytrain, train_predict_results])
test_results = np.column_stack([Xtest, Ytest, test_predict_results])

print train_results.view()
print test_results.view()
print train_results.shape
print test_results.shape

np.savetxt('data/results/train_results' + datetime.date.today().strftime('%Y%m%d') + '.csv', train_results, delimiter=',')
np.savetxt('data/results/test_results' + datetime.date.today().strftime('%Y%m%d') + '.csv', test_results, delimiter=',')

