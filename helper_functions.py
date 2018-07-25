import sql_credentials as creds
import pandas as pd
import numpy as np
import pymssql
import datetime as dt
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

class Hyperparameters:
	def __init__(self, experiment, query, col_list, num_nodes, num_iterations, learning_rate, last_day_testing, storage_path, to_query, save_results, raw_file_path):
		self.experiment = experiment
		self.query = query
		self.col_list = col_list
		self.num_nodes = num_nodes
		self.num_iterations = num_iterations
		self.learning_rate = learning_rate
		self.last_day_testing = last_day_testing
		self.storage_path = storage_path
		self.to_query = to_query
		self.save_results = save_results
		self.raw_file_path = raw_file_path


class Inputs:
	def __init__(self, IDtrain, Xtrain, Ytrain, IDtest, Xtest, Ytest):
		self.IDtrain = IDtrain
		self.Xtrain = Xtrain
		self.Ytrain = Ytrain
		self.IDtest = IDtest
		self.Xtest = Xtest
		self.Ytest = Ytest


def fetch_data(params, credentials):
	if params.to_query:
		print 'query starting'
		results = pd.DataFrame(sql_query(params.query, credentials))
		# ensure correct order
		results = results[params.col_list]
		results.to_csv('%s/raw/' %(params.storage_path) + params.experiment + '_' + dt.date.today().strftime('%Y%m%d') + '.csv', index=False, header=True, sep=',')
	else:
		print 'read file starting'
		results = pd.read_csv(params.raw_file_path, header=0, names=params.col_list, parse_dates=['churn_date'])
	results.columns = params.col_list
	results['churn_date'] = pd.to_datetime(results['churn_date'])
	#results = results.as_matrix()
	return results

def sql_query(query, credentials):
	conn = pymssql.connect(server=credentials.server, user=credentials.user, password=credentials.pwd, database=credentials.db)
	#conn = pymssql.connect(server='174.34.53.73', user='ksnow', password='Kk5N03!w', database='com_cdm_stage')
	cursor = conn.cursor(as_dict=True)
	# Run query
	cursor.execute(query)
	# Fetch results
	row = cursor.fetchall() 
	# close connection
	conn.close()
	return row

def shuffle_split(df, last_day_testing=True):
	if last_day_testing:
		train_set = df[(df['churn_date'] < (dt.datetime.now() - dt.timedelta(days=3)))]
		test_set = df[(df['churn_date'] >= (dt.datetime.now() - dt.timedelta(days=3)))]
		IDtrain = train_set.iloc[:,0:2]
		Xtrain = train_set.iloc[:, 2:-1]
		Ytrain = train_set.iloc[:, -1]
		IDtest = test_set.iloc[:,0:2]
		Xtest = test_set.iloc[:, 2:-1]
		Ytest = test_set.iloc[:, -1]
	else:
		ID = df[:,0:2]
		X = df[:2:-1]
		Y = df[:,-1]
		IDtrain, IDtest, \
		Xtrain, Xtest, \
		Ytrain, Ytest = train_test_split(ID, X, Y, test_size = .2, random_state = 12)
	Ytrain = Ytrain.astype(np.int32)
	Ytest = Ytest.astype(np.int32)
	return Inputs(IDtrain, Xtrain, Ytrain, IDtest, Xtest, Ytest)

def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y.iloc[i]] = 1
	return ind

def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(np.array(X.dot(W1) + b1, dtype=np.float32))
	#Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2), Z
	#return ReLU(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y,P):
	return np.mean(Y==P)

def cross_entropy(T, pY):
	return -np.mean(T*np.log(pY))

def ann(params, inputs):
	D = inputs.Xtrain.shape[1] # number of features
	K = len(set(inputs.Ytrain)) # number of classifications

	inputs.Xtrain = np.array(inputs.Xtrain, dtype=np.float32)

	Ytrain_ind = y2indicator(inputs.Ytrain, K)
	Ytest_ind = y2indicator(inputs.Ytest, K)
	
	# set seed for same result
	np.random.seed(1)

	# Randomly initialize weights
	W1 = np.random.randn(D, params.num_nodes)
	b1 = np.zeros(params.num_nodes)
	W2 = np.random.randn(params.num_nodes,K)
	b2 = np.zeros(K)

	# Initialize cost array and learning_rate
	train_costs = []
	test_costs = []

	for i in xrange(params.num_iterations):
		pYtrain, Ztrain = forward(inputs.Xtrain, W1, b1, W2, b2)
		pYtest, Ztest = forward(inputs.Xtest, W1, b1, W2, b2)

		ctrain = cross_entropy(Ytrain_ind, pYtrain) # if regression, cost function is squared error instead!
		ctest = cross_entropy(Ytest_ind, pYtest)
		train_costs.append(ctrain)
		test_costs.append(ctest)
	
		W2 -= params.learning_rate * Ztrain.T.dot(pYtrain - Ytrain_ind)
		b2 -= params.learning_rate * (pYtrain - Ytrain_ind).sum()
		dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain*Ztrain)
		W1 -= params.learning_rate * inputs.Xtrain.T.dot(dZ)
		b1 -= params.learning_rate * dZ.sum(axis=0)
		if i % 100 == 0:
			print i, ctrain, ctest

	train_predict_results = predict(pYtrain)
	test_predict_results = predict(pYtest)
	print "Final train classification_rate:", classification_rate(inputs.Ytrain, train_predict_results)
	print "Final test classification_rate:", classification_rate(inputs.Ytest, test_predict_results)
	print confusion_matrix(inputs.Ytrain, train_predict_results)
	tn, fp, fn, tp = confusion_matrix(inputs.Ytrain, train_predict_results).ravel()

	legend1, = plt.plot(train_costs, label='train cost')
	legend2, = plt.plot(test_costs, label='test_costs')
	plt.legend([legend1, legend2])
	plt.show() 
	if params.save_results:
		plt.savefig('{0}/data/results/{1}_cost_'.format(params.storage_path, params.experiment) + dt.date.today().strftime('%Y%m%d') + '.png')

	# Combine results back to tn list for analysis
	params.col_list.append('pred_noncomp_ind')

	train_results = pd.DataFrame(np.column_stack((inputs.IDtrain, inputs.Xtrain, inputs.Ytrain, train_predict_results)), columns=params.col_list)
	test_results = pd.DataFrame(np.column_stack((inputs.IDtest, inputs.Xtest, inputs.Ytest, test_predict_results)), columns=params.col_list)

	write_results(train_results, test_results, params.storage_path, params.experiment, params.save_results)

def write_results(train_results, test_results, storage_path, experiment, save_results=True):
		# Save results
	if save_results:
		train_results.to_csv('{0}/data/results/{1}_train_results'.format(storage_path, experiment) + dt.date.today().strftime('%Y%m%d') + '.csv', index=False, header=True, sep=',')
		test_results.to_csv('{0}/data/results/{1}_test_results'.format(storage_path, experiment) + dt.date.today().strftime('%Y%m%d') + '.csv', index=False, header=True, sep=',')

