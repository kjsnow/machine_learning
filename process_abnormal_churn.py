import numpy as np
import pandas as pd
import os

from sklearn.utils import shuffle

# def get_data():
# 	df = pd.read_csv('~/Documents/Classes/udemy_deep_learning/sample_disconnects.csv', header=None)
# 	#df = df.as_matrix()
# 	df.columns = ['vzw_ind', 'att_ind', 'spr_ind', 'tmo_ind', 'met_ind', 'churn_date', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']
	
# 	df = df.sort_values('sec_from_midnight').reset_index(drop=True)

# 	#dfx = df.iloc[:,:-1]
	
# 	#dfy = df.iloc[:,-1]
	

# 	seconds_in_day = 24*60*60

# 	df['sin_time'] = np.sin(2*np.pi*df.sec_from_midnight/seconds_in_day)
# 	df['cos_time'] = np.cos(2*np.pi*df.sec_from_midnight/seconds_in_day)

# 	del df['churn_date']
# 	del df['sec_from_midnight']

# 	df = df.as_matrix()

# 	dfx = df[:, :7]
# 	dfy = df[:, 7]

# 	return dfx, dfy


def get_data():
	
	#cwd = os.getcwd()
	
	#df = pd.read_csv('~/Documents/Classes/udemy_deep_learning/machine_learning_examples/ann_logistic_extra/ecommerce_data.csv')
	#df = pd.read_csv('~/Documents/Classes/udemy_deep_learning/disconnects_sample_neural_network.csv')
	
	# path within machine learning repo
	df = pd.read_csv('data/disconnects_sample_neural_network.csv')
	data = df.as_matrix()

	#random.seed(420)
	#sample = np.random.choice(data, 1000)
	sample = data[np.random.choice(data.shape[0], 10000, replace=False), :]

	sample = shuffle(sample)

	X = sample[:, :-1]
	Y = sample[:, -1]

	#X = data[:, :-1]
	#Y = data[:, -1]

	#X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
	#X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
	X[:,3] = (X[:,3] - X[:,3].mean()) / X[:,3].std()
	X[:,4] = (X[:,4] - X[:,4].mean()) / X[:,4].std()

	N, D = X.shape
	X2 = np.zeros((N, D+4))
	X2[:,0:(D-1)] = X[:,0:(D-1)]

	# one hot encoding
	for n in xrange(N):
		t = int(X[n,0])
		X2[n,t+D-1] = 1

	# delete first column now
	#X2 = X2[:,1:]

	#Z = np.zeros((N, 5))
	#Z[np.arange(N), X[:,D-1].astype(np.int32)] = 1
	#assert(np.abs(X2[:,-5:] - Z).sum() < 10e-10)


	return X2, Y

def get_binary_data():
	X, Y = get_data()
	X2 = X[Y <= 1] #only take classes 0 or 1
	Y2 = Y[Y <= 1]
	return X2, Y2