import numpy as np
import pandas as pd
import os
import pymssql
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from process_abnormal_churn import get_data

server='174.34.53.73'
db = 'com_cdm_stage'
user = 'ksnow'
pwd = 'Kk5N03!w'

def sql_query(server=server, user=user, pwd=pwd, db=db, query=''):
	conn = pymssql.connect(server=server, user=user, password=pwd, database=db)
	cursor = conn.cursor(as_dict=True)
	# Run query
	cursor.execute(query)
	# Fetch results
	row = cursor.fetchall() 
	# close connection
	conn.close()
	return row

def fetch_data():

	#, (DATEPART(HOUR, churn_date) * 3600 + DATEPART(MINUTE, churn_date) * 60 + DATEPART(SECOND, churn_date)) as sec_from_midnight
	query = 'select case when churn_losing_sp = 1 then 1 else 0 end as is_vzw, case when churn_losing_sp = 2 then 1 else 0 end as is_att, case when churn_losing_sp = 3 then 1 else 0 end as is_spr, case when churn_losing_sp = 4 then 1 else 0 end as is_tmo, case when churn_losing_sp = 6 then 1 else 0 end as is_met, case when datepart(hour, churn_date) >= 21 or datepart(hour, churn_date) < 9 then 1 else 0 end as night_ind, case when datepart(minute, churn_date) % 5 = 0 and datepart(second, churn_date) = 0 then 1 else 0 end as five_min_ind, case when noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind from com_cdm_stage..f_final_churn_reporting where churn_date >= \'2018-06-04\' and churn_date < \'2018-06-05\' and churn_losing_sp in (1,2,3,4,6) and churn_type like \'%DSV\' and noncompetitive_ind in (0,5)'

	results = pd.DataFrame(sql_query(query=query))

	#col_names = ['is_vzw', 'is_att', 'is_spr', 'is_tmo', 'is_met', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']

	# reorder columns
	#results = results[col_names]

	results = results.as_matrix()
	results = shuffle(results)

	X = results[:, :-1]
	Y = results[:, -1]

	return X, Y

def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y[i]] = 1
	return ind

X, Y = fetch_data()
X2, Y2 = get_data()

Y = Y.astype(np.int32)
Y2 = Y2.astype(np.int32)

print X.shape
print X2.shape
print Y.shape
print Y2.shape

M = 10 # hidden nodes
D = X.shape[1] # number of features
K = len(set(Y))


Xtrain, Xtest, \
Ytrain, Ytest = train_test_split(
    X, Y, test_size = .2, random_state = 12)

Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

# print Xtrain.view()
# print Xtest.view()
# print Ytrain_ind.view()
# print Ytest_ind.view()



W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)



def softmax(a):
	expA = np.exp(a)
	return expA / expA.sum(axis=1, keepdims=True)

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


# Save results
#train_results = np.column_stack([Xtrain, Ytrain, train_predict_results])
#test_results = np.column_stack([Xtest, Ytest, test_predict_results])


#np.savetxt('data/results/train_results' + datetime.date.today().strftime('%Y%m%d') + '.csv', train_results, delimiter=',')
#np.savetxt('data/results/test_results' + datetime.date.today().strftime('%Y%m%d') + '.csv', test_results, delimiter=',')

