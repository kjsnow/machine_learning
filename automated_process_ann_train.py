import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import helper_functions as hf
import datetime as dt

from sklearn.model_selection import train_test_split
from process_abnormal_churn import get_data


def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N,K))
	for i in xrange(N):
		ind[i, y[i]] = 1
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



def main():

	server='174.34.53.73'
	db = 'com_cdm_stage'
	user = 'ksnow'
	pwd = 'Kk5N03!w'

	start_date = dt.date(2018,6,10)
	end_date = dt.date(2018,6,13)

	#, (DATEPART(HOUR, churn_date) * 3600 + DATEPART(MINUTE, churn_date) * 60 + DATEPART(SECOND, churn_date)) as sec_from_midnight
	# query = 'select tn, case when churn_losing_sp = 1 then 1 else 0 end as is_vzw, case when churn_losing_sp = 2 then 1 else 0 end as is_att, case when churn_losing_sp = 3 then 1 else 0 end as is_spr, case when churn_losing_sp = 4 then 1 else 0 end as is_tmo, case when churn_losing_sp = 6 then 1 else 0 end as is_met, case when datepart(hour, churn_date) >= 21 or datepart(hour, churn_date) < 9 then 1 else 0 end as night_ind, case when datepart(minute, churn_date) % 5 = 0 and datepart(second, churn_date) = 0 then 1 else 0 end as five_min_ind, case when noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind from com_cdm_stage..f_final_churn_reporting where churn_date >= \'2018-06-04\' and churn_date < \'2018-06-05\' and churn_losing_sp in (1,2,3,4,6) and churn_type like \'%DSV\' and noncompetitive_ind in (0,5)'

	# (DATEPART(HOUR, a.churn_date) * 3600 + DATEPART(MINUTE, a.churn_date) * 60 + DATEPART(SECOND, a.churn_date)) as sec_from_midnight,

	# Make sure to update when changing queries
	col_list = ['tn', 'is_vzw', 'is_att', 'is_spr', 'is_tmo', 'is_met', 'night_ind', 'five_min_ind', 'count_same_minute', 'count_same_second', 'noncomp_ind']

	query = 'with cte as (select tn, churn_losing_sp, churn_date, noncompetitive_ind from com_cdm_stage..f_final_churn_reporting where churn_date >= \'2018-06-10\' and churn_date < \'2018-06-13\' and churn_losing_sp in (1,2,3,4,6) and churn_type like \'%DSV\' and noncompetitive_ind in (0,5)),cte_minute as (select churn_losing_sp, datepart(day, churn_date) as cday, datepart(hour,churn_date) as chour, datepart(minute,churn_date) as cminute, count(1) as count_same_minute from cte group by churn_losing_sp, datepart(day, churn_date), datepart(hour,churn_date), datepart(minute,churn_date)),cte_second as (select churn_losing_sp, datepart(day, churn_date) as cday, datepart(hour,churn_date) as chour, datepart(minute,churn_date) as cminute, datepart(second,churn_date) as csecond, count(1) as count_same_second from cte group by churn_losing_sp, datepart(day, churn_date), datepart(hour,churn_date), datepart(minute,churn_date), datepart(second,churn_date)) select a.tn,case when a.churn_losing_sp = 1 then 1 else 0 end as is_vzw, case when a.churn_losing_sp = 2 then 1 else 0 end as is_att, case when a.churn_losing_sp = 3 then 1 else 0 end as is_spr, case when a.churn_losing_sp = 4 then 1 else 0 end as is_tmo, case when a.churn_losing_sp = 6 then 1 else 0 end as is_met,case when datepart(hour, a.churn_date) >= 21 or datepart(hour, a.churn_date) < 9 then 1 else 0 end as night_ind,case when datepart(minute, a.churn_date) % 5 = 0 and datepart(second, a.churn_date) = 0 then 1 else 0 end as five_min_ind,m.count_same_minute,s.count_same_second,case when a.noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind from cte a inner join cte_minute m on a.CHURN_LOSING_SP = m.CHURN_LOSING_SP and datepart(day, a.churn_date) = m.cday and datepart(hour,a.churn_date) = m.chour and datepart(minute,a.churn_date) = m.cminute inner join cte_second s on a.CHURN_LOSING_SP = s.CHURN_LOSING_SP and datepart(day, a.churn_date) = s.cday and datepart(hour,a.churn_date) = s.chour and datepart(minute,a.churn_date) = s.cminute and datepart(second,a.churn_date) = s.csecond order by a.churn_losing_sp, a.churn_date'

	sample = hf.fetch_data(server=server,db=db,user=user,pwd=pwd,query=query, col_list=col_list)

	# Normalize count columns


	tn, X, Y = hf.shuffle_split(sample)

	Y = Y.astype(np.int32)

	M = 5 # hidden nodes
	D = X.shape[1] # number of features
	K = len(set(Y))

	tn_train, tn_test, \
	Xtrain, Xtest, \
	Ytrain, Ytest = train_test_split(
    tn, X, Y, test_size = .2, random_state = 12)

	Xtrain = np.array(Xtrain, dtype=np.float32)

	Ytrain_ind = y2indicator(Ytrain, K)
	Ytest_ind = y2indicator(Ytest, K)
	
	# Randomly initialize weights
	W1 = np.random.randn(D, M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M,K)
	b2 = np.zeros(K)

	# Initialize cost array and learning_rate
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

	# Combine results back to tn list for analysis
	col_list.append('pred_noncomp_ind')

	train_results = pd.DataFrame(np.column_stack((tn_train, Xtrain, Ytrain, train_predict_results)), columns=col_list)
	test_results = pd.DataFrame(np.column_stack((tn_test, Xtest, Ytest, test_predict_results)), columns=col_list)

	# Save results
	train_results.to_csv('data/results/train_results' + dt.date.today().strftime('%Y%m%d') + '.csv', index=False, header=True, sep=',')
	test_results.to_csv('data/results/test_results' + dt.date.today().strftime('%Y%m%d') + '.csv', index=False, header=True, sep=',')


if __name__ == '__main__':
    main()
