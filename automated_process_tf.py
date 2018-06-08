import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pymssql

from sklearn.utils import shuffle

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

	query = 'select case when churn_losing_sp = 1 then 1 else 0 end as is_vzw, case when churn_losing_sp = 2 then 1 else 0 end as is_att, case when churn_losing_sp = 3 then 1 else 0 end as is_spr, case when churn_losing_sp = 4 then 1 else 0 end as is_tmo, case when churn_losing_sp = 6 then 1 else 0 end as is_met, (DATEPART(HOUR, churn_date) * 3600 + DATEPART(MINUTE, churn_date) * 60 + DATEPART(SECOND, churn_date)) as sec_from_midnight, case when datepart(hour, churn_date) >= 21 or datepart(hour, churn_date) < 9 then 1 else 0 end as night_ind, case when datepart(minute, churn_date) % 5 = 0 and datepart(second, churn_date) = 0 then 1 else 0 end as five_min_ind, case when noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind from com_cdm_stage..f_final_churn_reporting where churn_date >= \'2018-05-28\' and churn_date < \'2018-06-01\' and churn_losing_sp in (1,2,3,4,6) and churn_type like \'%DSV\' and noncompetitive_ind in (0,5)'

	results = pd.DataFrame(sql_query(query=query))

	col_names = ['is_vzw', 'is_att', 'is_spr', 'is_tmo', 'is_met', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']

	# reorder columns
	results = results[col_names]

	results = results.as_matrix()
	results = shuffle(results)

	X = results[:, :-1]
	Y = results[:, -1]

	return X, Y

X, Y = fetch_data()

Y = Y.astype(np.int32)

Xtrain, Xtest, \
Ytrain, Ytest = train_test_split(
    X, Y, test_size = .2, random_state = 12)

M = 20 # hidden nodes
D = Xtrain.shape[1] # number of features
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
			print ('Minibatch loss at iteration {0}: {1}'.format(i, l))

	print ('Test accuracy: {0}%'.format(accuracy(test_prediction.eval(), Ytest)))

	legend1, = plt.plot(loss, label='loss')
	#legend2, = plt.plot(test_costs, label='test_costs')
	plt.legend([legend1])
	plt.show() 