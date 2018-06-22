import pandas as pd
import pymssql

from collections import OrderedDict
from sklearn.utils import shuffle


def sql_query(server, db, user, pwd, query):
	conn = pymssql.connect(server=server, user=user, password=pwd, database=db)
	cursor = conn.cursor(as_dict=True)
	# Run query
	cursor.execute(query)
	# Fetch results
	row = cursor.fetchall() 
	# close connection
	conn.close()
	return row


def fetch_data(server, db, user, pwd, query, col_list):

	results = pd.DataFrame(sql_query(server, db, user, pwd, query))

	#col_names = ['is_vzw', 'is_att', 'is_spr', 'is_tmo', 'is_met', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']

	# reorder columns
	#results = results[col_names]

	# correct order
	results = results[col_list]

	results = results.as_matrix()


	return results


def shuffle_split(df):

	# shuffle
	df = shuffle(df)

	# split into id col (tn), inputs (X) and outputs (Y)
	ID = df[:,0]
	X = df[:, 1:-1]
	Y = df[:, -1]

	return ID, X, Y