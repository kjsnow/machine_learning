import pymssql
import pandas as pd

server='174.34.53.73'
db = 'com_cdm_stage'
user = 'ksnow'
pwd = 'Kk5N03!w'

query = 'select case when churn_losing_sp = 1 then 1 else 0 end as is_vzw, case when churn_losing_sp = 2 then 1 else 0 end as is_att, case when churn_losing_sp = 3 then 1 else 0 end as is_spr, case when churn_losing_sp = 4 then 1 else 0 end as is_tmo, case when churn_losing_sp = 6 then 1 else 0 end as is_met, (DATEPART(HOUR, churn_date) * 3600 + DATEPART(MINUTE, churn_date) * 60 + DATEPART(SECOND, churn_date)) as sec_from_midnight, case when datepart(hour, churn_date) >= 21 or datepart(hour, churn_date) < 9 then 1 else 0 end as night_ind, case when datepart(minute, churn_date) % 5 = 0 and datepart(second, churn_date) = 0 then 1 else 0 end as five_min_ind, case when noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind from com_cdm_stage..f_final_churn_reporting where churn_date >= \'2018-05-28\' and churn_date < \'2018-06-01\' and churn_losing_sp in (1,2,3,4,6) and churn_type like \'%DSV\' and noncompetitive_ind in (0,5)'

def sql_query(server=server, user=user, pwd=pwd, db=db, query=query):
	conn = pymssql.connect(server=server, user=user, password=pwd, database=db)
	cursor = conn.cursor(as_dict=True)
	# Run query
	cursor.execute(query)
	# Fetch results
	row = cursor.fetchall() 
	# close connection
	conn.close()
	return row

results = pd.DataFrame(sql_query())

# reorder columns
results = results[['is_vzw', 'is_att', 'is_spr', 'is_tmo', 'is_met', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']]

results = results.as_matrix()

print results
