import pymssql

server='174.34.53.73'
db = 'com_cdm_stage'
user = 'ksnow'
pwd = 'Kk5N03!w'

conn = pymssql.connect(server=server, user=user, password=pwd, database=db)
cursor = conn.cursor()

# Run query
cursor.execute("select top 10 * from f_final_churn_reporting")

# Fetch results
row = cursor.fetchall() 
#while results:
print row

conn.close()
