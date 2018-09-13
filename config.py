
import configparser
import datetime as dt

config = configparser.ConfigParser()

# Path to your sql credentials ini file
#credentials_path = 'C:\Users\ksnow\Desktop\credentials'
credentials_path = 'D:\Kyle\Work\credentials'

# Input path to location to save all machine_learning related files
storage_path = '\\\\storage.comlinkdata.com\\Data Ops\\Users\\Snow\\machine_learning'

# Server for experiment
env = 'PROD'

# Enter name of experiment to run
experiment = 'baseline'

# Set True to run query, False to read file from raw_file_path location
to_query = False

# Number of hidden nodes
num_nodes = 5

# Number of hidden layers
num_layers = 2

# Number of iterations
num_epochs = 1000

# Learning Rate
learning_rate = .000001

# To save results of experiment set to True
save_results = True

# To make the testing set the last day of data set to True
last_day_testing = False

# Default to pull last week of data
start_date =  dt.datetime.now().date() - dt.timedelta(days=10)
end_date = dt.datetime.now().date() - dt.timedelta(days=2)
# Or choose custom date range
#start_date = dt.date(2018,5,10)
#end_date = dt.date(2018,5,13)

# raw file path to save query results in form of [experiment]_[date]
#raw_file_path = storage_path + '/data/raw/' + experiment + '_' + dt.date.today().strftime('%Y%m%d') + '.csv'
raw_file_path = storage_path + '/data/raw/' + experiment + '_20180720' + '.csv'


# Enter column list of query below
# FIGURE OUT HOW TO MAKE THIS READ AFTER THE QUERY
col_list = ['tn', 'churn_date', 'is_vzw', 'is_att', 'is_spr', 'is_tmo', 'is_met', 'night_ind', 'five_min_ind', 'count_same_minute', 'count_same_second', 'noncomp_ind']

query = """
			with cte as (
select tn, churn_losing_sp, churn_date, noncompetitive_ind 
from com_cdm_stage..f_final_churn_reporting where churn_date >= '{0}' and churn_date < '{1}' and churn_losing_sp in (1,2,3,4,6) and churn_type like '%DSV' and noncompetitive_ind in (0,5)
	),
	cte_minute as (
	select churn_losing_sp, datepart(day, churn_date) as cday, datepart(hour,churn_date) as chour, datepart(minute,churn_date) as cminute, count(1) as count_same_minute 
	from cte
	group by churn_losing_sp, datepart(day, churn_date), datepart(hour,churn_date), datepart(minute,churn_date)
	),
	cte_second as (
	select churn_losing_sp, datepart(day, churn_date) as cday, datepart(hour,churn_date) as chour, datepart(minute,churn_date) as cminute, datepart(second,churn_date) as csecond, count(1) as count_same_second 
	from cte
	group by churn_losing_sp, datepart(day, churn_date), datepart(hour,churn_date), datepart(minute,churn_date), datepart(second,churn_date) 
	)
	select a.tn, 
	a.churn_date,
	case when a.churn_losing_sp = 1 then 1 else 0 end as is_vzw, 
	case when a.churn_losing_sp = 2 then 1 else 0 end as is_att, 
	case when a.churn_losing_sp = 3 then 1 else 0 end as is_spr, 
	case when a.churn_losing_sp = 4 then 1 else 0 end as is_tmo, 
	case when a.churn_losing_sp = 6 then 1 else 0 end as is_met,
	case when datepart(hour, a.churn_date) >= 21 or datepart(hour, a.churn_date) < 9 then 1 else 0 end as night_ind,
	case when datepart(minute, a.churn_date) % 5 = 0 and datepart(second, a.churn_date) = 0 then 1 else 0 end as five_min_ind,
	m.count_same_minute,
	s.count_same_second,
	case when a.noncompetitive_ind = 5 then 1 else 0 end as noncomp_ind 
	from cte a
	inner join cte_minute m
	on a.CHURN_LOSING_SP = m.CHURN_LOSING_SP
	and datepart(day, a.churn_date) = m.cday
	and datepart(hour,a.churn_date) = m.chour
	and datepart(minute,a.churn_date) = m.cminute
	inner join cte_second s
	on a.CHURN_LOSING_SP = s.CHURN_LOSING_SP
	and datepart(day, a.churn_date) = s.cday
	and datepart(hour,a.churn_date) = s.chour
	and datepart(minute,a.churn_date) = s.cminute
	and datepart(second,a.churn_date) = s.csecond
	order by a.churn_losing_sp, a.churn_date
	""".format(start_date, end_date)

# DO NOT CHANGE BELOW
# Credentials read based on env
config.read(credentials_path + '\sql_credentials.ini')

class SQL_Credentials:
	def __init__(self, env):
		self.server = config.get(env,'server')
		self.db = config.get(env,'db')
		self.user = config.get(env,'user')
		self.pwd = config.get(env,'pwd')

creds = SQL_Credentials(env)