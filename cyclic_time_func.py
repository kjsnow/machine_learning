import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_data():
	data = pd.read_csv('E:\Work\machine_learning\sample_disconnects.csv', header=None)
	data.columns = ['vzw_ind', 'att_ind', 'spr_ind', 'tmo_ind', 'met_ind', 'churn_date', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']
	return data

def rand_times(n):
	rand_seconds = np.random.randint(0, 24*60*60, n)
	return pd.DataFrame(data=dict(seconds=rand_seconds))

# n_rows = 1000

# df = rand_times(n_rows)
# # sort
# df = df.sort_values('seconds').reset_index(drop=True)
# print df.head()

df = get_data()
df = df.sort_values('sec_from_midnight').reset_index(drop=True)
print df.head()

#df2 = df[[6]]
#print df2.head()

# df.sec_from_midnight.plot()
# plt.show()

seconds_in_day = 24*60*60

df['sin_time'] = np.sin(2*np.pi*df.sec_from_midnight/seconds_in_day)
df['cos_time'] = np.cos(2*np.pi*df.sec_from_midnight/seconds_in_day)

# df.sin_time.plot()
# plt.show()

# df.cos_time.plot()
# plt.show()

marker_size = 1

def c(noncomp_ind, vzw_ind, att_ind, spr_ind, tmo_ind, met_ind):
    if noncomp_ind == 1 and vzw_ind == 1: return "red"


#colors = [c(noncomp_ind, vzw_ind, att_ind, spr_ind, tmo_ind, met_ind) 
#for (noncomp_ind, vzw_ind, att_ind, spr_ind, tmo_ind, met_ind) in df(noncomp_ind, vzw_ind, att_ind, spr_ind, tmo_ind, met_ind) for (noncomp_ind, vzw_ind, att_ind, spr_ind, tmo_ind, met_ind)]

# colors = np.where((df['noncomp_ind']==1) & (df['vzw_ind'] == 1), 'red'
# 	,(df['noncomp_ind']==1) & (df['att_ind'] == 1), 'blue'
# 	,(df['noncomp_ind']==1) & (df['spr_ind'] == 1), 'yellow'
# 	,(df['noncomp_ind']==1) & (df['tmo_ind'] == 1), 'pink'
# 	,(df['noncomp_ind']==1) & (df['met_ind'] == 1), 'purple'
# 	, 'green')

def addcolor(b):

    a=b
    a['color']='black'
    print len(a)
    a.loc[(a['vzw_ind']==1 & a['noncomp_ind'] == 1), 'color'] = 'red'
    a.loc[a['att_ind']==1, 'color'] = 'blue'
    a.loc[a['spr_ind']==1, 'color'] = 'yellow'
    a.loc[a['tmo_ind']==1, 'color'] = 'pink'
    a.loc[a['met_ind']==1, 'color'] = 'purple'
    return a

	 #    for x in range(len(a['color'])):
 #        if a['vzw_ind'][x]==1: a['color'][x]='red'
 #        elif a['att_ind'][x]==1: a['color'][x]='blue'
 #        elif a['spr_ind'][x]==1: a['color'][x]='yellow'
 #    	elif a['tmo_ind'][x]==1: a['color'][x]='pink'
 #    	elif a['met_ind'][x]==1: a['color'][x]='purple'

#colors = ['red' if s==1 & v==1 else 'green']

df = addcolor(df)
print df.head()
#df.plot.scatter('sin_time', 'cos_time', marker_size, c=df['color']).set_aspect('equal')
#plt.show()