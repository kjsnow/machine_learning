import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# function to read in sample data
def get_data():
	data = pd.read_csv('data/sample_disconnects_20180528_20180601.csv', header=None)
	data.columns = ['vzw_ind', 'att_ind', 'spr_ind', 'tmo_ind', 'met_ind', 'churn_date', 'sec_from_midnight', 'night_ind', 'five_min_ind', 'noncomp_ind']
	return data

# unused function to generate random data
def rand_times(n):
	rand_seconds = np.random.randint(0, 24*60*60, n)
	return pd.DataFrame(data=dict(seconds=rand_seconds))

# n_rows = 1000

# df = rand_times(n_rows)
# # sort
# df = df.sort_values('seconds').reset_index(drop=True)
# print df.head()

# read data
df = get_data()

# sort data
df = df.sort_values('sec_from_midnight').reset_index(drop=True)

#df2 = df[[6]]
#print df2.head()

# df.sec_from_midnight.plot()
# plt.show()

# total seconds in day
seconds_in_day = 24*60*60

# calculate sin and cos time columns
df['sin_time'] = np.sin(2*np.pi*df.sec_from_midnight/seconds_in_day)
df['cos_time'] = np.cos(2*np.pi*df.sec_from_midnight/seconds_in_day)

# df.sin_time.plot()
# plt.show()

# df.cos_time.plot()
# plt.show()



def assign_color(row):
    if row['noncomp_ind'] == 1:
        return 'red'
    else:
        return 'black'

def add_color(df):
    a=df
    a['color'] = a.apply(lambda row: assign_color(row), axis=1)
    return a

	 #    for x in range(len(a['color'])):
 #        if a['vzw_ind'][x]==1: a['color'][x]='red'
 #        elif a['att_ind'][x]==1: a['color'][x]='blue'
 #        elif a['spr_ind'][x]==1: a['color'][x]='yellow'
 #    	elif a['tmo_ind'][x]==1: a['color'][x]='pink'
 #    	elif a['met_ind'][x]==1: a['color'][x]='purple'

#colors = ['red' if s==1 & v==1 else 'green']

df_vzw = df.loc[(df.vzw_ind == 1)]
df_att = df.loc[(df.att_ind == 1)]
df_spr = df.loc[(df.spr_ind == 1)]
df_tmo = df.loc[(df.tmo_ind == 1)]
df_met = df.loc[(df.met_ind == 1)]

df_vzw = add_color(df_vzw)
df_att = add_color(df_att)
df_spr = add_color(df_spr)
df_tmo = add_color(df_tmo)
df_met = add_color(df_met)

# adjust size of plot markers
marker_size = 10

# base - no color (works)
#df.plot.scatter('sin_time', 'cos_time', marker_size).set_aspect('equal')

# attempt with color column
df_vzw.plot.scatter('sin_time', 'cos_time', marker_size, c=df_vzw['color']).set_aspect('equal')
plt.show()

df_att.plot.scatter('sin_time', 'cos_time', marker_size, c=df_att['color']).set_aspect('equal')
plt.show()

df_spr.plot.scatter('sin_time', 'cos_time', marker_size, c=df_spr['color']).set_aspect('equal')
plt.show()

df_tmo.plot.scatter('sin_time', 'cos_time', marker_size, c=df_tmo['color']).set_aspect('equal')
plt.show()

df_met.plot.scatter('sin_time', 'cos_time', marker_size, c=df_met['color']).set_aspect('equal')
plt.show()




