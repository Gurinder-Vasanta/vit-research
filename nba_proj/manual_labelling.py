import pandas as pd
import numpy as np
df = pd.read_csv('data/manual_intervals.csv')
# print(df)

output_dict = {}

ls = np.array(df['left_start'])
le = np.array(df['left_end'])

left = []
for i in range(len(ls)):
    try: 
        splitted = ls[i].split('_')
        left.append([ls[i],le[i]])
    except: 
        continue
output_dict['left'] = left

rs = np.array(df['right_start'])
re = np.array(df['right_end'])

right = []
for i in range(len(rs)):
    try: 
        splitted = rs[i].split('_')
        right.append([rs[i],re[i]])
    except: 
        continue
output_dict['right'] = right

ns = np.array(df['none_start'])
ne = np.array(df['none_end'])

none = []
for i in range(len(ns)):
    try: 
        splitted = ns[i].split('_')
        none.append([ns[i],ne[i]])
    except: 
        continue
output_dict['none'] = none
print(output_dict)