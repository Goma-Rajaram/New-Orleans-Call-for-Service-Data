# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 13:56:10 2017
@author: Goma

Analysis of Calls for Service data sets. This includes
both calls from the public and from officers on patrol.
"""
import pandas as pd
import numpy as np
from pandas import Series as SR , DataFrame as DF
from numpy import nan as NA
import os
import datetime
from datetime import datetime
from dateutil.parser import parse
#------------------------------------------------------------------------------
#
os.getcwd()
os.chdir('C:\\Sabitha2\\Python projects')

in_df = pd.read_csv('Calls_for_Service_2016.csv')
print in_df.columns     

print len(in_df.index)  #404,064 records
print len(in_df.columns) # 21 columns
print in_df.shape

print in_df.head()

# Check if there are any missing data
in_df.isnull().sum()
# TimeDispatch, TimeArrive, Disposition, DispositionText, Beat, Block_address
# Zip, Location have null values

test_df = in_df.head(n=20)  # For the purpose of testing results
                            # View on original dataset            

test_df1 = in_df.head(n=20).copy()  # Make a copy of it
#------------------------------------------------------------------------------
# What fraction of calls are of the most common type

res1 = in_df.TypeText.describe() # Most common type 'COMPLAINT OTHER'
print 'Fraction of common type of calls', round(res1['freq'].astype(float) / res1['count'].astype(float),2)

#------------------------------------------------------------------------------
# Median response time (dispatch to arrival)
# There are 107,555 NA values for TimeDispatch and 88,634 NA values for 
# TimeArrive .

def print_response_time(value_type, resp_time_in_sec_p):
    print value_type , 'Response Time in seconds', round(resp_time_in_sec_p,2)
    minutes, sec = divmod(resp_time_in_sec_p, 60) 
    print '(',minutes, 'minutes ,', round(sec,2) , 'seconds.',')' 

# Using pandas to_Datetime . 
tmp_res2 = DF({'TimeDispatch' : pd.to_datetime(in_df.TimeDispatch),
               'TimeArrive' : pd.to_datetime(in_df.TimeArrive)})
response_time1 = (tmp_res2.TimeArrive - tmp_res2.TimeDispatch).astype('timedelta64[s]')
final_res1 =  response_time1[response_time1 > 0].median()
print_response_time('Median', final_res1)

# Alternate way - not efficient    
mask = (in_df.TimeDispatch.notnull()) & (in_df.TimeArrive.notnull())
print mask.sum()
tmp_res1 = DF(in_df[mask], columns=['TimeDispatch', 'TimeArrive'])

dispatch = SR([parse(x) for x in tmp_res1.TimeDispatch])
arrive = SR([parse(y) for y in tmp_res1.TimeArrive])
response_time = arrive - dispatch
print len(dispatch) , len(arrive), len(response_time)

resp_time_in_sec = SR([x.total_seconds() for x in response_time])
print len(resp_time_in_sec) , (resp_time_in_sec < 0).sum()
final_res =  resp_time_in_sec[resp_time_in_sec > 0].median()
print_response_time(final_res)

#------------------------------------------------------------------------------
# Average(mean) response time in each district.
# Difference between the avg. response times of the districts with the 
# longest and shortest times

np.set_printoptions(threshold = 175)   
print np.unique(in_df.PoliceDistrict)  
in_df.PoliceDistrict.value_counts()   # Freq 

policeDistricts_list =  np.unique(in_df.PoliceDistrict)                                 
dist_resp_times = SR(np.zeros(len(policeDistricts_list)))                                  
for i in policeDistricts_list:
    mask = (in_df.PoliceDistrict == i) & (response_time1 > 0)  
    dist_resp_times[i] = response_time1[mask].mean()
    print 'District ', i ,
    print_response_time('Mean',dist_resp_times[i])

print 'Longest avg. resp time ' , dist_resp_times.max(), 'for District', dist_resp_times.argmax() 
print 'Shortest avg. resp time ', min(dist_resp_times), 'for District', dist_resp_times.argmin()
print_response_time('Difference in avg.',  max(dist_resp_times) - min(dist_resp_times))       

#------------------------------------------------------------------------------
# Event types that occur more often in a district
     
print np.unique(in_df.Type_)
in_df.Type_.value_counts()  ## 21 has the highest freq
in_df.TypeText[in_df.Type_ == '21']  ## Corresponds to 'COMPLAINT OTHER'
in_df.TypeText.value_counts()

tmp_df1 = in_df.groupby(['PoliceDistrict','Type_','TypeText']).size().reset_index(name='Times')

for i in policeDistricts_list:
    common_event_freq =  tmp_df1.Times[tmp_df1.PoliceDistrict == i].max()
    mask = (tmp_df1.PoliceDistrict == i) & (tmp_df1.Times == common_event_freq)
    print tmp_df1[mask]            
## COMPLAINT OTHER is the common type in all districts.So, find the next common type

for i in policeDistricts_list:
    common_event_freq =  tmp_df1.Times[(tmp_df1.PoliceDistrict == i) & (tmp_df1.Type_ != '21') ].max()
    mask = (tmp_df1.PoliceDistrict == i) & (tmp_df1.Times == common_event_freq)
    print tmp_df1[mask]        
    
# Another way of implementation
for i in policeDistricts_list:
    event_freq =  tmp_df1.Times[(tmp_df1.PoliceDistrict == i)].sort_values(ascending = False)
    second_highest_freq = event_freq[1:2]
    print tmp_df1.iloc[second_highest_freq.index]        
 
#------------------------------------------------------------------------------       
# What percentage of calls are emergency calls for service?

in_df.Priority.notnull().sum()
in_df.Priority.value_counts()

priority_list = in_df.Priority.unique()
print len(priority_list)   #31 Priorities

# Find out unique Priority number and level          
res = []
for myString in priority_list:
    for letter in myString:
        res.append(letter)

res_sr = SR(res)
unique_priority_level = res_sr.unique()  
len(unique_priority_level)
np.set_printoptions(threshold = 20)   
print unique_priority_level

# Testing
test_df1['Priority_num'] = SR([ int((list(x))[0]) for x in test_df1.Priority])
  
# Create a new column with just the priority number
in_df['Priority_num'] = SR([ (list(x))[0] for x in in_df.Priority])

print '% of emergency calls' , ((((in_df.Priority_num == '2').sum() + (in_df.Priority_num == '3').sum()) / float(in_df.shape[0])) * 100)

# top reasons for emergency calls
mask = (in_df.Priority_num == '2') | (in_df.Priority_num == '3' )
mask.sum()
in_df.TypeText[mask].value_counts()

#------------------------------------------------------------------------------
