# -*- coding: utf-8 -*-
"""
Solution development for Section 1 of The Data Institute Fellowship Challenge
Arrest incidents in Los Angeles.
Carson Bulgin
George Mason University
cbulgin@gmu.edu
Created on Fri Nov  1 12:35:25 2019
"""

"""
Part 1: import necessary modules
"""
import os,sys
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from math import radians, degrees, sin, cos, asin, acos, sqrt


myDir = '/users/chanb/Documents/TDIChallenge2019'
sys.path.append( myDir )

"""
Part 2: Import the data, remove observations that are not needed for any of the challenge steps.
"""
crimedata = pd.read_csv('Arrest_Data_from_2010_to_Present.csv')
print (crimedata.head(5),"\n")
# print(crimedata.dtypes)
# convert Arrest Date to date
# note to self: this works but is very computationally intensive.  Revisit this based on a
# wildcard match of the Arrest Date format (e.g., "*2018") using just a string comparison
# if time permits.
crimedata['Arrest Date'] = crimedata['Arrest Date'].astype('datetime64[ns]')
print(crimedata.dtypes)
#remove data from 1/1/2019 to the present.
crimedata = crimedata[crimedata['Arrest Date'] < '01/01/2019']


"""
Part 3: Question 1.  Count arrests in 2018
"""
crimedata2018 = crimedata[crimedata['Arrest Date'] >= '01/01/2018']
print(len(crimedata2018.index))
#note the answer is 104,277.

"""
Part 4: Question 2.  Calculate the 95% quantile of the age of arrestees 
in 2018 for Vehicle Theft, Robbery, Burglary, Receive Stolen Property
"""
# reduce the data set to just those crimes.
q2_types = ['Vehicle Theft', 'Robbery', 'Burglary', 'Receive Stolen Property']
crimedata2018_q2 = crimedata2018[crimedata2018['Charge Group Description'].isin(q2_types)]

#calculate the 95% quantile for age of this new dataframe.
print(crimedata2018_q2.quantile(.95))
#note the answer is 52.

"""
Part 5: Question 3.  How many bookings of arrestees were made in the area with the most arrests in 2018?
"""
crimedata2018_q3 = crimedata2018.groupby(["Area ID"])["Area ID"].count()
print(crimedata2018_q3.max())
#note the answer is 10951

"""
Part 6: Question 4.  Calculate the Z-score of the average age for each charge group.
 - exclude years other than 2018
 - exclude "Pre-Delinquency" and "Non-Criminal Detention"
 - exclude unknown charge description
 """
q4_types = ['Pre-Delinquency', 'Non-Criminal Detention']
crimedata2018_q4 = crimedata2018[~crimedata2018['Charge Group Description'].isin(q4_types)]
#remove nan values for charge description
crimedata2018_q4 = crimedata2018_q4.dropna(subset=['Charge Group Description'])
crimedata2018_q4 = crimedata2018_q4[['Charge Group Description','Age']]

# first we need the mean and std deviation - we need the z-score based on the entire dataset, not just the groups.
ca_mean = crimedata2018_q4['Age'].mean()
ca_std = crimedata2018_q4['Age'].std()

# put the group means we need to evaluate in a dataframe.
q4_mean_df = pd.DataFrame(crimedata2018_q4.groupby('Charge Group Description').mean())
# calculate the z-score and add it to the dataframe.
abs_z = lambda cd_2018: abs((cd_2018 - ca_mean)/ca_std)
q4_mean_df.insert(1, 'Z-score', q4_mean_df.transform(abs_z))
q4_max_charge = q4_mean_df['Z-score'].idxmax()
q4_max_z = q4_mean_df['Z-score'].max()
print('The charge code with the largest absolute value z-score for age is:', q4_max_charge, q4_max_z)

"""
Part 7: Question 5.  Using a trend line, what is the projected number of felony arrests in 2019?
- include only felonies
- include data from 2010-2018
"""
#remove non-felonies
crimedata_q5 = crimedata[crimedata['Arrest Type Code'].isin(['F'])]

#convert date to year
crimedata_q5 = pd.DataFrame(crimedata['Arrest Date'].dt.year)

#group by year
q5_counts = crimedata_q5.groupby(['Arrest Date']).size().reset_index(name='counts')

#calculate the slope of the trend line.
m = ( ( (q5_counts['Arrest Date'].mean()* q5_counts['counts'].mean() ) - mean(q5_counts['Arrest Date'] * q5_counts['counts'])) /
       ((q5_counts['Arrest Date'].mean()*q5_counts['Arrest Date'].mean())- mean(q5_counts['Arrest Date']*q5_counts['Arrest Date'])))

#calculate the y-intercept of the trend line
b = q5_counts['counts'].mean() - m * q5_counts['Arrest Date'].mean()

#calculate the trend line
regression_line = [(m*x)+b for x in q5_counts['Arrest Date']]

#plot the results
plt.scatter(q5_counts['Arrest Date'], q5_counts['counts'])
plt.plot(q5_counts['Arrest Date'], regression_line)

#solve for y = 2019
predict_2019 = m * 2019 + b
#answer is 95229.77777682245 or 95230 as a rounded integer.

"""
Part 8: Question 6.  How many arrest incidents were made per km on Pico Boulevard in 2018?
- remove lat / lon is +/- 2 stdev of the mean of points
- use spherical Earth projected to a plane equation
"""

#remove all non-Pico locations

# implement spherical earth function


