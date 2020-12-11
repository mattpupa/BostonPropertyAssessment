#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:13:03 2020

@author: Matt
"""


# Import libraries
import pandas as pd
import numpy as np

#import matplotlib.ticker as ticker
#import matplotlib.pyplot as plt
#import datetime as dt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
The first step is to create dataframes for the earlier years of data.
We'll import all years going back to 2008, but since the
earlier years have few columns, we'll use data from 2017 - 2020 for our
modeling.

We'll start by importing the csv files from 2008 - 2019.
"""

# Creates a variable from CSV files and sets any blank values(' ') to NaN
property2019 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/fy19fullpropassess.csv', na_values=' ')
df_2019 = pd.DataFrame(property2019)

property2018 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/ast2018full.csv', na_values=' ')
df_2018 = pd.DataFrame(property2018)

property2017 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy2017.csv', na_values=' ')
df_2017 = pd.DataFrame(property2017)

property2016 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy2016.csv', na_values=' ')
df_2016 = pd.DataFrame(property2016)

property2015 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy2015.csv', na_values=' ')
df_2015 = pd.DataFrame(property2015)

property2014 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy2014.csv', na_values=' ')
df_2014 = pd.DataFrame(property2014)

property2013 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy13.csv', na_values=' ')
df_2013 = pd.DataFrame(property2013)

property2012 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy12.csv', na_values=' ')
df_2012 = pd.DataFrame(property2012)

property2011 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy11.csv', na_values=' ')
df_2011 = pd.DataFrame(property2011)

property2010 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy10.csv', na_values=' ')
df_2010 = pd.DataFrame(property2010)

property2009 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy09.csv', na_values=' ')
df_2009 = pd.DataFrame(property2009)

property2008 = pd.read_csv('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/property-assessment-fy08.csv', na_values=' ')
df_2008 = pd.DataFrame(property2008)


"""
2020 data is in a txt file that has multiple problems that need to be fixed.

it does have '\n' which makes splitting it into a dataframe with rows
very easy. However, each row has all the columns values together as one
giant string. We need to figure out how to clean this so we have the correct
columns filled out with the appropriate data.
"""
 
# Need to read and clean txt file for 2020
fulldata2020 = open('/Users/Matt/Desktop/DataScience/BostonData/PropertyAssessment/data2020-full.txt', 'r')
next(fulldata2020) #skip first line so column headers aren't first row of data
fulldata2020_r = fulldata2020.read()
fulldata2020_split = fulldata2020_r.split('\n')

df_2020 = pd.DataFrame(fulldata2020_split, columns=['text'])


"""
To clean the text,
the following needs to be done...

1. Split the string based on commas, which will make a list of all the columns
2. Get rid of quotation marks and trailing whitespaces
3. Save the list of column values to prop list
"""

# Create a function that will iterate through rows and clean the text
def clean_rows(row):
    proplist = []
    for r in row.split(','): #1
        proplist.append(r.replace('"', '').rstrip()) #2 and #3
    return proplist
 
# apply clean_rows() on each row
df_2020['cleaned_text'] = df_2020['text'].apply(lambda row: clean_rows(row))


"""
We now have our giant strings for each row cleansed and split into lists. 
However, these lists are being stored in 1 column. We need to write a 
function that will create columns for all the values in these lists.

Before that, we want to look at the list length for each row. This will
determine the number of columns we end up with. Since the dataframes from 
2017 - 2019 have 75 columns, that's what we are hoping to see.
"""

# First, we'll figure out the number of columns (list length) row each row
df_2020['text_length'] = df_2020['cleaned_text'].apply(lambda row: len(row))

# Second, we'll look at the distribution of rows by text_length
df_2020.groupby('text_length').size()


"""
text_length
1          1
75    167668
76      7329
77        48
78         4
79         2
80         1

The data still needs to be cleaned more, since different rows have different 
number of columns. This makes the data misaligned.

Since almost 96% of the rows in df_2020 have text_length 75, we will just
create a new data frame for rows with 75 columns. 167K rows is plenty to 
do our analysis.
"""

# Create new data frame for 2020
df_2020_75 = pd.DataFrame(df_2020.loc[df_2020.text_length == 75])


"""
We now have our new dataframe with the giant strings for each row cleansed 
and split. We will create a final dataframe for 2020 that has a function 
that will create columns for all the values in this list.
"""

# Function to create multiple columns filled with property data
df_2020cleansed = pd.DataFrame(df_2020_75.cleaned_text.tolist()
                               ,index=df_2020_75.index
                               ,columns = df_2019.columns)


"""
Now that all of the data is imported, we want get one dataframe with data 
for 2017-2020, we'll add a column 'Year'to the existing dataframes and 
then combine them.
"""

# Add year to dataframes
df_2020cleansed['Year'] = 2020
df_2019['Year'] = 2019
df_2018['Year'] = 2018
df_2017['Year'] = 2017

# Combine then into one dataframe
df = pd.concat([df_2017, df_2018, df_2019, df_2020cleansed])


"""
There are plenty of records with nan values. This seems to change based on
the property type. 

R_ = residential
S_ = condo main
U_ = condo unit

We'll start with cleaning the residential data first.

We want to get all the records of 'df' where the residential columns
are filled in. We'll do this by isolating the residential columns, and 
then dropping any records where those columns have 'nan' values.

For the records that remain with all the data, we'll create a list out of
their index positions. We'll then create a dataframe from 'df' using the
index positions in our new list.
"""

# Get column index locations for first and last residential columns
df.columns.get_loc('R_BLDG_STYL') #27
df.columns.get_loc('R_VIEW') #48


# Create a dataframe with just the residential columns
residentialcolumns = df[df.columns[27:49]].copy()
residentialcolumns['Year'] = df.Year

# drop all records with 'nan' values
residentialcolumns.replace('', np.nan, inplace=True)
residentialcolumns.dropna(inplace=True)


# create a list out of the index of the remaining records
residential_index = list(residentialcolumns.index)

# Create residential dataframe based on useful residential records
dfresidential = df.loc[residential_index]

# finally, we'll look to see what years our remaining data is from
dfresidential.groupby('Year').size()


"""
Year
2017    206830
2018      4431

It looks like the vast majority of our data is from 2017. This isn't ideal,
since we want our data spread out over multiple years, but having 200K+
records should still be fine to work with.

Need to reset df index so when the residential df is created, you can make
a list of the ORIGINAL DF index, not the new residential index!
"""

#res2020['R_HALF_BTH'][14]












