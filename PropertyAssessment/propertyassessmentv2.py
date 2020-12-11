#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:13:03 2020

@author: Matt

This project is practice for data science and machine learning.
I wanted to get some practice in data cleansing, exploratory data analysis,
and building a predictive model. The data included in this project is Boston
property assessment data from 2008 to 2020.

https://data.boston.gov/dataset/property-assessment

The first part of the project is importing csv files and txt files. The
txt file with 2020 data imported with a weird format, so there's some data 
cleansing that had to be done. After creating a few functions and applying 
them to the2020 data, I was able to get it into a format that matches the 
historical data in the csv files.


The second part is the exploratory data analysis. I noticed a few things
that required further data cleansing.

1. Property codes for 2020 were more descriptive than historical years.
2. CSV files for later years had more columns than csv files for earlier years.
3. 2020 had empty cells vs NaN cells.

In addition, the data seemed to cover 3 types of properties based on the
beginning of the column names. 

R_ = residential
S_ = condo main building
U_ = condo unit

I decided to focus on residential properties for this project, since it
seems like the most simple approach. The residential data includes about 57K
records of data, pretty evenly spread around 13K-14K records per year
for 2017 - 2020. This is plenty of data to build a model.

The main column that has the property assessment data is the AV_TOTAL
column. It combines the average value of the property's land, as well
as it's building. Here are some of the summary stats.

df.AV_TOTAL.astype('int').describe()

Out[9]: 
count    5.739800e+04
mean     7.344950e+05
std      4.859809e+05
min      0.000000e+00
25%      5.081000e+05
50%      6.245000e+05
75%      8.340000e+05
max      1.571170e+07
Name: AV_TOTAL, dtype: float64



The last part of the project is the predictive model. Based on the stats
above, I decided to build a model that will predict if the property
assessment value is at least $750,000. 

I stated earlier that the more recent years of data had more columns (75)
compared to earlier years (50 - 60). Because of that, this model includes
data from 2017 - 2020. I am using 2017 - 2019 data for my training and test 
sets, and use 2020 data as a hold out set that will act as 'new' data.

NOTE: I wonder if this is a bad practice since property values can increase
or decrease year over year. If so, I wonder if there's a way to normalize
this, or a different approach can be taken. Maybe I could build a second
model where I take a portion of 2017 - 2020 data as my test data. However,
that may not be the best since we want to be able to predict 'new' data.
The first approach seems more accurate in that sense.

Since a lot of the data is categorical, I'm using pd.get_dummies
to transform the data into indicator variables.

I started with a decision tree classifier as my first attempt
at building a model. I ran into some more data cleansing issues along the way,
and eventually ended up using a random forest classifier.

I ran into some more data cleansing issues throughout the project, which I'll
note. The end result for the model is interesting, and I have more
questions as I go forward with other projects.
"""


# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


#import matplotlib.ticker as ticker
#import matplotlib.pyplot as plt
#import datetime as dt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

"""
The first step is to create dataframes for the earlier years of data.
I imported all years going back to 2008, but since the
earlier years have few columns, I only used data from 2017 - 2020 for our
modeling.

2017 - 2019 is our training/test data, and 2020 is our holdout 'new' data.

The first step is importing the csv files from 2008 - 2019.
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
giant string. This data needs to be cleansed so we have the correct
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
create a new data frame for rows with 75 columns.
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
Now that all of the data is imported, we'll add a column 'Year'to the 
existing dataframes and then combine them later on.
"""

# Add year to dataframes
df_2020cleansed['Year'] = 2020
df_2019['Year'] = 2019
df_2018['Year'] = 2018
df_2017['Year'] = 2017


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

NOTE: The code below does this for each individual year, and then combines
the data. I tried combining the different yearly data first, then cleaning 
the data, but It didn't work out correctly. I'm not sure why. This approach
still works, but not as efficient codewise. It's something to work on going
forward.
"""

#### 2020 ####
# Get column index locations for first and last residential columns
df_2020cleansed.columns.get_loc('R_BLDG_STYL') #27
df_2020cleansed.columns.get_loc('R_VIEW') #48

# Create a dataframe with just the residential columns
residential20 = df_2020cleansed[df_2020cleansed.columns[27:49]].copy()
residential20['Year'] = df_2020cleansed.Year
residential20['original_index'] = df_2020cleansed.index

# drop all records with 'nan' values
residential20.replace('', np.nan, inplace=True)
residential20.dropna(inplace=True)

# create a list out of the index of the remaining records
residential20_index = list(residential20.index)

# Create residential dataframe based on useful residential records
df_residentrial20 = df_2020cleansed.loc[residential20_index]


#### 2019 ####
# Create a dataframe with just the residential columns
residential19 = df_2019[df_2019.columns[27:49]].copy()
residential19['Year'] = df_2019.Year
residential19['original_index'] = df_2019.index

# drop all records with 'nan' values
residential19.replace('', np.nan, inplace=True)
residential19.dropna(inplace=True)

# create a list out of the index of the remaining records
residential19_index = list(residential19.index)

# Create residential dataframe based on useful residential records
df_residential19 = df_2019.loc[residential19_index]


#### 2018 ####
# Create a dataframe with just the residential columns
residential18 = df_2018[df_2018.columns[27:49]].copy()
residential18['Year'] = df_2018.Year
residential18['original_index'] = df_2018.index

# drop all records with 'nan' values
residential18.replace('', np.nan, inplace=True)
residential18.dropna(inplace=True)

# create a list out of the index of the remaining records
residential18_index = list(residential18.index)

# Create residential dataframe based on useful residential records
df_residential18 = df_2018.loc[residential18_index]

#### 2017 ####
# Create a dataframe with just the residential columns
residential17 = df_2017[df_2017.columns[27:49]].copy()
residential17['Year'] = df_2017.Year
residential17['original_index'] = df_2017.index

# drop all records with 'nan' values
residential17.replace('', np.nan, inplace=True)
residential17.dropna(inplace=True)

# create a list out of the index of the remaining records
residential17_index = list(residential17.index)

# Create residential dataframe based on useful residential records
df_residential17 = df_2017.loc[residential17_index]

# Combine 2017 - 2019 into dataframe for training and test sets
df_model = pd.concat([df_residential17, df_residential18, df_residential19])

# Keep 2020 data seperate as a holdout set
df_holdout = pd.DataFrame(df_residentrial20)

"""
We now have dataframes for our test/training data and our holdout data, 
but let's create smaller dataframes for each year to do inspect the data 
and make sure there aren't NaN values for residential data.
"""

# Create dataframes to do visual checks for NaN values
d19 = df_model.loc[df_model.Year == 2019].copy()
d18 = df_model.loc[df_model.Year == 2018].copy()
d17 = df_model.loc[df_model.Year == 2017].copy()


"""
The NaN values are all gone, but a new issue has come up. 2020 data
has longer codes for the residential columns. IE...

'3B' vs '3B - Three Bedroom'

We need to write a function to trim the code for the appropriate columns, 
so their values match the values from previous years.

2020 also has all the columns in string format. We'll convert the appropriate
columns to ints/floats
"""

# Create a function that will shorten the code value
def shorten_code(row):
    temp = row.split(' ')
    return temp[0]
    
# Need different function for 'R_KITCH' since codes need to be trimmed to one
# character only
def shorten_kitch(row):
    temp = list(str(row))
    return temp[0]


# apply function to appropriate columns in 2020 data
df_holdout['R_BLDG_STYL'] = df_holdout['R_BLDG_STYL'].apply(lambda row: shorten_code(row))
df_holdout['R_ROOF_TYP'] = df_holdout['R_ROOF_TYP'].apply(lambda row: shorten_code(row))
df_holdout['R_EXT_FIN'] = df_holdout['R_EXT_FIN'].apply(lambda row: shorten_code(row))
df_holdout['R_BTH_STYLE'] = df_holdout['R_BTH_STYLE'].apply(lambda row: shorten_code(row))
df_holdout['R_BTH_STYLE2'] = df_holdout['R_BTH_STYLE2'].apply(lambda row: shorten_code(row))
df_holdout['R_BTH_STYLE3'] = df_holdout['R_BTH_STYLE3'].apply(lambda row: shorten_code(row))
df_holdout['R_KITCH'] = df_holdout['R_KITCH'].apply(lambda row: shorten_kitch(row))
df_holdout['R_KITCH_STYLE'] = df_holdout['R_KITCH_STYLE'].apply(lambda row: shorten_code(row))
df_holdout['R_KITCH_STYLE2'] = df_holdout['R_KITCH_STYLE2'].apply(lambda row: shorten_code(row))
df_holdout['R_KITCH_STYLE3'] = df_holdout['R_KITCH_STYLE3'].apply(lambda row: shorten_code(row))
df_holdout['R_HEAT_TYP'] = df_holdout['R_HEAT_TYP'].apply(lambda row: shorten_code(row))
df_holdout['R_AC'] = df_holdout['R_AC'].apply(lambda row: shorten_code(row))
df_holdout['R_EXT_CND'] = df_holdout['R_EXT_CND'].apply(lambda row: shorten_code(row))
df_holdout['R_OVRALL_CND'] = df_holdout['R_OVRALL_CND'].apply(lambda row: shorten_code(row))
df_holdout['R_INT_CND'] = df_holdout['R_INT_CND'].apply(lambda row: shorten_code(row))
df_holdout['R_INT_FIN'] = df_holdout['R_INT_FIN'].apply(lambda row: shorten_code(row))
df_holdout['R_VIEW'] = df_holdout['R_VIEW'].apply(lambda row: shorten_code(row))


"""
After looking at the 2020 data again, there's another issue. The 'YR_BUILT'
and 'YR_REMOD' columns have empty cell values. These need to be converted to
0 to match 2017 to 2019 data. We'll drop the 2 'YR_BUILT' records with a
value of 0.

In addition, the 'R_KITCH' columns had a few values that need to be dropped
from the data set

Finally, we had to convert the appropriate columns in our 2020 dataset to
integers/floats so they can be used correctly in the model
"""

# Convert blank values from YR_BUILT and YR_REMOD to 0
# This could probably be done via a function!
df_holdout['YR_BUILT'] = df_holdout['YR_BUILT'].apply(lambda row: row.replace('', '0') if row == '' else row).astype('float')
df_holdout['YR_REMOD'] = df_holdout['YR_REMOD'].apply(lambda row: row.replace('', '0') if row == '' else row).astype('float')

# Drop 2020 records with 'YR_BUILT' = 0
yr_built_to_drop = df_holdout.loc[(df_holdout['YR_BUILT'] == 0)].index.tolist()
df_holdout.drop(yr_built_to_drop, axis='index', inplace=True)

# R_KITCH has two values with strings. It's easiest to drop these records
r_kitch_to_drop = df_holdout.loc[(df_holdout['R_KITCH'] == 'O') | (df_holdout['R_KITCH'] == 'N')].index.tolist()
df_holdout.drop(r_kitch_to_drop, axis='index', inplace=True)

# Convert datatypes for df holdout to match df_model
df_holdout['PTYPE'] = df_holdout['PTYPE'].astype('int')
df_holdout['LAND_SF'] = df_holdout['LAND_SF'].astype('float64')             
#df_holdout['YR_BUILT'] = df_holdout['YR_BUILT'].astype('int')
#df_holdout['YR_REMOD'] = df_holdout['YR_REMOD'].astype('float64')
df_holdout['GROSS_AREA'] = df_holdout['GROSS_AREA'].astype('float64')
df_holdout['LIVING_AREA'] = df_holdout['LIVING_AREA'].astype('float64')
df_holdout['NUM_FLOORS'] = df_holdout['NUM_FLOORS'].astype('float64')
df_holdout['R_TOTAL_RMS'] = df_holdout['R_TOTAL_RMS'].astype('float64')
df_holdout['R_BDRMS'] = df_holdout['R_BDRMS'].astype('float64')
df_holdout['R_FULL_BTH'] = df_holdout['R_FULL_BTH'].astype('float64')
df_holdout['R_HALF_BTH'] = df_holdout['R_HALF_BTH'].astype('float64')
df_holdout['R_KITCH'] = df_holdout['R_KITCH'].astype('float64')
df_holdout['R_FPLACE'] = df_holdout['R_FPLACE'].astype('float64')

"""
Now that our data is cleansed, the model is ready to be built. 

As a reminder, we are going to predict whether residential properties
are assessed at $750,000 or more.

There are a few steps to this...

1. Take out categorical features that aren't relevant to prediction. This
includes things like owner names, etc. Look at the data key for this!
2. Use pd.get_dummies to convert the remaining features to indicators.

NOTE: I later found out that there are columns with different values
in 2020 data vs earlier data. That means when the features are converted
to indicator columns, the number of columns between X_model and X_holdout
won't match. WHAT IS THE BEST WAY TO HANDLE THIS?!? ARE THE COLUMNS SUPPOSED
TO BE IN THE SAME ORDER?

For now, I looked at what columns were missing for each dataset, and
added them in with all 0's. The number of columns in X_model and X_holdout
now match.

NOTE: In line 511 I'm dropping 'ZIPCODE'. I ran the model once, and it 
had 4 out of 5 most important features of prediction. Since location is such
and obvious indicator of property value, I decided to remove it from this
project. There is also a different mix of zip codes between 2017 - 2019 data,
and 2020 data.
"""

# Create predictor variable column that will be predicted by the features
df_model['R_750K'] = df_model['AV_TOTAL'].apply(lambda row: 1 if row >= 750000 else 0)

# Remove columns that aren't relevant to model
df_model.drop(['PID' ,'CM_ID' ,'GIS_ID' ,'ST_NUM' ,'ST_NAME' ,'ST_NAME_SUF'
               ,'UNIT_NUM' ,'OWNER', 'MAIL_ADDRESSEE', 'MAIL_ADDRESS'
               ,'MAIL CS' ,'MAIL_ZIPCODE', 'AV_LAND', 'AV_BLDG', 'AV_TOTAL'
               ,'GROSS_TAX', 'Year', 'STRUCTURE_CLASS'], axis=1 ,inplace=True)
df_model.drop(df_model.columns[32:-1], axis=1 ,inplace=True)

# before I drop columns in the 2020 holdout data, I'm creating a df that 
# will be used for the final evaluation on 'new' data
df_holdout_for_eval = df_holdout.copy()

df_holdout.drop(['PID' ,'CM_ID' ,'GIS_ID' ,'ST_NUM' ,'ST_NAME' ,'ST_NAME_SUF'
               ,'UNIT_NUM' ,'OWNER', 'MAIL_ADDRESSEE', 'MAIL_ADDRESS'
               ,'MAIL CS' ,'MAIL_ZIPCODE', 'AV_LAND', 'AV_BLDG', 'AV_TOTAL'
               ,'GROSS_TAX', 'Year', 'STRUCTURE_CLASS'], axis=1 ,inplace=True)
df_holdout.drop(df_holdout.columns[32:], axis=1 ,inplace=True)

# https://datascience.stackexchange.com/questions/11928/valueerror-input-contains-nan-infinity-or-a-value-too-large-for-dtypefloat32
# Drop remaining NaN values in YR_REMOD column
df_model.dropna(inplace=True)

# Create y_model now since we will drop 'R_750K' in df_model
y_model = df_model['R_750K']

# Remove 'R_750K' column since that is our y, and convert df_model to 
# indicator feature set and create X_model
df_model.drop(['R_750K', 'ZIPCODE'],axis=1,inplace=True)
X_model = pd.get_dummies(df_model)

# Add appropriate columns for X_model that will have all 0's
X_model['R_AC_Y'] = 0

# Create testing and training data sets
X_train, X_test, y_train, y_test = train_test_split(X_model, y_model
                                                    , random_state=0)

# create decision tree classifier as first model attempt
dtc = DecisionTreeClassifier(random_state = 0
                             ,max_depth=20 #Usually this is all you need in practice
                             #,max_leaf_nodes = 2500
                             ).fit(X_train, y_train)


"""
The model is built, so now we want to analyze its performance. We can
use several metrics including accuracy, AUC, recall, etc.

The accuracy score of the training set is .97 without any pruning, so 
this may be overfitting. Try recreating the model with pre-pruning 
arguments like...
1. max_depth # depth is 41 without pruning
2. max_leaf_nodes # n_leaves is 3831 without pruning
3. min_samples_leaf

We can analyze the decision tree using methods at the link below...
https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

We can use a confusion matrix to analyze the TYPES of errors. This includes
false positives (type 1 error) and false negative (type 2 error)
TN FP
FN TP

We can use decision functions and probability to make our predictions 
more conservative with predict_proba. The higher the threshold, 
the more conservative the prediction is!

We can't easily look at AUC because the decision_function doesn't
exist for decision tree classifiers. In the commented link, there
is an approach that binarizes 'y'. You can check this out later
if you want.

Once we have different single evaluation scores, it might make sense
to do some type of cross validation. This makes sense for decision trees
since they tend to naturally overfit. Running different variations of
the tree may product a more reliable score. Look at 'scoring' parameter
for different measurements!

Can you use GridSearchSV with decision tree classifiers?

The last evaluation metric I looked at was random forest. I wanted to
make the decision tree classifier more reliable by using an ensemble approach.

NOTE: Overall, the accuracy scores came out well. Accuracy is an acceptable 
metric to use for this project since I don't have a clear purpose of doing 
anything with the data.

Initially, my decision tree classifier had an accuracy score of .97 on the
training data. I was worried there might be overfitting, so I added a
max_depth param of 20. This brought the accuracy score down to .95 on the
training data, and .89 on the test data. There probably isn't overfitting
anymore, but I do wonder what's the ideal accuracy score to have. To be safe,
I also ran a random forest classifier.

My random forest accuracy scores came out even better, which I'm pleased with.
The training data had .97 and the test data had .92. Both of my classifiers
performed much better than a dummy classifier, which got an accuracy score of
.73.
"""

# accuracy
accuracy_score_train = dtc.score(X_train, y_train) 
accuracy_score_test = dtc.score(X_test, y_test)

# compare accuracy to dummy classifier
dummy = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)

y_dummy_predictions = dummy.predict(X_test)

accuracy_score_dummy = dummy.score(X_test, y_test)

# confusion matrix 
confusion = confusion_matrix(y_test, dtc.predict(X_test))

# precision score...TP / (TP + FP)
# What fraction of positive predictions are correct
precision_score(y_test,dtc.predict(X_test))

# recall score...TP / (TP+FN)
# what fraction of all positives are correctly predicted positive
recall_score(y_test,dtc.predict(X_test))

# f1 score...(2 * TP) / (2 * TP + FN + FP)
# combines precision and recall into one number
f1_score(y_test,dtc.predict(X_test))

# prediction probabilities
y_proba_dtc = dtc.fit(X_train, y_train).predict_proba(X_test)
y_proba_list = list(zip(y_test[0:20], y_proba_dtc[0:20,1]))

# show the probability of positive class for first 20 instances
y_proba_list

# ROC curves, AUC
# https://stackoverflow.com/questions/45376410/how-to-get-roc-curve-for-decision-tree

# Set feature importances
fi = dtc.feature_importances_

# Sort in descending order
top_5_values = np.sort(fi)[::-1].tolist()[0:5]
   
# Sort indexes in descending order
top_5 = np.argsort(fi)[::-1].tolist()[0:5]

# Create list for top 5 most important features
top_5_list = []   
for t in top_5:
    top_5_list.append(X_train.columns[t])

 ### further model evaluation ###

# cross validation score
cross_validation_score = cross_val_score(dtc, X_train, y_train, cv=5)

# grid search

# Random Forest
rfc = RandomForestClassifier(random_state = 0)
rfc.fit(X_train, y_train)

rfc_score_train = rfc.score(X_train, y_train)
rfc_score_test = rfc.score(X_test, y_test)


"""
Now that the model has been built using df_model data, the model will be
used to predict our 750K variable for the initial holdout group.

Just like we did with X_model, we'll have to add in missing columns with
all 0's.

From there, we'll run the classifier predictions on the 2020 holdout data.


NOTE: I'm a little surprised by these results. When I manually look at the
accuracy of these predictions, the performance is MUCH lower than my training
and test data. Only .6 of the properties are predicted correctly. I have a few
thoughts/questions as to why...

1. Did the order of rows get mixed up before I created 'df_final', which
would screw up the data?
2. Did my decision tree and random forest actually overfit way more than I
expected?
3. Is 2020 dat fundamentally different than 2017 - 2019 data? This could be
possible based on the difference in zipcodes I mentioned above. Also, it was
the only data that was in a txt file that I had to clean. Maybe that skewed
the data more than I thought.
4. Are adding these extra columns to X_model and X_holdout right before
running the models skewing things? Do the fact that those columns need to
be added also indicate that the data is more different than I initially
thought?

If I think of ways to check this, I'll investigate more. I could remove 2020
data all together, and take a holdout group from the 2017 - 2019 data before
then splitting it for train/test groups. That would be my first experiment.

Since I want to work on different projects, for now my take away will
be to make sure the data I'm holding out is very representative of the 
data I'm building the model with.
"""

# Convert df_holdout to indicator feature set and create X_holdout
df_holdout.drop(['ZIPCODE'],axis=1,inplace=True)
X_holdout = pd.get_dummies(df_holdout)

# Add appropriate columns for X_model that will have all 0's
X_holdout['LU_A'] = 0
X_holdout['LU_CM'] = 0
X_holdout['LU_E'] = 0
X_holdout['LU_EA'] = 0
X_holdout['LU_RC'] = 0
X_holdout['R_BLDG_STYL_VT'] = 0
X_holdout['R_ROOF_TYP_O'] = 0
X_holdout['R_EXT_FIN_G'] = 0

X_holdout.drop(['R_BLDG_STYL_105'], axis=1, inplace=True)

# Predict decision tree classifier score for holdout group
dtc_predicted = dtc.predict(X_holdout)

# Predict random forest classifier score for holdout group
rfc_predicted = rfc.predict(X_holdout)

# Analyze performance on 'future' data (in this case, 2020 data)
df_holdout_for_eval['R_750K'] = df_holdout_for_eval['AV_TOTAL'].astype('float').apply(lambda row: 1 if row >= 750000 else 0)
df_holdout_for_eval['prediction'] = rfc_predicted

# FINAL ANALYSIS ON 'NEW' 2020 DATA!!!
df_final = df_holdout_for_eval[['AV_TOTAL', 'R_750K', 'prediction']]

df_final_grouped = df_final.groupby(['R_750K','prediction']).size()



