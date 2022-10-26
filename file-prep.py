import os
import pandas as pd
import numpy as np
import math
import random
import collections
import timeit
import xgboost as xgb
import sklearn.metrics

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

path = "/mnt/c/School/ml/project/data/"
desc_train = pd.read_csv(f'{path}train_desc_df.csv')
meta_train = pd.read_csv(f'{path}train_meta_df.csv')
image_train = pd.read_csv(f'{path}train_image_df.csv')
title_train = pd.read_csv(f'{path}train_title_df.csv')

# Load public datasets (datasets used for the rankings)
desc_test = pd.read_csv(f'{path}public_desc_df.csv')
meta_test = pd.read_csv(f'{path}public_meta_df.csv')
image_test = pd.read_csv(f'{path}public_image_df.csv')
title_test = pd.read_csv(f'{path}public_title_df.csv')

####Get dummies for categorical data######
embed = pd.get_dummies(meta_train.embed, prefix ='embed')
partner = pd.get_dummies(meta_train.partner, prefix ='partner')
partner_active = pd.get_dummies(meta_train.partner_active, prefix ='partner_a')
language = pd.get_dummies(meta_train['language'], prefix='language')
weekday = pd.get_dummies(meta_train['dayofweek'], prefix='day')
weekday['day_6'] = 0

####Cyclical features encoding#######
sin_hour = np.sin(2*np.pi*meta_train['hour']/24.0)
sin_hour.name = 'sin_hour'
cos_hour = np.cos(2*np.pi*meta_train['hour']/24.0)
cos_hour.name = 'cos_hour'

# Join all dataframes for final training df.
meta_final_df = pd.concat([meta_train[['comp_id', 'views', 'ratio', 'language', 'n_likes', 'duration']].reset_index(drop=True),
                           embed, partner, partner_active, language, weekday, sin_hour, cos_hour], axis=1)
meta_final_df.head()
meta_final_df.shape

###Using lasso regression to speed up image 
# obtain several predictors that minimize the prediction 
#error for a quantitative target variable by imposing 
#constraint on the model parameters that causes some 
#variables to shrink to zero (allowing for shrinkage of 
#image data)
# Set the target as well as dependent variables from image data.
y = meta_train['views']
x = image_train.loc[:, image_train.columns != 'comp_id'] #ignore comp_id variable

# Run Lasso regression for feature selection.
sel_model = SelectFromModel(LogisticRegression(C=1, penalty='l1', solver='liblinear'))

# time the model fitting
start = timeit.default_timer()

# Fit the trained model on our data
sel_model.fit(x, y)

stop = timeit.default_timer()
print('Time: ', stop - start) 

# get index of good features
sel_index = sel_model.get_support()

# count the no of columns selected
counter = collections.Counter(sel_model.get_support())
counter

# Reconstruct the image dataframe using the index information above.
image_index_df = pd.DataFrame(x[x.columns[(sel_index)]])
image_final_df = pd.concat([image_train[['comp_id']], image_index_df], axis=1)
image_final_df.head()

# Merge all tables based on the column 'comp_id'
final_df = pd.merge(pd.merge(meta_final_df, image_final_df, on = 'comp_id'), 
                    pd.merge(desc_train, title_train, on = 'comp_id'), on = 'comp_id')

final_df.shape # (3000, 1389)

final_df.to_csv("final_train_df.csv")


###Now on public/test data
# Test set
p_embed = pd.get_dummies(meta_test.embed, prefix ='embed')
p_partner = pd.get_dummies(meta_test.partner, prefix ='partner')
p_partner_active = pd.get_dummies(meta_test.partner_active, prefix ='partner_a')
p_language = pd.get_dummies(meta_test['language'], prefix='language')
p_language['language_6'] = 0
p_weekday = pd.get_dummies(meta_test['dayofweek'], prefix='day')
p_weekday['day_3'] = 0
p_weekday['day_4'] = 0
p_weekday['day_5'] = 0

## Cyclical encoding 
p_sin_hour = np.sin(2*np.pi*meta_test['hour']/24.0)
p_sin_hour.name = 'sin_hour'
p_cos_hour = np.cos(2*np.pi*meta_test['hour']/24.0)
p_cos_hour.name = 'cos_hour'

# Join all dataframes.
p_meta_final_df = pd.concat([meta_test[['comp_id', 'ratio', 'language', 'n_likes', 'duration']].reset_index(drop=True),
                             p_embed, p_partner, p_partner_active, p_language, p_weekday, p_sin_hour, p_cos_hour], axis=1)
p_meta_final_df.head()

# subset our test image dataframe with index used on training set
p_image_final_df = pd.concat([image_test[['comp_id']], image_index_df], axis=1)

# Merge all test set tables.
p_final_df = pd.merge(pd.merge(p_meta_final_df, p_image_final_df, on = 'comp_id'), 
                    pd.merge(desc_test, title_test, on = 'comp_id'), on = 'comp_id')
p_final_df.shape

p_final_df.to_csv("final_test_df.csv")

