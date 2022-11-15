import pandas as pd
import numpy as np

###connect to main instead!!!####
test_file = "/mnt/c/School/ml/project/data/final_test_df.csv"
train_file = "/mnt/c/School/ml/project/data/final_train_df.csv"

test = pd.read_csv(test_file) #986 x 1389
train = pd.read_csv(train_file) #300 x 1390

from sklearn.model_selection import train_test_split
###Splitting train datasets######
train_labels = np.array([train['views']]).T
train_data = np.array([train.T[3:]]).T #omit first three columns

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, 
                                    test_size = .33, random_state = 42)

#https://www.kaggle.com/code/mrleritaite/cnn-xgboost/notebook
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import models
from keras import Sequential
from keras import layers
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn import metrics
from keras.layers import Dense, Activation,Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Flatten, Input, Dropout
from keras.utils import np_utils
from keras.models import Sequential, Model
import xgboost

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

inputs = Input(shape=(1387, 1))

conv1d1 = Convolution1D(filters=64, kernel_size=5, padding='same', data_format='channels_first')(inputs)
activation1 = Activation('relu')(conv1d1)
maxpooling1 = MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_first')(activation1)
dropout1 = Dropout(0.25)(maxpooling1)

conv1d2 = Convolution1D(filters=64, kernel_size=4, padding='same', data_format='channels_first')(dropout1)
activation2 = Activation('relu')(conv1d2)
maxpooling2 = MaxPooling1D(pool_size=2, strides=2, padding='same', data_format='channels_first')(activation2)

flatten = Flatten()(maxpooling2)
dense1 = Dense(1024)(flatten)
activation3 = Activation('relu')(dense1)
dense2 = Dense(10)(activation3)
activation4 = Activation('linear')(dense2)
model_dense2_output = Model(inputs=inputs, outputs=activation4)

X_train_xg = model_dense2_output.predict(X_train)
X_test_xg = model_dense2_output.predict(X_test)

z = np.concatenate([np.array(X_train_xg),np.array(y_train)],axis=1)
z = pd.DataFrame(z)
z.to_csv('train_xg.csv',index=False) ###look more into what this is 

z = np.concatenate([np.array(X_test_xg),np.array(y_test)],axis=1)
z = pd.DataFrame(z)
z.to_csv('test_xg.csv',index=False) ###look more into what this is 

#####Another model -- https://www.datatechnotes.com/2019/12/how-to-fit-regression-data-with-cnn.html#:~:text=Convolutional%20Neural%20Network%20%28CNN%29%20models%20are%20mainly%20used,and%20reshape%20the%20input%20data%20according%20to%20it.######
#...add file writing
inputs = Input(shape=(1387, 1))
model = Sequential()
model.add(Conv1D(32, 2, activation="relu", input_shape=(1387, 1)))
model.add(Flatten())
model.add(Dense(64, activation="linear"))
#model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")

X_train_xg = model.predict(X_train)
X_test_xg = model.predict(X_test)

z = np.concatenate([np.array(X_train_xg),np.array(y_train)],axis=1)
z = pd.DataFrame(z)
z.to_csv('train_xg.csv',index=False) ###look more into what this is 

z = np.concatenate([np.array(X_test_xg),np.array(y_test)],axis=1)
z = pd.DataFrame(z)
z.to_csv('test_xg.csv',index=False) ###look more into what this is 


#####XGBoost#####################
train = pd.read_csv('train_xg.csv')
train_y = train['64'].astype('int')
train_x = train.drop(['64'],axis=1)
dtrain = xgboost.DMatrix(train_x, label=train_y)

test = pd.read_csv('test_xg.csv')
test_y = test['64'].astype('int')
test_x = test.drop(['64'],axis=1)
dtest= xgboost.DMatrix(test_x, label=test_y)

# Set parameters.
param = {'max_depth': 7, 
         'eta': 0.2,
         'objective': 'reg:squarederror',
         'nthread': 5,
         'eval_metric': 'rmse'
        }

evallist = [(dtest, 'eval'), (dtrain, 'train')]


# Train the model.
num_round = 70
bst = xgboost.train(param, dtrain, num_round, evallist)

# Make prediction.
ypred = bst.predict(dtest).round()

import math
from sklearn.metrics import mean_squared_error
# Compute RMSE on test set.
mse_xgboost = mean_squared_error(y_test, ypred)
rmse_xgboost = math.sqrt(mse_xgboost)

print('RMSE with XGBoost', rmse_xgboost) #our 1263 vs. medium post 1133

'''
#### Still to do: ########
###1 : save model 
###2 : comparisons
###3 : hyperparamter tuning 

d_public = xgb.DMatrix(p_final_df.loc[:, p_final_df.columns != 'comp_id'][bst.feature_names])
solution = bst.predict(d_public).round()
solution_df = pd.concat([p_final_df[['comp_id']], pd.DataFrame(solution, columns = ['views'])], axis=1)
solution_df.to_csv('solution.csv', index=False)
'''
