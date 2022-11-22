import numpy as np 
import pandas as pd

test_file = "/mnt/c/School/ml/project/data/final_test_df.csv"
train_file = "/mnt/c/School/ml/project/data/final_train_df.csv"

test = pd.read_csv(test_file) #986 x 1389
train = pd.read_csv(train_file) #300 x 1390

def separate(df): 
	image_test = df.filter(like='image')
	title_test = df.filter(like='title')
	desc_test = df.filter(like='desc')
	labels_to_drop = (image_test).columns.append((title_test).columns.append((desc_test).columns))
	meta_test = df.drop(columns=list(labels_to_drop))
	meta_test = meta_test.drop(columns = ['Unnamed: 0', 'comp_id'])
	return image_test, title_test, desc_test, meta_test

image_test, title_test, desc_test, meta_test = separate(test)
image_train, title_train, desc_train, meta_train = separate(train)
train_labels = np.array(meta_train['views']).T
meta_train = meta_train.drop(columns = ['views'])
#Goal : [[image][desc][meta][title]] 986 of [[1258][50][29][50]]

def reposition(arrs):
	arrs = np.swapaxes(arrs,0,1)
	arrs = np.swapaxes(arrs,0,2)
	return arrs

def stack_uneven(arrays, fill_value=0):
    '''
    Fits arrays into a single numpy array, even if they are
    different sizes. `fill_value` is the default value.

    Args:
            arrays: list of np arrays of various sizes
                (must be same rank, but not necessarily same size)
            fill_value (float, optional):

    Returns:
            np.ndarray
    '''
    sizes = [a.shape for a in arrays]
    max_sizes = np.max(list(zip(*sizes)), -1)
    # The resultant array has stacked on the first dimension
    result = np.full((len(arrays),) + tuple(max_sizes), fill_value)
    for i, a in enumerate(arrays):
      # The shape of this array `a`, turned into slices
      slices = tuple(slice(0,s) for s in sizes[i])
      # Overwrite a block slice of `result` with this array `a`
      result[i][slices] = a
    result = reposition(result)
    return result


list_train = [np.array(image_train).T,np.array(title_train).T,np.array(meta_train).T,np.array(desc_train).T]
#list_test = [np.array(image_test).T,np.array(title_test).T,np.array(meta_test).T,np.array(desc_test).T]
train_data = stack_uneven(list_train)

#woohoo!!
#https://github.com/nitsourish/CNN-automated-Feature-Extraction/blob/master/CNN_feature_extraction.ipynb
import librosa
import librosa.display
import scipy.io.wavfile
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model,load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
from glob import glob
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import tqdm
import cv2
from keras import applications,models, losses,optimizers
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
import numpy as np
import keras.backend as K
from scipy.spatial import distance
from PIL import Image
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm,tqdm_pandas
import seaborn as sns

import tensorflow as tf
from keras import backend as K

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, 
                                    test_size = .33, random_state = 42)

X_train = np.reshape(X_train,(X_train.shape[0],4,1258,1))
X_test = np.reshape(X_test,(X_test.shape[0],4,1258,1))


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc
batch_size = 256
epochs = 100
filepath='keras_model_conv2D'

np.random.seed(1337)          # for reproducibility
print('Building model...')
model = Sequential()
model = Sequential()

#1st conv layer
model.add(Conv2D(32, (3,10), padding="same",
                 input_shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3]),data_format="channels_last"))
#model.add(BatchNormalization())
model.add(Activation("relu"))

2nd conv layer
model.add(Conv2D(32, (3,10), padding="same"))
model.add(BatchNormalization())
model.add(Activation("relu"))

#3rd conv layer
#model.add(Conv2D(32, (3,10), padding="same"))
#model.add(BatchNormalization())
#model.add(Activation("relu"))

#4th conv layer
#model.add(Conv2D(32, (3,10), padding="same"))
#model.add(BatchNormalization())
#model.add(Activation("relu"))
#model.add(MaxPooling2D())

model.add(Flatten())

#FC1
model.add(Dense(128))
#model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.3))

#FC2
model.add(Dense(100,name ='feature_dense'))
#model.load_weights(by_name=True,filepath = filepath)
#model.add(BatchNormalization())
model.add(Activation("relu"))

#output FC
model.add(Dense(2))
model.add(Activation('linear'))
adam = optimizers.Adam(lr=0.01)

model.compile(loss='mse', optimizer='adam')
model.summary()
'''
early_stops = EarlyStopping(patience=5, monitor='val_auc')
filepath='keras_model_conv2D'
ckpt_callback = ModelCheckpoint(filepath,
                                 monitor='val_loss', 
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='auto')
'''
model.fit(X_train, y_train, validation_split=0.05, epochs=epochs)
model.save('keras_model_conv2D.h5')

model = load_model('keras_model_conv2D.h5')
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('feature_dense').output)
intermediate_layer_model.summary()

#feauture_engg_data = intermediate_layer_model.predict(X_train)
#feauture_engg_data = pd.DataFrame(feauture_engg_data)
#print('feauture_engg_data shape:', feauture_engg_data.shape)
#feauture_engg_data.head(5)  #The features are unnamed now

model.train(X_train)


X_train_xg =intermediate_layer_model.predict(X_train)
X_test_xg = intermediate_layer_model.predict(X_test)

z = np.concatenate([X_train_xg,np.transpose([y_train])],axis=1)
z = pd.DataFrame(z)
z.to_csv('train_xg.csv',index=False) ###look more into what this is 

z = np.concatenate([X_test_xg,np.transpose([y_test])],axis=1)
z = pd.DataFrame(z)
z.to_csv('test_xg.csv',index=False) ###look more into what this is 


#####XGBoost#####################
import xgboost
train = pd.read_csv('train_xg.csv')
train_y = train['100'].astype('int')
train_x = train.drop(['100'],axis=1)
dtrain = xgboost.DMatrix(train_x, label=train_y)

test = pd.read_csv('test_xg.csv')
test_y = test['100'].astype('int')
test_x = test.drop(['100'],axis=1)
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

print('RMSE with XGBoost', rmse_xgboost) #our 1169 vs. medium post 1133

'''
#### Still to do: ########
###1 : figure out cnn
###2 : comparisons??? rmse, f1, precision, recall -- graph 
###3 : hyperparamter tuning 

d_public = xgb.DMatrix(p_final_df.loc[:, p_final_df.columns != 'comp_id'][bst.feature_names])
solution = bst.predict(d_public).round()
solution_df = pd.concat([p_final_df[['comp_id']], pd.DataFrame(solution, columns = ['views'])], axis=1)
solution_df.to_csv('solution.csv', index=False)
'''
