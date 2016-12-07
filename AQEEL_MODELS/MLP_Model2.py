
# coding: utf-8

# In[2]:

__author__ = 'aqeel'
'''Train and evaluate a simple MLP on the Souq.com Reviews newswire topic classification task.
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/NNClassifiyReviews.py
CPU run command:
    python examples/NNClassifiyReviews.py
'''
import numpy as np
from keras.models import Sequential, load_model,Model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.recurrent import LSTM
import random
import math
import pandas as pd
import re

#For the baseline
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from ALutils import Calculate_Score,RANK
np.random.seed(1377)


# ### Prepare the data

# In[3]:

train = pd.read_csv('../Data/train.csv')
ftest = pd.read_csv('../Data/test.csv')
#ftest[ftest.columns[[0]+[i for i in range(10,18)]]]
#train.head()
def GetData(ds):#, splitper=0.2): Splitter is stopped 
    np.random.seed(1337)
    #Convert The Percentage to split point
    splitper = 50 #int(math.floor(splitper * ds.shape[0] + 1))

    #Shuffle the list
    #Shuffle is stopped so we can get stable measurements
    #ds = ds.iloc[np.random.permutation(len(ds))]
    
    #Get tarin,test
    ls = [i for i in range(10,18)]
    ls+=[0,2]
    x_train = ds.iloc[splitper:][np.delete(ds.columns, ls)]*1000
    y_train = ds.iloc[splitper:][ds.columns[10:18]]*1000
    x_test = ds.iloc[:splitper][np.delete(ds.columns, ls)]*1000
    y_test = ds.iloc[:splitper][ds.columns[10:18]]*1000
    return (x_train.as_matrix(),y_train.as_matrix()),(x_test.as_matrix(),y_test.as_matrix()) 


# In[4]:

#1-inf
batch_size = 1
#1-inf
nb_epoch = 1000
#Done
#SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax
theoptimizer = 'adam'
#DONE
#1-inf
layernodes = 128
#DONE
#0.1-0.9
thedropout =0.5
#DONE
#softmax,softplus,relu,tanh,sigmoid,hard_sigmoid,linear,
FirstActivation = 'relu'
SecondActivation='sigmoid'
#DONE
#mean_squared_error / mse,root_mean_squared_error / rmse,mean_absolute_error / mae,mean_absolute_percentage_error / mape
#mean_squared_logarithmic_error / msle,squared_hinge, hinge,binary_crossentropy: Also known as logloss,categorical_crossentropy: Also known as multiclass logloss. Note: using this objective requires that your labels are binary arrays of shape (nb_samples, nb_classes).
#poisson: mean of (predictions - targets * log(predictions))# cosine_proximity: the opposite (negative) of the mean cosine proximity between predictions and targets.
theloss='mse'
#======================
print('Loading data...')

#(X_train, y_train), (X_test, y_test) =GetData()
(x_train,y_train),(x_test,y_test) = GetData(train)
#y_train = y_train[:,0]
#y_test = y_test[:,0]
print('train:',x_train.shape,y_train.shape)
print('test: ',x_test.shape,y_test.shape)


# In[4]:

print('Building model...')
model = Sequential()
model.add(Dense(128,init='normal', input_dim=x_train.shape[1]))
model.add(Activation(SecondActivation))
model.add(Dropout(thedropout))
model.add(Dense(8,init='normal'))
model.add(Activation(FirstActivation))
model.compile(loss=theloss, optimizer=theoptimizer,metrics=[theloss])

early_stopping = EarlyStopping(monitor='val_loss', patience=42)
model_checkpoint = ModelCheckpoint('output_files/Model_NN:2_2_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='min')
history = model.fit(x_train, y_train, nb_epoch=nb_epoch,callbacks=[early_stopping, model_checkpoint], batch_size=batch_size,verbose=1, validation_split=0.1)

score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
#results.write(','+str(score[1])+','+str(max(history.history.get('acc'))))
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[14]:

org = RANK(y_test)
print('Perfect Score:',Calculate_Score(org,org))
#bm = load_model('output_files/t0/Model_01-106548.69.h5')
#model = load_model('output_files/t0/Model_05-0.12.h5')
pred = model.predict(x_test)
pred = RANK(pred)
print('Current Score:',Calculate_Score(pred,org))
