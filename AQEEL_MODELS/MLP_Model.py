
# coding: utf-8

# In[28]:

__author__ = 'aqeel'
'''Train and evaluate a simple MLP on the Souq.com Reviews newswire topic classification task.
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python examples/NNClassifiyReviews.py
CPU run command:
    python examples/NNClassifiyReviews.py
'''
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
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


# ### Prepare the data

# In[29]:

train = pd.read_csv('../Data/train.csv')
ftest = pd.read_csv('../Data/test.csv')
train.columns


# In[33]:

def GetData(ds, splitper=0.2):
    np.random.seed(1337)
    #Convert The Percentage to split point
    splitper = int(math.floor(splitper * ds.shape[0] + 1))

    #Shuffle the list
    ds = ds.iloc[np.random.permutation(len(ds))]
    
    #Get tarin,test
    ls = range(10,18)
    ls+=[0,2]
    x_train = ds.iloc[splitper:][np.delete(ds.columns, ls)]
    y_train = ds.iloc[splitper:][ds.columns[10:11]]
    x_test = ds.iloc[:splitper][np.delete(ds.columns, ls)]
    y_test = ds.iloc[:splitper][ds.columns[10:11]]
    return (x_train.as_matrix(),y_train.as_matrix()),(x_test.as_matrix(),y_test.as_matrix())


# In[34]:

def baseline_model(indim,outdim):
    # create model
    model = Sequential()
    model.add(Dense(13, input_dim=indim, init='normal', activation='relu'))
    model.add(Dense(outdim, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# In[35]:

#1-inf
batch_size = 1
#1-inf
nb_epoch = 100
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
SecondActivation='linear'
#DONE
#mean_squared_error / mse,root_mean_squared_error / rmse,mean_absolute_error / mae,mean_absolute_percentage_error / mape
#mean_squared_logarithmic_error / msle,squared_hinge, hinge,binary_crossentropy: Also known as logloss,categorical_crossentropy: Also known as multiclass logloss. Note: using this objective requires that your labels are binary arrays of shape (nb_samples, nb_classes).
#poisson: mean of (predictions - targets * log(predictions))# cosine_proximity: the opposite (negative) of the mean cosine proximity between predictions and targets.
theloss='mean_absolute_error'
#======================
print('Loading data...')

#(X_train, y_train), (X_test, y_test) =GetData()
(x_train,y_train),(x_test,y_test) = GetData(train,splitper=0.1)

print x_train.shape,y_train.shape
print x_test.shape,y_test.shape
print x_train.shape[1]


# In[36]:

print('Building model...')
model = Sequential()
model.add(Dense(layernodes,init='normal', input_dim=x_train.shape[1]))
model.add(Activation(FirstActivation))
model.add(Dropout(thedropout))
model.add(Dense(y_train.shape[1],init='normal'))
model.add(Activation(SecondActivation))
model.compile(loss=theloss, optimizer=theoptimizer,metrics=["mean_absolute_error"])

early_stopping = EarlyStopping(monitor='val_loss', patience=31)
model_checkpoint = ModelCheckpoint('output_files/models/model_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
#checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True,monitor='val_acc',mode='max')
#losshistory= LossHistory()
history = model.fit(x_train, y_train, callbacks=[early_stopping, model_checkpoint, lr_reducer], nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_split=0.1)
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)
#results.write(','+str(score[1])+','+str(max(history.history.get('acc'))))
print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[27]:

print x_train[0],y_train[0]


# train.head()

# check = 'what the f**k'
# print check
# check = clean_str(check)
# print check
# #tokenizer.fit_on_texts(check)
# #Sequence The Training and Testing Set
# check = tokenizer.texts_to_sequences(check)
# print check
# check = tokenizer.sequences_to_matrix(check, mode=sequencemode)
# print check
# #model.predict_classes(x_train[0:10])

# model.save('Models/model_{}.h5'.format(sequencemode))

# score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

# clss = model.predict_classes(x_test)

# with open('Models/performance_{}.txt'.format(sequencemode),'w') as f:
#     f.write('Accuracy,PPrecision,NPrecision,Precall,Nrecall,\n')
#     f.write('{},{},{},{},{}\n'.format(Accuracy,PPrecision,NPrecision,Precall,Nrecall))
