import librosa
from librosa import display
import numpy as np
import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import sys
import shap
import pickle
from sklearn.metrics import accuracy_score

def DecisionTree(X_train, X_test, y_train, y_test):

  dtree = DecisionTreeClassifier()
  dtree.fit(X_train, y_train)
  predictions = dtree.predict(X_test)
  #print(classification_report(y_test,predictions))
  print(accuracy_score(y_test, predictions))

  return dtree

def RandomForest(X_train, X_test, y_train, y_test):

  rforest = RandomForestClassifier(criterion="gini", max_depth=10, max_features="log2", 
                                   max_leaf_nodes = 100, min_samples_leaf = 3, min_samples_split = 20, 
                                   n_estimators= 22000, random_state= 5)

  rforest.fit(X_train, y_train)
  predictions = rforest.predict(X_test)
  #print(classification_report(y_test,predictions))
  print(accuracy_score(y_test, predictions))

  return rforest

def NueralNetwork(X_train, X_test, y_train, y_test):

  x_traincnn = np.expand_dims(X_train, axis=2)
  x_testcnn = np.expand_dims(X_test, axis=2)

  #Begin constructing model layers
  model = Sequential()

  model.add(Conv1D(128, 5,padding='same',
                   input_shape=(40,1)))
  model.add(Activation('relu'))
  model.add(Dropout(0.1))
  model.add(MaxPooling1D(pool_size=(8)))
  model.add(Conv1D(128, 5,padding='same',))
  model.add(Activation('relu'))
  model.add(Dropout(0.1))
  model.add(Flatten())

  #Change the final decision
  if(sys.argv[1] == 'ravdess'):
    classnum = 8
  else:
    classnum = 7
  model.add(Dense(classnum))
  model.add(Activation('softmax'))
  opt = tf.keras.optimizers.RMSprop(learning_rate=0.00005, rho=0.9, epsilon=None, decay=0.0)

  model.summary()

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'])

  #Due to differences in the dataset, each set requires a different onehot label encoder
  le = preprocessing.LabelEncoder()
  if(sys.argv[1] == 'ravdess'):
    le.fit(["01", "02", "03", "04", "05", "06", "07", "08"])
  #1) anger; 2) boredom; 3) anxiety; 4) happiness; 5) sadness; 6) disgust; and 7) neutral

  #Annoyingly all of these are the first letter of the German words for these emotions
  elif(sys.argv[1] == 'emodb'): 
    le.fit(["F", "N", "W", "T", "A", "L", "E"])
    
  #Italian dataset, emotions are first three letters of: (neutral, disgust, joy, fear, anger, surprise, sadness)
  elif(sys.argv[1] == 'EMOVO'): 
    le.fit(["dis", "gio", "nue", "pau", "rab", "sor", "tri"])

  #The letters 'a', 'd', 'f', 'h', 'n', 'sa' and 'su' represent 'anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness' and 'surprise' emotion classes respectively. 
  elif(sys.argv[1] == 'savee'): 
    le.fit(["a", "d", "f", "h", "n", "sa", "su"])

  y_train_encoded = le.transform(y_train)
  y_test_encoded = le.transform(y_test)

  cnnhistory=model.fit(x_traincnn, y_train_encoded, batch_size=16, epochs=100, validation_data=(x_testcnn, y_test_encoded), verbose = 0)

  return model, cnnhistory, x_testcnn

def explainTree(model,X):
  shap.initjs()
  explainer = shap.TreeExplainer(model)
  #emotional subset for X
  shap_values = explainer.shap_values(X)
  #print(shap_values)
  shap.summary_plot(shap_values, features=X, class_names=model.classes_, max_display= 40)

def explainNN(model,xTrain,xTest):
  shap.initjs()
  explainer = shap.KernelExplainer(model.predict,xTrain)
  shap_values = explainer.shap_values(xTest,nsamples=100)
  #print(shap_values)
  shap.summary_plot(shap_values, class_names=model.classes_, max_display= 40)


if (sys.argv[1] not in ['ravdess', 'emodb', 'EMOVO', 'savee', '']):
  raise Exception("Argument Error: Please select an appropriate dataset: [ravdess, emodb, EMOVO, savee]")

if (sys.argv[2] not in ['all', 'tree', 'forest', 'NN']):
  raise Exception("Argument Error: Please select the models you would like to run: [decision tree (tree), random forest (forest), nueral network (NN), or all three")

if (sys.argv[3] not in ['pickle', 'pickled', 'pickless',]):
  raise Exception("Argument Error: Please specify whether you would like to pickle the MFFCs (pickle), run without pickleling (pickleless), or load a previously pickled list (pickled)")

dataset_name = sys.argv[1]
model_type = sys.argv[2]
path = 'C:/Users/ronna/Documents/GitHub/Audio-Privacy-Capstone/NNDatasets/NNDatasets/audio/' + sys.argv[1] #Change to appropriate path
print(path)
if (sys.argv[1] == 'emodb'):
  path = path + "/wav"
lst = []

output_dir = 'model_pickles/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#decide whether to use an already pickled list, pickle a new list, or run without pickling at all
#pickle - pickle a new list
#pickled - use an already pickled list
#pickleless - run without pickling

if sys.argv[3] == 'pickle' or sys.argv[3] == 'pickleless':
  for subdir, dirs, files in os.walk(path):
    for file in files:
        try:
          #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
          X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
          mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
          #The emotion label for each audio file is saved in a different portion of the filename for
          #each dataset. Must subset appropriately
          if(sys.argv[1] == 'ravdess'):
            file = file[6:8]
          elif(sys.argv[1] == 'emodb'): 
            file = file[5]
          elif(sys.argv[1] == 'EMOVO'): 
            file = file[0:3]
          elif(sys.argv[1] == 'savee'): 
            #The way the authors formatted this is rather cumbersome.
            #Most emotions are labelled as a single letter in the filename, but some or two letters. Thus we need to do some work.
            if(file[0] == "s"):
              file = file[0:2]
            else:
              file = file[0]
          #mask = np.ones(len(mfccs),dtype=bool)
          #mask[[0,30,39,6]] = False
          arr = mfccs, file
          lst.append(arr)
        # If the file is not valid, skip it
        except ValueError:
          continue

    #pickle if necessary
    if(sys.argv[3] == 'pickle'):
      with open(output_dir + str(dataset_name) + '.pkl', 'wb') as f:
        pickle.dump(lst, f)

#load if necessary
elif(sys.argv[3] == 'pickled'):
  f = open(output_dir + str(dataset_name) + '.pkl', 'rb')
  lst = pickle.load(f)

for n in range(40):
  mask = np.ones(40,dtype=bool)
  ind = [0,30,39,6,21,34,25,17,13,31,19,33,36,27,29,4,2,32,14,12,37,5,9,11,18,16,35,15,38,1,8,23,22,28,26,24,3,10,7,20]
  ind = ind[0:n]
  mask[ind] = False

  X, y = zip(*lst)
  X = np.asarray(X)
  X = X[:,mask]
  y = np.asarray(y)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  #which model to run? defaults to all

  if(model_type == 'tree'):
    treeModel = DecisionTree(X_train, X_test, y_train, y_test)
    #explainTree(treeModel,X)
  elif(model_type == 'forest'):
    rfModel = RandomForest(X_train, X_test, y_train, y_test)
    explainTree(rfModel,X)
  elif(model_type == 'NN'):
    nnModel, cnnhistory, nnxTest = NueralNetwork(X_train, X_test, y_train, y_test)
    #explainNN(nnModel,X_train,X_test)
  else: #Else we run all
    treeModel = DecisionTree(X_train, X_test, y_train, y_test)
    rfModel = RandomForest(X_train, X_test, y_train, y_test)
    nnModel, cnnhistory, nnxTest = NueralNetwork(X_train, X_test, y_train, y_test)
    explainTree(treeModel,X)
    explainTree(rfModel,X)
    explainNN(nnModel,X_train,X_test)

  # plt.plot(cnnhistory.history['loss'])
  # plt.plot(cnnhistory.history['val_loss'])
  # plt.title('model loss')
  # plt.ylabel('loss')
  # plt.xlabel('epoch')
  # plt.legend(['train', 'test'], loc='upper left')
  # plt.show()
