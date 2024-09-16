import nltk

from nltk.stem import WordNetLemmatizer

import json
import pickle

import numpy as np
import pickle
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import random
import pandas as pd

class DecisionTree:
    def desision_tree_model_evaluate(self):
        with open(r'raw_files/decisiontreemodelnodupe.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open(r'raw_files/decisiontreevectorizernodupe.pkl', 'rb') as f:
            cv = pickle.load(f)
        data =pd.read_csv(r'raw_files/dialog_act_testnodupe.csv')
        data = data.fillna('no dialog')
        x_test = data['utterance']
        y_test = data['dialog_act']
        # cv = CountVectorizer()
        x_test_cv=cv.transform(x_test)
        result=lr.predict(x_test_cv)
        data['prediction']=result
        data.to_csv(r'raw_files/decisiontreeresultsnodupe.csv',index=False)
        accuracy = sum(data['dialog_act'] == data['prediction']) / len(data)
        print(f"accuracy of dt: {accuracy:.2f}")
        print(result)
        
                
    def descion_tree_train(self):
        data =pd.read_csv(r'raw_files/dialog_act_trainnodupe.csv')
        data = data.fillna('no dialog')
        x_train = data['utterance']
        y_train = data['dialog_act']    
        vectorizer = TfidfVectorizer()
        x_train_vector = vectorizer.fit_transform(x_train)
        dt = DecisionTreeClassifier(random_state=20)
        dt.fit(x_train_vector, y_train)
        test_sample=['thank you','how about thai']
        x_ts_vector=vectorizer.transform(test_sample)
        result=dt.predict(x_ts_vector)
        print(result)
        with open(r'raw_files/decisiontreemodelnodupe.pkl','wb') as f:
            pickle.dump(dt,f)
        with open(r'raw_files/decisiontreevectorizernodupe.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
