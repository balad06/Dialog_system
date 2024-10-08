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

class LogisticReg:
    def logistic_reg_model_evaluate(self):
        with open(r'raw_files/logisticmodelnodupe.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open(r'raw_files/logisticvectorizernodupe.pkl', 'rb') as f:
            cv = pickle.load(f)
        data =pd.read_csv(r'raw_files/dialog_act_testnodupe.csv')
        data = data.fillna('no dialog')
        x_test = data['utterance']
        y_test = data['dialog_act']
        # cv = CountVectorizer()
        x_test_cv=cv.transform(x_test)
        result=lr.predict(x_test_cv)
        data['prediction']=result
        data.to_csv(r'raw_files/logisticmodelresultsnodupe.csv',index=False)
        accuracy = sum(data['dialog_act'] == data['prediction']) / len(data)
        print(f"accuracy of lr: {accuracy:.2f}")
        # print(result)

    def logistic_reg_train(self):
        data =pd.read_csv(r'raw_files/dialog_act_trainnodupe.csv')
        data = data.fillna('no dialog')
        x_train = data['utterance']
        y_train = data['dialog_act']
        lr = LogisticRegression()
        cv = CountVectorizer()
        x_train_cv = cv.fit_transform(x_train)
        
        lr.fit(x_train_cv, y_train)
        test_sample=['thank you','how about thai']
        x_ts_cv=cv.transform(test_sample)
        result=lr.predict(x_ts_cv)
        print(result)
        with open(r'raw_files/logisticmodelnodupe.pkl','wb') as f:
            pickle.dump(lr,f)
        with open(r'raw_files/logisticvectorizernodupe.pkl', 'wb') as f:
            pickle.dump(cv, f)
        
