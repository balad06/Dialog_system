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
from rulebased import RulebasedBaseline
from majority_based import MajorityBasedBasline
from logisticregrssion import LogisticReg
from decisiontree import DecisionTree

def data_prep():
    utterances = []
    dialog_act = []
    with open(r'raw_files/dialog_acts.dat', "r") as file:
        for line in file:
            parts = line.strip().split(maxsplit=1)  
            if len(parts) == 2:
                dialog_act.append(parts[0].strip().lower())  
                utterances.append(parts[1].strip().lower())  

    data = pd.DataFrame({'dialog_act': dialog_act, 'utterance': utterances})
    data= data.drop_duplicates()
    rng = RandomState()

    train = data.sample(frac=0.85, random_state=rng)
    train_sample_test= train.sample(frac =.2 ,random_state=rng)
    test = data.loc[~data.index.isin(train.index)]
    print(data.head())
    data.to_csv(r"raw_files/dialog_actnodupe.csv", index=False)
    train.to_csv(r"raw_files/dialog_act_trainnodupe.csv",index=False)
    test.to_csv(r"raw_files/dialog_act_testnodupe.csv",index =False)
    train_sample_test.to_csv(r'raw_files/dialog_act_trainsamplenodupe.csv',index =False)


if __name__ == "__main__":
    
    no = input('enter a choiceto evaluate the model 1) majority based 2) Rule based 3) Logistic reg 4) Descison tree (data) preprocess' )
    print(no)
    if no =='1':
        mjr=MajorityBasedBasline()
        mjr.evaluate_majority_baseline_based_bot()
    elif no =='2':
        rb =RulebasedBaseline()
        rb.evaluate_rule_based_bot()
    elif no =='3':
        lr = LogisticReg()
        lr.logistic_reg_train()
        lr.logistic_reg_model_evaluate()
    elif no =='4':
        dt= DecisionTree()
        dt.descion_tree_train()
        dt.desision_tree_model_evaluate()
    elif no == 'data':
        data_prep()
    else:
        print('enter valid number')
