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
from keras.optimizers import SGD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import random
import pandas as pd

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
    rng = RandomState()

    train = data.sample(frac=0.85, random_state=rng)
    train_sample_test= train.sample(frac =.2 ,random_state=rng)
    test = data.loc[~data.index.isin(train.index)]
    print(data.head())
    data.to_csv(r"raw_files/dialog_act.csv", index=False)
    train.to_csv(r"raw/files/dialog_act_train.csv",index=False)
    test.to_csv(r"raw_files/dialog_act_test.csv",index =False)
    train_sample_test.to_csv(r'raw_files/dialog_act_trainsample.csv',index =False)

def majority_class_bot_message(utterance):
    df = pd.read_csv('dialog_act.csv')
    # result = random.choice(df['inform'])
    diag_acts= df['dialog_act']
    inform_df = df[df['dialog_act'] == 'inform']
    inform_values = inform_df['utterance'].tolist()
    random_value = random.choice(inform_values)+'\n'
    return random_value

def majority_class_baseline(utterance):
    return 'inform'

def rule_based_dialogact_detect (utterance):
    if ("kay" in utterance or "okay and" in utterance or "okay um" in utterance):
        return 'ack'
    elif ("yes" in utterance and len(utterance)==3 or "yeah" in utterance or "yea" in utterance or "yes" in utterance):
        return 'affirm'
    elif ("is it" in utterance or "do they" in utterance or "does it" in utterance):
        return 'confirm'
    elif ("wrong" in utterance or "dont want" in utterance or "no not" in utterance or "change" in utterance or "not" in utterance):
        return 'deny'
    elif ("hi" in utterance  or "hello" in utterance ):
        return 'hello'
    elif ("looking for" in utterance or "any" in utterance or "seafood" in utterance or "mediterranean" in utterance or "east" in utterance):
        return 'inform'
    elif ("no" in utterance):
        return 'negate'
    elif ("welcome" in utterance or "okay" in utterance or "noise" in utterance):
        return 'null'
    elif ("else" in utterance or "how about" in utterance or "anything else" in utterance):
        return 'reqalts'
    elif ("what" in utterance or "whats" in utterance or "could i" in utterance):
        return 'request'
    elif ("start" in utterance or "reset" in utterance):
        return 'restart'
    elif ("thank you" in utterance or "thankyou" in utterance):
        return 'thankyou'
    elif ("goodbye" in utterance or "good bye" in utterance):
        return 'bye'
    elif ("again" in utterance or "go back" in utterance or "back" in utterance):
        return 'repeat'
    elif ("more" in utterance):
        return 'reqmore'
    else:
        return 'inform'
    

def rulebased_class_baseline(utterance):
    df = pd.read_csv('dialog_act.csv')
    # result = random.choice(df['inform'])
    diag_acts= df['dialog_act']
    msg_intent=rule_based_dialogact_detect(utterance)
    inform_df = df[df['dialog_act'] == msg_intent]
    inform_values = inform_df['utterance'].tolist()
    random_value = random.choice(inform_values)+'\n'
    return random_value


def majority_class_bot():
    flag= True
    msg = input("Hi I'm a bot is anything i can help you with:")
    while flag:
        bot_response=majority_class_baseline(msg)
        msg= input(bot_response)
        if msg == 'quit':
            flag= False
def evaluate_majority_baseline_based_bot():
    data =pd.read_csv(r'raw_files/dialog_act_test.csv')
    data['predicted_dialogact'] = data['utterance'].apply(majority_class_baseline)

    accuracy = sum(data['dialog_act'] == data['predicted_dialogact']) / len(data)
    data.to_csv("rulebased_results.csv", index=False)
    print(f"accuracy: {accuracy:.2f}")         
def evaluate_rule_based_bot():
    data =pd.read_csv(r'raw_files/dialog_act_test.csv')
    data['predicted_dialogact'] = data['utterance'].apply(rule_based_dialogact_detect)

    accuracy = sum(data['dialog_act'] == data['predicted_dialogact']) / len(data)
    data.to_csv("rulebased_results.csv", index=False)
    print(f"accuracy: {accuracy:.2f}")  
def rule_based_chat_bot():
    flag= True
    msg = input("Hi I'm a bot is anything i can help you with:")
    while flag:
        bot_response=rulebased_class_baseline(msg)
        msg= input(bot_response)
        if msg == 'quit':
            flag= False
            

def evaluate_rule_based_bot():
    data =pd.read_csv('dialog_act_test.csv')
    data['predicted_dialogact'] = data['utterance'].apply(rule_based_dialogact_detect)

    accuracy = sum(data['dialog_act'] == data['predicted_dialogact']) / len(data)
    data.to_csv("rulebased_results.csv", index=False)
    print(f"accuracy: {accuracy:.2f}")            


def logistic_reg_model_evaluate():
    with open(r'raw_files/logisticmodel.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open(r'raw_files/logisticvectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    data =pd.read_csv(r'raw_files/dialog_act_test.csv')
    data = data.fillna('no dialog')
    x_test = data['utterance']
    y_test = data['dialog_act']
    # cv = CountVectorizer()
    x_test_cv=cv.transform(x_test)
    result=lr.predict(x_test_cv)
    data['prediction']=result
    data.to_csv(r'raw_files/logisticmodelresults.csv',index=False)
    accuracy = sum(data['dialog_act'] == data['prediction']) / len(data)
    print(f"accuracy: {accuracy:.2f}")
    print(result)

def logistic_reg_train():
    data =pd.read_csv(r'raw_files/dialog_act_train.csv')
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
    with open(r'raw_files/logisticmodel.pkl','wb') as f:
        pickle.dump(lr,f)
    with open(r'raw_files/logisticvectorizer.pkl', 'wb') as f:
        pickle.dump(cv, f)
        
        
def desision_tree_model_evaluate():
    with open(r'raw_files/decisiontreemodel.pkl', 'rb') as f:
        lr = pickle.load(f)
    with open(r'raw_files/decisiontreevectorizer.pkl', 'rb') as f:
        cv = pickle.load(f)
    data =pd.read_csv(r'raw_files/dialog_act_test.csv')
    data = data.fillna('no dialog')
    x_test = data['utterance']
    y_test = data['dialog_act']
    # cv = CountVectorizer()
    x_test_cv=cv.transform(x_test)
    result=lr.predict(x_test_cv)
    data['prediction']=result
    data.to_csv(r'raw_files/decisiontreeresults.csv',index=False)
    accuracy = sum(data['dialog_act'] == data['prediction']) / len(data)
    print(f"accuracy: {accuracy:.2f}")
    print(result)
    
            
def descion_tree_train():
    data =pd.read_csv(r'raw_files/dialog_act_train.csv')
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
    with open(r'raw_files/decisiontreemodel.pkl','wb') as f:
        pickle.dump(dt,f)
    with open(r'raw_files/decisiontreevectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
if __name__ == "__main__":
    # majority_class_bot()
    # rule_based_chat_bot()
    #data_prep()
    # logistic_reg()
    # evaluate_rule_based_bot()
    # logistic_reg_train()
    #logistic_reg_model_evaluate()
    # descion_tree_train()
    # desision_tree_model_evaluate()
    evaluate_majority_baseline_based_bot()