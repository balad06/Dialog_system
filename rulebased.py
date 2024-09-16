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


class  RulebasedBaseline:
    def evaluate_rule_based_bot(self):
        data =pd.read_csv(r'raw_files/dialog_act_testnodupe.csv')
        data = data.fillna('no dialog')
        data['predicted_dialogact'] = data['utterance'].apply(self.rule_based_dialogact_detect)

        accuracy = sum(data['dialog_act'] == data['predicted_dialogact']) / len(data)   
        data.to_csv(r"raw_files/rulebased_resultsnodupe.csv", index=False)
        print(f"accuracy: {accuracy:.2f}")  

    def rule_based_chat_bot(self):
        flag= True
        msg = input("Hi I'm a bot is anything i can help you with:")
        while flag:
            bot_response=self.rulebased_class_baseline(msg,self)
            msg= input(bot_response)
            if msg == 'quit':
                flag= False
            
    def rule_based_dialogact_detect (self,utterance):
        if (" kay" in utterance or "okay and" in utterance or "okay um" in utterance):
            return 'ack'
        elif ("yes" in utterance and len(utterance)==3 or "yeah" in utterance or "yea" in utterance or "yes" in utterance):
            return 'affirm'
        elif ("is it" in utterance or "do they" in utterance or "does it" in utterance):
            return 'confirm'
        elif ("wrong" in utterance or "dont want" in utterance or "no not" in utterance or "change" in utterance or "not" in utterance):
            return 'deny'
        elif ("hi" in utterance  or "hello" in utterance ):
            return 'hello'
        elif ("what" in utterance or "whats" in utterance or "could i" in utterance or 'can i' in utterance or 'address' in utterance or 'number' in utterance or 'post code' in utterance):
            return 'request'
        elif ("looking for" in utterance or "any" in utterance or "seafood" in utterance or "mediterranean" in utterance or "east" in utterance or 'type' in utterance or 'north' in utterance or 'fusion food' in utterance ):
            return 'inform'
        elif ("no " in utterance):
            return 'negate'

        elif ("else" in utterance or "how about" in utterance or "anything else" in utterance):
            return 'reqalts'
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
        elif ("welcome" in utterance or "okay" in utterance or "noise" in utterance):
            return 'no dialog'
        else:
            return 'inform'
    

    def rulebased_class_baseline_bot(self,utterance):
        df = pd.read_csv('dialog_act.csv')
        # result = random.choice(df['inform'])
        diag_acts= df['dialog_act']
        msg_intent=self.rule_based_dialogact_detect(utterance)
        inform_df = df[df['dialog_act'] == msg_intent]
        inform_values = inform_df['utterance'].tolist()
        random_value = random.choice(inform_values)+'\n'
        return random_value
          
          
# if __name__ == "__main__":
    # majority_class_bot()
    # rb= RulebasedBaseline()
    # rb.rule_based_chat_bot()
    #data_prep()
    # logistic_reg()
    # rb.evaluate_rule_based_bot()
    # logistic_reg_train()
    #logistic_reg_model_evaluate()
    # descion_tree_train()
    # desision_tree_model_evaluate()
    # evaluate_majority_baseline_based_bot()