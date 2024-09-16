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

class MajorityBasedBasline:
    def majority_class_bot_message(utterance):
        df = pd.read_csv('dialog_act.csv')
        # result = random.choice(df['inform'])
        diag_acts= df['dialog_act']
        inform_df = df[df['dialog_act'] == 'inform']
        inform_values = inform_df['utterance'].tolist()
        random_value = random.choice(inform_values)+'\n'
        return random_value

    def majority_class_baseline(self,utterance):
        return 'inform'

    def majority_class_bot(self):
        flag= True
        msg = input("Hi I'm a bot is anything i can help you with:")
        while flag:
            bot_response=self.majority_class_baseline(msg)
            msg= input(bot_response)
            if msg == 'quit':
                flag= False
                
    def evaluate_majority_baseline_based_bot(self):
        data =pd.read_csv(r'raw_files/dialog_act_testnodupe.csv')
        data['predicted_dialogact'] = data['utterance'].apply(self.majority_class_baseline)

        accuracy = sum(data['dialog_act'] == data['predicted_dialogact']) / len(data)
        data.to_csv("rulebased_results.csv", index=False)
        print(f"accuracy: {accuracy:.2f}")  
    
# if __name__ == "__main__":
#     majority_class_bot()
#     # rule_based_chat_bot()
#     #data_prep()
#     # logistic_reg()
#     # evaluate_rule_based_bot()
#     # logistic_reg_train()
#     #logistic_reg_model_evaluate()
#     # descion_tree_train()
#     # desision_tree_model_evaluate()
#     evaluate_majority_baseline_based_bot()