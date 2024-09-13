import nltk

from nltk.stem import WordNetLemmatizer

import json
import pickle

import numpy as np
from numpy.random import RandomState
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
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
    data.to_csv("dialog_act.csv", index=False)
    train.to_csv("dialog_act_train.csv",index=False)
    test.to_csv("dialog_act_test.csv",index =False)
    train_sample_test.to_csv('dialog_act_trainsample.csv',index =False)
def majority_class_baseline(utterance):
    df = pd.read_csv('dialog_act.csv')
    # result = random.choice(df['inform'])
    diag_acts= df['dialog_act']
    inform_df = df[df['dialog_act'] == 'inform']
    inform_values = inform_df['utterance'].tolist()
    random_value = random.choice(inform_values)+'\n'
    return random_value
def rule_based_dialogact_detect (utterance):
    if 'goodbye' in utterance:
        return 'bye'
    elif 'phone number' in utterance or 'phone' in utterance:
        return 'request'
    elif 'address' in utterance:
        return 'request'
    elif 'thank' in utterance:
        return 'thankyou'
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
            
def rule_based_chat_bot():
    flag= True
    msg = input("Hi I'm a bot is anything i can help you with:")
    while flag:
        bot_response=rulebased_class_baseline(msg)
        msg= input(bot_response)
        if msg == 'quit':
            flag= False
def train():
    nltk.download('punkt')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    words=[]
    classes = []
    documents = []
    ignore_words = ['?', '!']
    data_file = open('D:\FlaskChatbot\intents.json').read()
    intents = json.loads(data_file)
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))
    print (len(documents), "documents")
    print (len(classes), "classes", classes)
    print (len(words), "unique lemmatized words", words)
    pickle.dump(words,open('words.pkl','wb'))
    pickle.dump(classes,open('classes.pkl','wb'))
    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])
    random.shuffle(training)
    training = np.array(training,dtype=object)
    train_x = list(training[:,0])
    train_y = list(training[:,1])
    print("Training data created")


    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    model.save('chatbot_model.h5', hist)

if __name__ == "__main__":
    # majority_class_bot()
    # rule_based_chat_bot()
    data_prep()