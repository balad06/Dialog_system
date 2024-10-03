import pickle
import nltk
from keybert import KeyBERT
import pandas as pd
import spacy
import os
import Levenshtein
import nltk.corpus
from nltk.corpus import wordnet as wn

import random
import json

class Chatbot:
    def __init__(self):
        self.current_state = 'hello'
        self.prevbotresp = ''
        pass
    def textclassfier(self,msg):
        with open(r'raw_files/logisticmodelnodupe.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open(r'raw_files/logisticvectorizernodupe.pkl', 'rb') as f:
            cv = pickle.load(f)
        msg_cv=cv.transform([msg])
        result=lr.predict(msg_cv)    
        return result[0]
    
    def prepresponse(self,message,intent,results,matched_columns):
        #Use for preparping responses for diffrent dialog acts
        if self.current_state=='hello':
            if len(results)>0:
                selected_restaurant=results.sample(n=1)
                response_restarent=selected_restaurant.iloc[0].to_dict()
                selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
                missing_columns=list(set(['pricerange','area','food'])-set(matched_columns))
                if len(missing_columns)==0: 
                # matched_columns==['pricerange','area','food']:
                    bot_msg='Do you have any other additional preferences?\n'
                    # bot_msg='Would you like to try {}'.format(response_restarent['restaurantname'])
                    self.current_state='select_prefs'
                else:
                    missing_columns=list(set(['pricerange','area','food'])-set(matched_columns))
                    missing_columns_str=','.join(missing_columns)
                    bot_msg='Do you have any preferences of {} ?'.format(missing_columns_str)
                    self.current_state='select_prefs'
            else:
                keywords=self.keyextractions(message)
                results=self.closest_results(keywords,results)
                if len(results)==0:
                    if intent=='inform' or intent =='request':
                        bot_msg='Could not find a match wanna look for something else\n'
                    else:
                        bot_msg = self.generic_response(intent,message)
                else:
                    selected_restaurant=results.sample(n=1)
                    response_restarent=selected_restaurant.iloc[0].to_dict()
                    # selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
            
                    if response_restarent['min_dist']=='lv_pricerange': 
                        bot_msg= 'Did you mean {} price range If so do you have any other preferences\n'.format(response_restarent['pricerange'])
                        matched_columns.append('pricerange')
                    if response_restarent['min_dist']=='lv_area':
                        bot_msg= 'Did you mean {} price range If so do you have any other preferences\n'.format(response_restarent['area'])
                        matched_columns.append('area')                        
                    if response_restarent['min_dist']=='lv_food':
                        bot_msg= 'Did you mean {} price range If so do you have any other preferences\n'.format(response_restarent['food'])
                        matched_columns.append('food')
                    
            return bot_msg,matched_columns
        if self.current_state=='select_prefs':
            if len(results)>0:
                selected_restaurant=results.sample(n=1)
                response_restarent=selected_restaurant.iloc[0].to_dict()
                selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
                missing_columns=list(set(['pricerange','area','food'])-set(matched_columns))
                if len(missing_columns)==0:
                    bot_msg='Do you have any other another preferences?\n'
                    # bot_msg='Would you like to try {}'.format(response_restarent['restaurantname'])
                    self.current_state='select_prefs'
                else:
                    missing_columns_str=','.join(missing_columns)
                    bot_msg='Do you have any preferences of {} ?'.format(missing_columns_str)
                    self.current_state='select_prefs'
            else:
                keywords=self.keyextractions(message)
                results=self.closest_results(keywords,results)
                if len(results)==0:
                    if intent=='inform' or intent =='request':
                        bot_msg='Could not find a match wanna look for do you have someother preferences\n'
                    else:
                        bot_msg = self.generic_response(intent,message)
                else:
                    selected_restaurant=results.sample(n=1)
                    response_restarent=selected_restaurant.iloc[0].to_dict()
                    selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
            
                    if response_restarent['min_dist']=='lv_pricerange': 
                        bot_msg= 'Did you mean {} price range If so do you have any other preferences\n'.format(response_restarent['pricerange'])
                        matched_columns=['pricerange']
                    if response_restarent['min_dist']=='lv_area':
                        bot_msg= 'Did you mean {} area If so do you have any other preferences\n'.format(response_restarent['area'])
                        matched_columns=['area']                        
                    if response_restarent['min_dist']=='lv_food':
                        bot_msg= 'Did you mean {} food If so do you have any other preferences\n'.format(response_restarent['food'])
                        matched_columns=['food']
        
        
            return bot_msg,matched_columns
    
    def keyextractions(self,msg):
        nlp = spacy.load("en_core_web_sm")#REmove stop words
        listRemove=['restaurants','hotels','hotel','restaurant','food','hi','hello',"greetings", "welcome", "thanks"]
        for i in listRemove:
            nlp.Defaults.stop_words.add(i)
        text = msg
        doc = nlp(text)
        filtered_words = [token.text for token in doc if not token.is_stop]
        clean_text = ' '.join(filtered_words)
        kw_model = KeyBERT()
        keywords = kw_model.extract_keywords(clean_text)
        return keywords
    
    def levenshtein_distance(self,value, target):
        if isinstance(value, str):
            return Levenshtein.distance(value, target)
        else:
            return None
    
    def closest_results(self,keywords,results):
        if len(results)==0:
            df =pd.read_csv(r'raw_files/restaurant_info_withrecommendations.csv')
        else:
            df=results
        if keywords!=[]:
            for keyword in keywords:
                keyword = keyword[0]
                #to avoid unwanted loops
                for column in ['pricerange','area','food']:
                     df['lv_{}'.format(column)] = df[column].apply(lambda x: self.levenshtein_distance(x, keyword))
            df['lvDistance'] = df[['lv_pricerange', 'lv_area', 'lv_food']].min(axis=1)
            min = df['lvDistance'].min()
            df['min_dist'] = df[['lv_pricerange', 'lv_area', 'lv_food']].idxmin(axis=1)
            df.drop(['lv_pricerange', 'lv_area', 'lv_food'], axis=1, inplace=True)
            if min<5.0:
                closest_restaurants = df[df['lvDistance'] == min]
                closest_restaurants.to_csv(r'raw_files/closest_results.csv',index=False)
            return closest_restaurants
        else:
            return  pd.DataFrame(columns=df.columns)
    def get_results(self,df,message,matched_cols_prev):
        keywords=self.keyextractions(message)
        matched_restarent =pd.DataFrame(columns=df.columns)
        mactched_columns=matched_cols_prev
        for keyword in keywords:
            keyword = keyword[0]
                #to avoid unwanted loops
            for column in list(set(['pricerange','area','food']) - set(matched_cols_prev)) :
                matches = df[df[column].str.contains(keyword, case=False, na=False)]
                if not matches.empty:
                    if matched_restarent.empty:
                        matched_restarent=matches
                        mactched_columns.append(column)
                    else:
                        mactched_columns.append(column)
                        matches = matches.rename(columns={'restaurantname_x': 'restaurantname', 'area_x': 'area','pricerange_x':'pricerange','food_x':'food','addr_x':'addr','postcode_x':'postcode'})
                        matched_restarent = matched_restarent.rename(columns={'restaurantname_x': 'restaurantname', 'area_x': 'area','pricerange_x':'pricerange','food_x':'food','addr_x':'addr','postcode_x':'postcode'})
                        matched_restarent = pd.merge(matched_restarent, matches, how='inner',on=['restaurantname','pricerange','area','food','phone','addr','postcode'])

        matched_restarent.to_csv(r'raw_files/filtered_hotels.csv',index=False)
        # mactched_columns_json={'matchedColumns':mactched_columns}
        matchedcolumnstr=','.join(mactched_columns)
        with open(r"raw_files/matched_Columns.txt", "w") as f:
            f.write(matchedcolumnstr)
        return matched_restarent,mactched_columns
    
    def alternate_response(self,filtered_hotels,old_restaurant):
        new_filtered = pd.concat([filtered_hotels, old_restaurant]).drop_duplicates(keep=False)
        if len(new_filtered)>0:
            new_restaurant=new_filtered.sample(n=1)
        else:
            new_restaurant=pd.DataFrame()        
        return new_restaurant
        
    def generic_response(self,intent,msg):
        df = pd.read_csv(r'raw_files/dialog_act.csv')
        # result = random.choice(df['inform'])
        
        df_filtered = df[df['dialog_act'] == intent]
        inform_values = df_filtered['utterance'].tolist()
        random_value = random.choice(inform_values)+'\n'
        return random_value
    
    def additional_details(self,msg):
        details=[]
        selected_hotel= pd.read_csv(r'raw_files/selected_restaurant.csv')
        response_restarent=selected_hotel.iloc[0].to_dict()
        keywords=self.keyextractions(msg)
        keywords, scores = map(list, zip(*keywords))
        
        if 'address' in keywords:
            details.append(response_restarent['addr'])
            # response =' The address of the restaurant is {}'.format(response_restarent['addr'])
            # return response

        if 'phone' in keywords or 'number' in keywords:
            details.append(response_restarent['phone'])
            # response =' The number of the restaurant is {}'.format(response_restarent['phone'])

        if 'postcode' in keywords or 'post' in keywords:
            details.append(response_restarent['postcode'])
            # response =' The post code of the restaurant is {}'.format(response_restarent['postcode'])
            # return response
        return details
    
    def recommendations(self,msg):
        keywords=self.keyextractions(msg)
        keywords, scores = map(list, zip(*keywords))
        filtered_hotels=pd.read_csv(r'raw_files/filtered_hotels.csv')
        synonyms = {}
        synonyms = {
        "touristic": [
        "touristic",
        "touristy",
        "travel-related",
        "vacation-oriented",
        "leisure-focused",
        "sightseeing",
        "visitor-friendly",
        "recreational",
        "holiday-centric",
        "excursion-based",
        "traveler-focused",
        "travel friendly"
        ]
        }
        for word in ['children','romantic']:
            wrdsynms=[]
            for syn in wn.synsets(word):
                for i in syn.lemmas():
                    wrdsynms.append(i.name())
            synonyms[word]=list(set(wrdsynms))
        
        for keyword in keywords:
            if keyword in synonyms['touristic']:
                filtered_hotels=filtered_hotels[filtered_hotels['food'] != "romanian"]
                filtered_hotels = filtered_hotels[((filtered_hotels['pricerange'] == "cheap") 
                                                  & (filtered_hotels['food quality'] == "good"))] 
                reason ='it is touristic because it is cheap and has good food'
            if (keyword in synonyms['children'] or keyword == 'children'):
                filtered_hotels = filtered_hotels[filtered_hotels['length of stay'] == "short stay"]
                reason = "it child-friendly, because spending a long time is not advised when taking children"
            if keyword in synonyms['romantic']:
                filtered_hotels = filtered_hotels[(filtered_hotels['crowdedness'] == "not busy") 
                                                  | (filtered_hotels['length of stay'] == "long stay")]
                reason = "it is romantic, because it allows you to stay for a long time or is not busy"
            if keyword=="assigned seats":
                filtered_hotels = filtered_hotels[filtered_hotels['crowdedness'] == "busy"]
                reason =" it has assigned seats, because in a busy restaurant the waiter decides where you sit"
        
        if len(filtered_hotels)==0:
            bot_msg='Sorry could not find the hotel with your requirements want to try something else>\n'
        else:
            selected_restaurant=filtered_hotels.sample(n=1)
            response_restarent=selected_restaurant.iloc[0].to_dict()
            filtered_hotels.to_csv(r'raw_files/filtered_hotels.csv',index=False)
            selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
            bot_msg='Would you like to try {} '.format(response_restarent['restaurantname'])+ reason+'\n'
            self.current_state='suggest_rest'
        return bot_msg
    
    def bot_response(self,message):
        message = message.lower()
        bot_msg=''
        intent=self.textclassfier(message)
        df =pd.read_csv(r'raw_files/restaurant_info_withrecommendations.csv')
        matched_columns=[]
        results=pd.DataFrame()
        config={}
        with open(r'config.txt', "r") as file:
            config_str=file.read()
        config=json.loads(config_str)
        if intent in ['restart']:
            if config['allow_Restart']=='true': 
                self.current_state='hello'
                bot_msg='Sure,I\'m a bot is there something you are looking for?\n'
                return bot_msg
            else:
                return 'Cannot start over try chamging the configuration'
        else:
            if self.current_state == 'hello':
                if intent in ['hello','inform']:
                    results,matched_columns=self.get_results(df,message,matched_columns)           
                # if intent not in ['hello','inform','no dialog','reqalts','request','reqmore']:
                #     results=pd.DataFrame()
                    bot_msg,matched_columns=self.prepresponse(message,intent,results,matched_columns)
                if intent=='no dialog':
                    bot_msg='Sorry I could not understand want to try something else?\n'
                if config['caps'] == 'true':
                    bot_msg=bot_msg.upper()
                self.prevbotresp=bot_msg
                return bot_msg            
            elif self.current_state == 'select_prefs':
                results =pd.read_csv(r'raw_files/filtered_hotels.csv')
                if intent == 'inform' or intent == 'no dialog':
                    with open(r"raw_files/matched_Columns.txt", "r") as f:
                        matched_columns_str= f.read()
                    matched_columns=list(matched_columns_str.split(','))
                    if list(set(['pricerange','area','food']) - set(matched_columns)) ==[]:
                        bot_msg = self.recommendations(message)
                    else:
                        results,matched_columns=self.get_results(results,message,matched_columns)
                        bot_msg,matched_columns=self.prepresponse(message,intent,results,matched_columns)
                if intent == 'negate':
                    prevbotresp=self.prevbotresp.lower()
                    if ('area' in prevbotresp or 'pricerange' in prevbotresp or 
                        'price range' in prevbotresp or 'food' in prevbotresp):
                        bot_msg="Do you have any other addtional preferences?\n"
                        matchedcolumnstr=','.join(['pricerange','area','food'])
                        with open(r"raw_files/matched_Columns.txt", "w") as f:
                            f.write(matchedcolumnstr)  
                    else:
                        selected_restaurant=results.sample(n=1)
                        response_restarent=selected_restaurant.iloc[0].to_dict()
                        selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
                        bot_msg='Would you like to try {} '.format(response_restarent['restaurantname'])
                        self.current_state='suggest_rest'
                if (intent in['bye','thankyou'] or 'bye' in message):
                    bot_msg ='Bye See you\n'
                    self.current_state='hello' 
                if (intent in['ack','affirm','confirm']):
                    bot_msg ='Cool is there anything else you need\n'
                # if intent == 'no dialog':
                    # bot_msg ='Sorry did not get that try "start over" if you want to find a new restaurant\n'
                if config['caps'] == 'true':
                    bot_msg=bot_msg.upper()
                self.prevbotresp=bot_msg
                return bot_msg
            elif self.current_state == 'suggest_rest':
                if (intent in['bye','thankyou'] or 'bye' in message):
                    bot_msg ='Bye See you\n'
                    self.current_state='hello' 
                if (intent in['ack','affirm','confirm']):
                    bot_msg ='Cool is there anything else you need\n'
                # if intent in['deny','negate']:
                #     bot_msg='Sorry to hear that what else would you like\n'
                #     self.current_state='hello'
                if intent in ['reqalts','reqmore','deny','negate']:
                    df=pd.read_csv(r'raw_files/filtered_hotels.csv')
                    selected_hotel= pd.read_csv(r'raw_files/selected_restaurant.csv')
                    results=self.alternate_response(df,selected_hotel)
                    if len(results)==0:
                        bot_msg='Sorry could not find any other restaurant with the criteria'
                    else:
                        results.to_csv(r'raw_files/selected_restaurant.csv',index=False)
                        response_restarent=results.iloc[0].to_dict()
                        bot_msg='Would you like to try {} '.format(response_restarent['restaurantname'])
                if intent =='request':
                    details=self.additional_details(message)
                    details_Str=','.join(details)
                    bot_msg ='The requested details are {}'.format(details_Str)
                    self.current_state='Information'
                if intent == 'no dialog':
                    bot_msg ='Sorry did not get that try "start over" if you want to find a new restaurant\n'
                if config['caps'] == 'true':
                    bot_msg=bot_msg.upper()
                self.prevbotresp=bot_msg
                return bot_msg
            elif self.current_state=='Information':
                if intent =='request':
                    details=self.additional_details(message)
                    details_Str=','.join(details)
                    bot_msg ='The requested details are {}'.format(details_Str)
                    self.current_state='Information'
                if (intent in['bye','thankyou'] or 'bye' in message):
                    bot_msg ='Bye See you\n'
                    self.current_state='hello' 
                if (intent in['ack','affirm','confirm']):
                    bot_msg ='Cool is there anything else you need\n'
                if intent == 'no dialog':
                    bot_msg ='Sorry did not get that try "start over" if you want to find a new restaurant\n'
                if config['caps'] == 'true':
                    bot_msg=bot_msg.upper()
                self.prevbotresp=bot_msg
                return bot_msg
        
    
    def state_transition(self,state):
        self.current_state=state
    
    def class_bot(self):
        flag= True
        msg = input("Hi I'm a bot is anything i can help you with:")
        while flag:
            try:
                bot_response=self.bot_response(msg)
                msg= input(bot_response)
                if msg == 'quit':
                    if os.path.exists(r'raw_files/selected_restaurant.csv'):
                        os.remove(r'raw_files/selected_restaurant.csv')
                    if os.path.exists(r'raw_files/filtered_hotels.csv'):
                        os.remove(r'raw_files/filtered_hotels.csv')
                    if os.path.exists(r'raw_files/closest_results.csv'):
                        os.remove(r'raw_files/closest_results.csv')    
                    flag= False
            except:
                if os.path.exists(r'raw_files/selected_restaurant.csv'):
                        os.remove(r'raw_files/selected_restaurant.csv')
                if os.path.exists(r'raw_files/filtered_hotels.csv'):
                        os.remove(r'raw_files/filtered_hotels.csv')
                if os.path.exists(r'raw_files/closest_results.csv'):
                        os.remove(r'raw_files/closest_results.csv')                  
            