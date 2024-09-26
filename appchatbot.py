import pickle
import nltk
from keybert import KeyBERT
import pandas as pd
import spacy
import os
import Levenshtein
class Chatbot:
    def textclassfier(self,msg):
        with open(r'raw_files/logisticmodelnodupe.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open(r'raw_files/logisticvectorizernodupe.pkl', 'rb') as f:
            cv = pickle.load(f)
        msg_cv=cv.transform([msg])
        result=lr.predict(msg_cv)
        return result
    def prepresponse():
        #Use for preparping responses for diffrent dialog acts
        return 'text'
    def keyextractions(self,msg):
        nlp = spacy.load("en_core_web_sm")#REmove stop words
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
    
    def closest_results(self,keywords):
        df =pd.read_csv(r'raw_files/restaurant_info.csv')
        for keyword in keywords:
            keyword = keyword[0]
            #to avoid unwanted loops
            if keyword!= 'hotels' and keyword!='restaurants' and keyword!= 'hotel' and keyword!='restaurant' and keyword!='food':
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
    def get_results(self,message,intent,keywords):
        df =pd.read_csv(r'raw_files/restaurant_info.csv')
        matched_restarent =pd.DataFrame(columns=df.columns)
        columns=[]
        if intent in ['hi','inform','no dialog','request']:
            for keyword in keywords:
                keyword = keyword[0]
                #to avoid unwanted loops
                if keyword!= 'hotels' and keyword!='restaurants' and keyword!= 'hotel' and keyword!='restaurant' and keyword!='food':
                    for column in ['pricerange','area','food']:
                        matches = df[df[column].str.contains(keyword, case=False, na=False)]
                        if not matches.empty:
                            if matched_restarent.empty:
                                matched_restarent=matches
                                columns.append(column)
                            else:
                                columns.append(column)
                                matches = matches.rename(columns={'restaurantname_x': 'restaurantname', 'area_x': 'area','pricerange_x':'pricerange','food_x':'food','addr_x':'addr','postcode_x':'postcode'})
                                matched_restarent = matched_restarent.rename(columns={'restaurantname_x': 'restaurantname', 'area_x': 'area','pricerange_x':'pricerange','food_x':'food','addr_x':'addr','postcode_x':'postcode'})
                                matched_restarent = pd.merge(matched_restarent, matches, how='inner',on=['restaurantname','pricerange','area','food','phone','addr','postcode'])

            matched_restarent.to_csv(r'raw_files/filtered_hotels.csv',index=False)
        return matched_restarent
    
    def bot_response(self,message):
        message = message.lower()
        bot_msg=''
        intent=self.textclassfier(message)
        keywords=self.keyextractions(message)
        results=self.get_results(message,intent,keywords)
        if len(results)>0:
            selected_restaurant=results.sample(n=1)
            response_restarent=selected_restaurant.iloc[0].to_dict()
            selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
            bot_msg='Would you like to try {}'.format(response_restarent['restaurantname'])
        else:
            response_restarent='empty'
            results=self.closest_results(keywords)
            if len(results)==0:
                bot_msg='Could not find a match wanna look for something else'
            else:
                selected_restaurant=results.sample(n=1)
                response_restarent=selected_restaurant.iloc[0].to_dict()
                selected_restaurant.to_csv(r'raw_files/selected_restaurant.csv',index=False)
           
                if response_restarent['min_dist']=='lv_pricerange': 
                    bot_msg= 'Did you mean {} Here is a restaurant for the result {}'.format(response_restarent['pricerange'],response_restarent['restaurantname'])
                if response_restarent['min_dist']=='lv_area':
                    bot_msg= 'Did you mean {} Here is a restaurant for the result {}'.format(response_restarent['area'],response_restarent['restaurantname'])
                if response_restarent['min_dist']=='lv_food':
                    bot_msg= 'Did you mean {} Here is a restaurant for the result {}'.format(response_restarent['food'],response_restarent['restaurantname'])
        return bot_msg
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
                flag= False