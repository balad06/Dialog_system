import pickle
from rake_nltk import Rake
import nltk
from keybert import KeyBERT
import pandas as pd
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
        return 'text'
    def keyextractions(self,msg):
        # text = "Your text here"
        # r = Rake()
        # r.extract_keywords_from_text(msg)
        # keywords = r.get_ranked_phrases()
        kw_model = KeyBERT()
        text = msg
        keywords = kw_model.extract_keywords(text)
# print(keywords) 
        return keywords
    def distance():
        return 'distance'
    def get_results(self,message,intent,keywords):
        df =pd.read_csv(r'raw_files/restaurant_info.csv')
        # df.index.name='restaurantname'
        matched_restarent =pd.DataFrame()
        for keyword in keywords:
            keyword = keyword[0]
            for column in df.columns:
                matches = df[df[column].str.contains(keyword, case=False, na=False)]
                if not matches.empty:
                    matched_restarent = pd.concat([matched_restarent, matches])
        return matched_restarent
    
    def bot_response(self,message):
        message = message.lower()
        intent=self.textclassfier(message)
        print(intent)
        keywords=self.keyextractions(message)
        print(keywords)
        results=self.get_results(message,intent,keywords)
        # print(results)
        if len(results)>0:
            response_restarent=results.sample().to_dict()
            print(response_restarent)
        else:
            response_restarent='empty'
        return response_restarent
    def class_bot(self):
        flag= True
        msg = input("Hi I'm a bot is anything i can help you with:")
        while flag:
            bot_response=self.bot_response(msg)
            msg= input(bot_response)
            if msg == 'quit':
                flag= False