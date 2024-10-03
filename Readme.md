The aim of this repository is to create a working dialog system along with classifiers that are required for the bot to identify the intents

Using the bot:
    1) Download the files or clone the repository
    2) use the requirements.txt to download required packages(set up a venv if required)
    3) extract the raw_files
    4) Run the app.py file using the command py app.py

1a Text classification
    We used for methods for text classification majority base line, rule based base line and two machine learning algorithms (Logistic reggression,Descsion Tree)

    Results of the classifiers

    1) Majority based: Accuracy .40
    2) Rule based: Accuracy .83
    3) Logistic regression: Accuracy .98
    4) Decision Tree: Accuracy .98 


    Results of the classifiers after removing duplicates

    1) Majority based: Accuracy .55
    2) Rule based: Accuracy .77
    3) Logistic regression: Accuracy .92
    4) Decision Tree: Accuracy .89 

1b Dialog system:
    In this part we have implement a dialog system to enable the user to search for restaurants in the city

    Implementations

    1) We have used logistic regression to classify the users text input after converting the messages to lower cases.However due to the varied ways user can type messages we are processing the dialog acts hi, inform ,no dialog(None) , requests.
    2) After this we remove the stop words using spacy in the message to extract keywords using keybert(the best performing among yake-NLTK,Rake).
    3) We use these keywords to identify the restaurants specified by the user in the csv file provide and return a random restaurant from the selection.
    4) If the system is not able to find an exact match it uses the levenshtein distance to provide the closest value.
    5) We store the selected restaurant and the filtered results as a CSV file to help the user with further questions and providing with alternative response
    6) Implemented a fully functional state transition flow for the Dialog system

1c Reasoning and configurations

    1) We have implemented the reasoning part as described in the black board
        1a) We added the parameters for the new columns at random
        1b) We again used keybert for keyword extraction
        1c) Then using the keywords we filtered the results based on the conditons
        1d) We then added the reasoning for the result with the bot message
    2) for the configurations we have used
        2a) Allow preferences to be stated in random order or not
        2b) Allow preferences to be stated in a single utterance only, or in multiple utterances with one preference per utterance only, or without restrictions (any number of utterances and any number of preferences per utterance)
        2c) Allow dialog restarts or not
        2d) OUTPUT IN ALL CAPS OR NOT