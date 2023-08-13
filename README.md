# chatbot-python-sayabidevs-week-1
Topic: A Cafe Chatbot
Tags: greetings, goodbye, name, product list, recommendations, payments,
delivery, wifi, parking, location, phone, website, working hours
(https://github.com/Ammar-Asim-23/chatbot-python-sayabidevs-week-1)
In this approach, we utilize a traditional approach which includes training a
tensorflow model and then using it to generate response. Firstly, we first initialize
a WordNetLemmatizer and intents.json through the loads functions in the json
module. Then we detect all the tags, patterns and responses in the intent file. We
then tokenize and lemmatize each word in patterns. We then dump the words
and their classes in pickle files. After this we train a tensorflow model and save it.
Lastly we use the model to predict the patterns and give responses.
