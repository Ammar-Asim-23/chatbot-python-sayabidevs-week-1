import os
import random 
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf


nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('server/src/chatbot/intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words=sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('server/src/chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('server/src/chatbot/classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1    
    training.append([bag, output_row])

random.shuffle(training)

trainx = [item[0] for item in training]
trainy = [item[1] for item in training]

#defining the model layers and optmizer             
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainx[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainy[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(trainx), np.array(trainy), epochs=200, batch_size=5, verbose=1)
model.save('server/src/chatbot/chatbot_model.model')

print('Done')



