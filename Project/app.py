from flask import Flask, request, render_template, jsonify
from chatbot.chatbot import get_response, predict_class
import json

app= Flask(__name__)

intents = json.loads(open('Project/chatbot/intents.json').read())

@app.route('/', methods=['GET', 'POST'])
def home():
    chatbot_response = ''
    
    if request.method == 'POST':
        user_input = request.form['user_input']
        chatbot_response = get_response(predict_class(user_input), intents)
        return jsonify({'response': chatbot_response})
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)