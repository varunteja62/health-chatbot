from flask import Flask, request, jsonify, render_template
from model import chatbot_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please enter a question.'})
    
    try:
        response = chatbot_model.get_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'response': 'Sorry, there was an error processing your request.'})

if __name__ == '__main__':
    print("Medical Chatbot is starting...")
    print("Open your browser and navigate to http://127.0.0.1:5000/")
    app.run(debug=True) 