import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string

class MEDIQA2019Chatbot:
    def __init__(self):
        # Initialize NLTK components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load and preprocess the dataset
        self.qa_pairs = self._load_dataset()
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Vectorize questions
        self.question_vectors = self.vectorizer.fit_transform(
            [self._preprocess_text(q) for q in self.qa_pairs['question']]
        )
    
    def _load_dataset(self):
        """Load and parse MEDIQA2019 dataset"""
        qa_pairs = {'question': [], 'answer': []}
        
        # Load training sets
        training_files = [
            'MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml',
            'MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml'
        ]
        
        for file_path in training_files:
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                for qa_pair in root.findall('.//QAPair'):
                    question = qa_pair.find('Question').text
                    answer = qa_pair.find('Answer').text
                    
                    if question and answer:
                        qa_pairs['question'].append(question)
                        qa_pairs['answer'].append(answer)
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        return pd.DataFrame(qa_pairs)
    
    def _preprocess_text(self, text):
        """Preprocess text for similarity matching"""
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                 if word not in self.stop_words and word not in string.punctuation]
        
        return ' '.join(tokens)
    
    def get_response(self, user_query):
        """Get response for user query"""
        # Preprocess the query
        processed_query = self._preprocess_text(user_query)
        
        if not processed_query:
            return "I'm sorry, I couldn't understand your question. Could you please rephrase it?"
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.question_vectors)[0]
        
        # Get the most similar question
        max_sim_idx = np.argmax(similarities)
        max_similarity = similarities[max_sim_idx]
        
        # Return response if similarity is above threshold
        if max_similarity > 0.2:
            return self.qa_pairs.iloc[max_sim_idx]['answer']
        else:
            return "I'm sorry, I don't have enough information to answer that question. Please consult a healthcare professional for medical advice."

# Create Flask app for the chatbot
from flask import Flask, request, jsonify, render_template
from model import chatbot_model

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'response': 'Please provide a message.'}), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({'response': 'Please enter a question.'}), 400
        
        response = chatbot_model.get_response(user_message)
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({'response': 'An error occurred while processing your request.'}), 500

if __name__ == "__main__":
    print("============================================")
    print("      Medical QA Chatbot")
    print("============================================")
    print("Loading model and starting server...")
    try:
        # This will initialize the model and load the dataset
        response = chatbot_model.get_response("test")
        print("Model loaded successfully!")
        print("Starting Flask web server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to quit the application.")
        print("============================================")
        app.run(debug=True)
    except Exception as e:
        print(f"Error starting application: {e}")
        print("============================================") 