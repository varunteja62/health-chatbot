import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ChatbotModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000,
            min_df=1,  # Include terms that appear in at least 1 document
            strip_accents='unicode',
            token_pattern=r'[a-zA-Z]+',  # Only keep alphabetic tokens
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.questions = []
        self.answers = []
        self.load_mediqa_dataset()
        self.train_model()

    def preprocess_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Convert to lowercase and remove special characters
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token.isalpha() and token not in self.stop_words]
        
        # Only return if we have valid tokens
        processed_text = ' '.join(tokens)
        return processed_text if processed_text else ""

    def load_mediqa_dataset(self):
        try:
            # First try to load the RQE dataset
            tree = ET.parse('MEDIQA2019-master/MEDIQA_Task2_RQE/MEDIQA2019-Task2-RQE-TrainingSet-AMIA2016.xml')
            root = tree.getroot()

            # Extract pairs from the XML
            for pair in root.findall('.//pair'):
                chq = pair.find('chq').text  # Consumer health question
                faq = pair.find('faq').text  # FAQ question
                label = pair.get('value', '0')  # Get the entailment label
                
                if chq and faq and label == '1':  # Only use positive entailment pairs
                    processed_chq = self.preprocess_text(chq)
                    if processed_chq:  # Only add non-empty questions
                        self.questions.append(processed_chq)
                        self.answers.append(faq)

            if not self.questions:  # If no questions were loaded, use fallback
                raise Exception("No valid questions found in the dataset")

        except Exception as e:
            print(f"Error loading RQE dataset: {e}")
            print("Loading fallback Q&A pairs...")
            # Fallback to basic Q&A pairs
            qa_pairs = [
                ("what are the symptoms of headache", 
                 "Common headache symptoms include pain in the head or neck, pressure in the head, sensitivity to light and sound, and sometimes nausea."),
                ("how to treat fever", 
                 "To treat fever: 1. Rest 2. Stay hydrated 3. Take over-the-counter fever reducers 4. Use a cool compress 5. Consult a doctor if fever is high or persistent."),
                ("what causes high blood pressure", 
                 "High blood pressure can be caused by: age, family history, poor diet, lack of exercise, stress, smoking, obesity, and certain medical conditions."),
                ("diabetes symptoms treatment", 
                 "Diabetes symptoms include increased thirst, frequent urination, hunger, fatigue, and blurred vision. Treatment includes medication, diet control, and regular exercise."),
                ("common cold remedies", 
                 "Common cold remedies include: rest, staying hydrated, over-the-counter decongestants, throat lozenges, and nasal sprays. Symptoms usually resolve within 7-10 days.")
            ]
            
            for question, answer in qa_pairs:
                processed_q = self.preprocess_text(question)
                if processed_q:
                    self.questions.append(processed_q)
                    self.answers.append(answer)

        print(f"Loaded {len(self.questions)} question-answer pairs")

    def train_model(self):
        if not self.questions:
            raise ValueError("No questions available for training")
        
        try:
            self.question_vectors = self.vectorizer.fit_transform(self.questions)
            print("Model trained successfully")
        except Exception as e:
            print(f"Error training model: {e}")
            raise

    def get_response(self, user_input):
        if not self.questions:
            return "I'm sorry, I couldn't load the medical dataset. Please try again later."

        try:
            # Preprocess user input
            processed_input = self.preprocess_text(user_input)
            if not processed_input:
                return "Please enter a valid medical question with some medical terms."
            
            # Vectorize user input
            input_vector = self.vectorizer.transform([processed_input])
            
            # Calculate similarities
            similarities = cosine_similarity(input_vector, self.question_vectors)
            
            # Find top 3 most similar questions
            top_indices = similarities[0].argsort()[-3:][::-1]
            max_similarity = similarities[0][top_indices[0]]
            
            # Return best answer if similarity is above threshold
            if max_similarity > 0.2:  # Lowered threshold for better matching
                return self.answers[top_indices[0]]
            else:
                return "I'm sorry, I don't have enough information about that medical condition. Please try rephrasing your question or consult a healthcare professional."
        except Exception as e:
            print(f"Error processing response: {e}")
            return "I'm sorry, there was an error processing your question. Please try again."

# Create a global instance of the chatbot model
chatbot_model = ChatbotModel() 