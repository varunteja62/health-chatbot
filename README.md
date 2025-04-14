# Medical QA Chatbot

A chatbot that answers medical questions using the MEDIQA2019 dataset. The chatbot uses TF-IDF vectorization and cosine similarity to find the most relevant answers to user queries.

## Setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the MEDIQA2019 dataset and place it in the project directory:
   - `MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet1-LiveQAMed.xml`
   - `MEDIQA2019-master/MEDIQA_Task3_QA/MEDIQA2019-Task3-QA-TrainingSet2-Alexa.xml`

## Running the Chatbot

1. Start the Flask server:
   ```
   python mediqa_chatbot.py
   ```
2. Open your web browser and navigate to `http://localhost:5000`
3. Start asking medical questions!

## Features

- Simple and intuitive web interface
- Real-time responses to medical questions
- Based on the MEDIQA2019 dataset
- Uses TF-IDF vectorization and cosine similarity for answer matching

## Disclaimer

This chatbot is for demonstration purposes only and should not be used as a substitute for professional medical advice. Always consult with qualified healthcare providers for medical concerns. 