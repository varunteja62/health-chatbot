"""
Medical Chatbot - Main Entry Point
---------------------------------
This script runs both the machine learning model and Flask application.
"""

# Import the ML model and Flask app
from model import chatbot_model
from flask_app import app

if __name__ == "__main__":
    print("============================================")
    print("      Medical Chatbot Application")
    print("============================================")
    print("Model loaded successfully!")
    print("Starting Flask web server...")
    print("Open your browser and go to: http://127.0.0.1:5000/")
    print("Press Ctrl+C to quit the application.")
    print("============================================")
    
    # Run the Flask application
    app.run(debug=True)