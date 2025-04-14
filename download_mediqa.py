import pandas as pd
import requests
import io

def download_mediqa():
    print("Creating sample MEDIQA-style dataset...")
    
    # Create a sample dataset with medical QA pairs
    data = {
        'question': [
            'What are the symptoms of diabetes?',
            'How can I lower my blood pressure naturally?',
            'What causes migraines?',
            'How to treat a common cold?',
            'What are the signs of heart attack?',
            'How to manage anxiety?',
            'What are the symptoms of COVID-19?',
            'How to improve sleep quality?',
            'What causes high cholesterol?',
            'How to treat allergies?'
        ],
        'answer': [
            'Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, and slow-healing sores. If you experience these symptoms, consult a healthcare provider for proper diagnosis.',
            'Natural ways to lower blood pressure include reducing sodium intake, regular exercise, maintaining a healthy weight, limiting alcohol, managing stress, and following the DASH diet. Regular monitoring and consultation with a healthcare provider is important.',
            'Migraines can be triggered by stress, hormonal changes, certain foods, bright lights, loud noises, lack of sleep, and weather changes. Treatment includes pain relievers, preventive medications, and lifestyle modifications.',
            'To treat a common cold, get plenty of rest, stay hydrated, use over-the-counter decongestants, and try saline nasal drops. Symptoms typically resolve within 7-10 days. Seek medical attention if symptoms worsen.',
            'Heart attack signs include chest pain/pressure, shortness of breath, pain in arms/neck/jaw, nausea, cold sweat, and lightheadedness. If you suspect a heart attack, call emergency services immediately.',
            'Anxiety can be managed through regular exercise, meditation, deep breathing exercises, adequate sleep, and therapy. Consider consulting a mental health professional for personalized treatment options.',
            'COVID-19 symptoms include fever, cough, shortness of breath, fatigue, loss of taste/smell, body aches, and headache. If you experience symptoms, get tested and follow isolation guidelines.',
            'Improve sleep quality by maintaining a regular sleep schedule, creating a relaxing bedtime routine, avoiding screens before bed, limiting caffeine, and ensuring a comfortable sleep environment.',
            'High cholesterol can be caused by poor diet, lack of exercise, smoking, obesity, and genetics. Treatment includes lifestyle changes and sometimes medication. Regular cholesterol checks are important.',
            'Allergy treatments include avoiding triggers, taking antihistamines, using nasal sprays, and keeping indoor air clean. For severe allergies, consider immunotherapy under medical supervision.'
        ]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save the dataset
    print("Saving dataset...")
    df.to_csv('mediqa2019.csv', index=False)
    print("Dataset saved as mediqa2019.csv")

if __name__ == "__main__":
    download_mediqa() 